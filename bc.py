#!/usr/bin/env python3
# Team-conditioned behavior cloning; see README.md for usage.
import argparse
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import features
from model import HourFormer, count_parameters

FIELDS  = ("market_op", "unit_op", "item", "qty")
COUNTS  = tuple(f"{field}_{kind}" for field in FIELDS for kind in ("correct", "count")) + ("hour_correct",)
READERS = ThreadPoolExecutor(16)


def append_jsonl(path, rows):
    with open(path, "a") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def save_checkpoint(checkpoint, directory, step, keep):
    # Save atomically, update latest.pt, and prune old checkpoints.
    directory = Path(directory)
    target = directory / f"ckpt-{step:06d}.pt"
    temporary = directory / f"ckpt-{step:06d}.tmp"
    with temporary.open("wb") as stream:
        torch.save(checkpoint, stream)
    temporary.replace(target)
    shutil.copyfile(target, temporary)
    temporary.replace(directory / "latest.pt")
    numbered = sorted((p for p in directory.glob("ckpt-*.pt") if p.stem[5:].isdigit()), key=lambda p: int(p.stem[5:]))
    for old in numbered[:-keep]:
        old.unlink()


def export_team(argv):
    # Export a player with its team embedding folded into the bias.
    parser = argparse.ArgumentParser(description="Export a player with its team embedding folded into the bias.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("meta", type=Path, help="the corpus meta.json listing the teams")
    parser.add_argument("out", type=Path)
    parser.add_argument("--team", help="team label to export (default: the corpus's finetune_team)")
    args = parser.parse_args(argv)

    meta = json.loads(args.meta.read_text())
    labels, label = meta["teams"], args.team or meta["finetune_team"]
    index = labels.index(label)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = {key: value for key, value in checkpoint["config"].items() if key != "n_teams"}
    assert checkpoint["config"]["n_teams"] == len(labels), (checkpoint["config"], len(labels))

    weights = {key: value.clone() for key, value in checkpoint["model"].items() if not key.startswith("team_embed")}
    weights["global_mlp.2.bias"] += checkpoint["model"]["team_embed.weight"][index]

    plain, teamed = HourFormer(**config).eval(), HourFormer(**checkpoint["config"]).eval()
    plain.load_state_dict(weights, strict=True)
    teamed.load_state_dict(checkpoint["model"], strict=True)
    g = torch.randn(4, teamed.global_mlp[0].in_features)
    assert torch.allclose(plain.global_mlp(g), teamed.global_mlp(g) + teamed.team_embed.weight[index], atol=1e-5)

    torch.save({"model": weights, "config": config, "step": checkpoint["step"], "team": {"index": index, "label": label}}, args.out)
    print(args.out)


def load_pack(directory):
    return {key: torch.from_numpy(np.load(Path(directory) / f"{key}.npy", mmap_mode="c")) for key in ("tiles", "units", "globals", "tokens", "n_units", "margin", "team")}


def read_batch(data, rows, device):
    # Read rows concurrently and transfer them to the device.
    chunks = rows.chunk(16)
    batch = {key: torch.cat(list(READERS.map(lambda chunk: data[key][chunk], chunks))) for key in data}
    exists = torch.arange(features.N_UNITS)[None] < batch["n_units"][:, None].long()
    return tuple(t.to(device, non_blocking=True) for t in (batch["tiles"], batch["units"], batch["globals"].float(), batch["tokens"].long(), exists, batch["margin"], batch["team"].long()))


@torch.no_grad()
def evaluate(model, val, rows, device, pooled):
    # Validate actions and values, optionally without team conditioning.
    model.eval()
    sums = {key: 0.0 for key in COUNTS + ("value_se",)}
    nll = pooled_nll = 0.0
    team_nll, team_n = torch.zeros(model.n_teams, dtype=torch.float64), torch.zeros(model.n_teams, dtype=torch.float64)

    for start in range(0, len(rows), 512):
        tiles, units, globals_, tokens, exists, margin, team = read_batch(val, rows[start:start + 512], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.log_probs(tiles, units, globals_, tokens, exists, team=team)
            if pooled:
                pooled_nll -= model.log_probs(tiles, units, globals_, tokens, exists)["log_prob"].sum().item()
        nll -= output["log_prob"].sum().item()
        team_nll.index_add_(0, team.cpu(), -output["log_prob"].double().cpu())
        team_n.index_add_(0, team.cpu(), torch.ones(len(team), dtype=torch.float64))
        for key in COUNTS:
            sums[key] += int(output[key])
        sums["value_se"] += float((output["value"] - margin).square().sum())

    model.train()
    n = len(rows)
    result = {f"val_{field}_acc": sums[f"{field}_correct"] / max(1, sums[f"{field}_count"]) for field in FIELDS}
    result.update(val_hour_acc=sums["hour_correct"] / n, val_nll=nll / n, val_value_rmse=math.sqrt(sums["value_se"] / n))
    if pooled:
        result["val_nll_pooled"] = pooled_nll / n
    result.update({f"val_nll_team{i}": float(team_nll[i] / team_n[i]) for i in range(model.n_teams) if team_n[i] > 0})
    return result


def main():
    parser = argparse.ArgumentParser(description="Team-conditioned behavior cloning; see README.md for usage.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="packed training corpus")
    parser.add_argument("--rows", default="rows.npy", help="file in --data listing the rows to train on")
    parser.add_argument("--val", type=Path, required=True, help="packed validation corpus")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init", type=Path, help="checkpoint to start from")
    parser.add_argument("--resume", type=Path, help="latest.pt of a stopped run, continued with the same settings (the paths may differ)")
    parser.add_argument("--steps", type=int, default=114000)
    parser.add_argument("--minibatch", type=int, default=256, help="rows per minibatch")
    parser.add_argument("--accum", type=int, default=7, help="minibatches per step, over all GPUs")
    parser.add_argument("--lr", type=float, default=1e-3, help="peak learning rate; linear warmup, then cosine decay to a tenth")
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient norm clip")
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--team-drop", type=float, default=0.1, help="fraction of the training hours shown without their team")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--val-stride", type=int, default=100, help="validate on every Nth validation hour")
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=0.0, help="minutes after which to save and stop (0: none)")
    parser.add_argument("--device", default="cuda", help="CUDA device; training needs a GPU")
    args = parser.parse_args()

    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
        args.device = f"cuda:{os.environ['LOCAL_RANK']}"
    device = torch.device(args.device)
    torch.cuda.set_device(device.index or 0)

    resumed = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if resumed:
        changed = [key for key, value in resumed["args"].items() if key not in ("data", "rows", "val", "init", "out", "resume", "time_limit", "device") and str(value) != str(getattr(args, key))]
        if changed:
            parser.error(f"--resume continues a run with the same settings; changed: {changed}")
    first_step = resumed["step"] + 1 if resumed else 1
    torch.manual_seed(args.seed * 1000 + rank + 7919 * (first_step - 1))
    args.out.mkdir(parents=True, exist_ok=True)

    train = load_pack(args.data)
    labels = json.loads((args.data / "meta.json").read_text())["teams"]
    rows = torch.from_numpy(np.load(args.data / args.rows))[rank::world]
    val = load_pack(args.val) if rank == 0 else None
    val_rows = torch.arange(len(val["n_units"]))[::args.val_stride] if rank == 0 else None

    config = {"d_model": args.d_model, "n_layers": args.layers, "n_heads": args.heads, "n_dec_layers": args.dec_layers, "ffn": args.ffn, "dropout": args.dropout, "n_teams": len(labels)}
    model = HourFormer(**config).to(device)
    if resumed:
        model.load_state_dict(resumed["model"])
    elif args.init:
        missing, unexpected = model.load_state_dict(torch.load(args.init, map_location=device, weights_only=False)["model"], strict=False)
        if unexpected or [key for key in missing if not key.startswith("team_embed")]:
            raise RuntimeError(f"--init {args.init} does not fit the network: missing {missing}, unexpected {unexpected}")
    model.train()
    wrapped = DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model

    decay = [p for name, p in model.named_parameters() if p.ndim >= 2 and "norm" not in name]
    no_decay = [p for name, p in model.named_parameters() if p.ndim < 2 or "norm" in name]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    if resumed:
        opt.load_state_dict(resumed["optimizer"])

    if rank == 0:
        print(f"{count_parameters(model)} parameters, {len(rows) * world} training rows of {len(train['n_units'])} hours, {len(val_rows)} of {len(val['n_units'])} validation hours, {len(labels)} teams" + (f", resumed at step {first_step}" if resumed else ""), flush=True)
        if not resumed:
            (args.out / "config.json").write_text(json.dumps({**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "world": world, "model": config}, indent=1))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, args.steps - args.warmup))))

    keys = COUNTS + ("nll", "value_loss", "n", "grad_norm", "steps")
    sums = torch.zeros(len(keys) + 1, dtype=torch.float64, device=device)  # Last entry: timeout flag.
    started = time.time()
    timed_out = False
    for step in range(first_step, args.steps + 1):
        for param_group in opt.param_groups:
            param_group["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            tiles, units, globals_, tokens, exists, margin, team = read_batch(train, rows[torch.randint(len(rows), (args.minibatch,))], device)
            if args.team_drop > 0:
                team = torch.where(torch.rand(team.shape, device=device) < args.team_drop, -1, team)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = wrapped(tiles, units, globals_, tokens, exists, team=team)
            nll = -output["log_prob"].mean()
            value_loss = (output["value"] - margin).square().mean()
            ((nll + args.value_coef * value_loss) / args.accum).backward()
            sums[:len(COUNTS)] += torch.stack([torch.as_tensor(output[key], device=device) for key in COUNTS]).double()
            sums[len(COUNTS):len(COUNTS) + 3] += torch.stack([nll.detach() * args.minibatch, value_loss.detach() * args.minibatch, torch.tensor(float(args.minibatch), device=device)]).double()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        sums[len(COUNTS) + 3] += grad_norm.double()
        sums[len(COUNTS) + 4] += 1

        if step % args.log_every == 0:
            sums[-1] = float(args.time_limit > 0 and time.time() - started > 60 * args.time_limit)
            if world > 1:
                dist.all_reduce(sums)
            total = dict(zip(keys, sums.tolist()))
            timed_out = sums[-1].item() > 0
            row = {"step": step, "lr": lr_at(step), "elapsed": round(time.time() - started), "nll": total["nll"] / total["n"], "value_loss": total["value_loss"] / total["n"], "grad_norm": total["grad_norm"] / total["steps"],
                   "hour_acc": total["hour_correct"] / total["n"], "peak_gib": round(torch.cuda.max_memory_reserved(device) / 2 ** 30, 2), **{f"{field}_acc": total[f"{field}_correct"] / max(1.0, total[f"{field}_count"]) for field in FIELDS}}
            if step % args.eval_every == 0 and rank == 0:
                row.update(evaluate(model, val, val_rows, device, pooled=args.team_drop > 0))
            if rank == 0:
                append_jsonl(args.out / "metrics.jsonl", [row])
                print(json.dumps({key: round(value, 4) if isinstance(value, float) else value for key, value in row.items()}), flush=True)
            sums.zero_()

        if (step % args.ckpt_every == 0 or step == args.steps or timed_out) and rank == 0:
            save_checkpoint({"model": model.state_dict(), "config": config, "step": step, "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "optimizer": opt.state_dict()}, args.out, step, args.keep_checkpoints)
        if world > 1 and (step % args.eval_every == 0 or timed_out):
            dist.barrier()
        if timed_out:
            if rank == 0:
                print(f"time limit reached, stopped at step {step}; continue with --resume {args.out / 'latest.pt'}", flush=True)
            break

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    if sys.argv[1:2] == ["export"]:
        export_team(sys.argv[2:])
    else:
        main()
