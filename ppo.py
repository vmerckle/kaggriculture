#!/usr/bin/env python3
# PPO against frozen opponents on one GPU; see README.md.
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # Reduce rollout fragmentation.
import torch

from bc import append_jsonl, save_checkpoint
from workers import Policy, Workers, load_model, load_opponents, play

MEASURES = ("policy_loss", "value_loss", "entropy", "clipfrac", "approx_kl")


def gae(rewards, values, lam):
    # Return GAE advantages and returns with gamma 1.
    advantage = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for hour in reversed(range(len(rewards))):
        next_value = values[hour + 1] if hour + 1 < len(rewards) else torch.zeros_like(values[0])
        last = rewards[hour] + next_value - values[hour] + lam * last
        advantage[hour] = last
    return advantage, advantage + values


def schedule(args, iteration, pool, opponent_names):
    # Schedule newest losses, then fresh games; return games and replay count.
    fresh = []
    for index in range(args.games):
        seed = args.seed + (iteration - 1) * args.games + index
        fresh.append((seed, seed % 2, opponent_names[((iteration - 1) * (args.games // 2) + index // 2) % len(opponent_names)]))
    replayed = [pool.pop() for _ in range(min(len(pool), int(args.replay_frac * args.games)))]
    return replayed + fresh[len(replayed):], len(replayed)


def update(model, opt, flat, args, progress):
    # Run one PPO pass and return metrics and step counts.
    device = flat["adv"].device
    count = len(flat["adv"])
    permutation = torch.randperm(count, device=device)
    starts = list(range(0, count, args.minibatch))
    totals = torch.zeros(len(MEASURES), device=device)
    steps = skipped = 0
    stopped = False

    for first in range(0, len(starts), args.accum):
        chunk = starts[first:first + args.accum]
        opt.zero_grad(set_to_none=True)
        current = torch.zeros(len(MEASURES), device=device)
        for start in chunk:
            rows = permutation[start:start + args.minibatch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(flat["tiles"][rows], flat["units"][rows], flat["globals"][rows], flat["tokens"][rows], flat["exists"][rows], temperature=args.temperature)
            log_ratio = output["log_prob"] - flat["log_prob"][rows]
            ratio = log_ratio.exp()
            advantage = flat["adv"][rows]
            policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - args.clip, 1 + args.clip) * advantage).mean()
            value_loss = (output["value"] - flat["ret"][rows]).square().mean()
            loss = policy_loss + args.value_coef * value_loss
            (loss / len(chunk)).backward()
            with torch.no_grad():
                measures = [policy_loss, value_loss, output["entropy"].mean(), ((ratio - 1).abs() > args.clip).float().mean(), (ratio - 1 - log_ratio).mean()]
                current += torch.stack([m.detach() for m in measures]).float() / len(chunk)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            skipped += 1
            continue
        progress["total_updates"] += 1
        for param_group in opt.param_groups:
            param_group["lr"] = args.lr * min(1.0, progress["total_updates"] / max(1, args.warmup))
        opt.step()
        totals += current
        steps += 1
        if current[-1] > args.target_kl:
            stopped = True
            break
    return dict(zip(MEASURES, (totals / max(1, steps)).tolist())), steps, skipped, stopped


def main():
    parser = argparse.ArgumentParser(description="PPO against frozen opponents on one GPU; see README.md.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", required=True, help="the player to start from")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--opponents", required=True, help="comma-separated frozen:PATH or frozen+RULES:PATH, rotating over the games")
    parser.add_argument("--resume", type=Path, help="OUT/latest.pt of a stopped run")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--games", type=int, default=688, help="games per iteration")
    parser.add_argument("--minibatch", type=int, default=512, help="hours per minibatch")
    parser.add_argument("--accum", type=int, default=44, help="minibatches per optimizer step")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=20, help="optimizer steps of linear learning-rate warmup")
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=0.05, help="end the iteration's pass once a step's mean approximate KL exceeds this")
    parser.add_argument("--temperature", type=float, default=0.5, help="the learner's sampling temperature; the update uses the same tempered policy")
    parser.add_argument("--replay-frac", type=float, default=0.25)
    parser.add_argument("--replay-max", type=int, default=2, help="how many times a lost game can be replayed")
    parser.add_argument("--workers", type=int, default=12, help="engine processes")
    parser.add_argument("--worker-sets", type=int, default=2, help="the workers are split into sets whose engine steps overlap the other sets' decoding")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt-every", type=int, default=1)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--device", default="cuda", help="CUDA device; training needs a GPU")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device.index or 0)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    opponent_names = args.opponents.split(",")

    model, init = load_model(args.init, device)  # Stays in eval mode: no dropout, so rollout and update log-probabilities agree.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-6, weight_decay=0.0)
    progress = {"iteration": 0, "total_updates": 0}
    pool, replay_counts = [], Counter()
    pool_path = args.out / "replay_pool.json"
    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resumed["model"])
        opt.load_state_dict(resumed["optimizer"])
        progress = {"iteration": int(resumed["iteration"]), "total_updates": int(resumed["total_updates"])}
        if pool_path.exists():
            saved = json.loads(pool_path.read_text())
            if saved["iteration"] == progress["iteration"]:
                pool, replay_counts = [tuple(game) for game in saved["pool"]], Counter({tuple(game): count for game, count in saved["replays"]})

    (args.out / "config.json").write_text(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=1))
    print(f"{init.get('step')} -> {args.out}: {args.iterations} iterations of {args.games} games against {len(opponent_names)} opponents, starting after iteration {progress['iteration']}, {len(pool)} lost games in the pool", flush=True)
    opponents = load_opponents(opponent_names, device)
    worker_sets = [Workers(n) for n in [len(range(i, args.workers, args.worker_sets)) for i in range(args.worker_sets)]]

    iteration = progress["iteration"]
    try:
        for iteration in range(progress["iteration"] + 1, args.iterations + 1):
            started = time.perf_counter()
            games, n_replayed = schedule(args, iteration, pool, opponent_names)
            policy = Policy(model, device, sample=True, temperature=args.temperature, seed=args.seed + iteration)
            data, outcomes = play(worker_sets, games, policy, opponents)
            rollout_seconds = time.perf_counter() - started

            # Keep recent losses below the replay limit.
            replay_counts.update(games[:n_replayed])
            lost = sorted({game for game, outcome in zip(games, outcomes) if outcome["margin"] <= 0 and replay_counts[game] < args.replay_max})
            pool = ([game for game in pool if game not in lost] + lost)[-args.games:]

            count = data["rewards"].numel()
            advantage, returns = gae(data["rewards"].to(device), data["value"], args.gae_lambda)
            flat = {key: value.flatten(0, 1) for key, value in data.items() if key != "rewards"}
            flat["adv"], flat["ret"] = advantage.reshape(-1).double(), returns.reshape(-1)
            flat["adv"] = ((flat["adv"] - flat["adv"].mean()) / flat["adv"].std(unbiased=False).clamp(min=1e-6)).float()
            del data, advantage, returns
            update_started = time.perf_counter()
            measures, steps, skipped, stopped = update(model, opt, flat, args, progress)
            del flat

            margins = [outcome["margin"] for outcome in outcomes]
            row = {"iteration": iteration, "games": len(outcomes), "win_rate": sum(m > 0 for m in margins) / len(margins), "mean_margin": sum(margins) / len(margins),
                   "replayed_games": n_replayed, "replay_win_rate": sum(m > 0 for m in margins[:n_replayed]) / max(1, n_replayed), "fresh_win_rate": sum(m > 0 for m in margins[n_replayed:]) / max(1, len(margins) - n_replayed),
                   "seat_win_rate": {seat: sum(o["margin"] > 0 for o in outcomes if o["seat"] == seat) / max(1, sum(o["seat"] == seat for o in outcomes)) for seat in (0, 1)},
                   "pool": len(pool), "rows": count, "update_steps": steps, "skipped_steps": skipped, "total_updates": progress["total_updates"], "lr": opt.param_groups[0]["lr"], "early_stop": stopped,
                   "rollout_s": round(rollout_seconds), "update_s": round(time.perf_counter() - update_started), "iteration_s": round(time.perf_counter() - started),
                   "peak_gib": round(torch.cuda.max_memory_reserved(device) / 2 ** 30, 2), **measures}
            torch.cuda.reset_peak_memory_stats(device)
            append_jsonl(args.out / "metrics.jsonl", [row])
            print(json.dumps(row), flush=True)

            stopping = args.stop_file is not None and args.stop_file.exists()
            if iteration % args.ckpt_every == 0 or iteration == args.iterations or stopping:
                save_checkpoint({"model": model.state_dict(), "config": init["config"], "step": iteration, "iteration": iteration, "total_updates": progress["total_updates"], "optimizer": opt.state_dict()}, args.out, iteration, args.keep_checkpoints)
                temporary = pool_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"iteration": iteration, "pool": pool, "replays": [[game, count] for game, count in replay_counts.items()]}))
                temporary.replace(pool_path)
            if stopping:
                print(f"stop file {args.stop_file} found; saved after iteration {iteration}", flush=True)
                break
            torch.cuda.empty_cache()

        if iteration == args.iterations:
            target = args.out / "player.pt"
            temporary = target.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "config": init["config"], "step": iteration}, temporary)
            temporary.replace(target)
    finally:
        for workers in worker_sets:
            workers.close()


if __name__ == "__main__":
    main()
