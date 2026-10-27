#!/usr/bin/env python3
# Replay, verify, and pack training data; see DATA.md for inputs.
import argparse
import gzip
import json
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import actions
import features
import game

VAL_SEATS, VAL_FRACTION, RECENT_DAYS, RECENT_WEIGHT = 30, 0.05, 1.0, 4
SHAPES = {"tiles": ((2, features.N_TILES, features.TILE_D), np.uint8), "units": ((features.N_UNITS, features.UNIT_D), np.uint8), "globals": ((features.GLOBAL_D,), np.float16), "tokens": ((actions.N_ROWS, 3), np.int16),
          "n_units": ((), np.uint8), "margin": ((), np.float32), "team": ((), np.int16)}


def eligible(episode):
    agents = episode.get("agents", [])
    return episode.get("state") == "COMPLETED" and episode.get("on_pin") and len(agents) == 2 and all(a.get("reward") is not None for a in agents)


def select(episodes, labels_by_id, replays):
    # Select team seats from completed, pinned-engine games.
    jobs = []
    for episode in filter(eligible, episodes):
        for agent in episode["agents"]:
            if int(agent["team_id"]) in labels_by_id:
                jobs.append({"episode": int(episode["episode_id"]), "path": str(replays / episode["path"]), "seat": int(agent["index"]), "team_id": int(agent["team_id"]), "team": labels_by_id[int(agent["team_id"])],
                             "submission_id": int(agent["submission_id"]), "rewards": [float(a["reward"]) for a in sorted(episode["agents"], key=lambda a: a["index"])]})
    return jobs


def held_out(jobs):
    # Hold out games using seeded per-team shuffles.
    by_team = {}
    for job in jobs:
        by_team.setdefault(job["team"], []).append(job["episode"])
    rng, val_episodes = random.Random(0), set()
    for episodes in by_team.values():
        episodes = sorted(set(episodes))
        rng.shuffle(episodes)
        val_episodes.update(episodes[:max(1, round(VAL_FRACTION * len(episodes)))])
    return val_episodes


class Recorder:
    # Record features and round-trip actions through the codec.

    def __init__(self, recorded, seat):
        self.recorded, self.seat, self.hours = recorded, seat, []

    def __call__(self, obs):
        feats = features.encode(obs, self.seat)
        tokens = actions.encode(self.recorded[obs.step][self.seat], feats["n_units"])
        self.hours.append((feats["tiles"].astype(np.uint8), feats["units"].astype(np.uint8), feats["globals"].astype(np.float16), tokens, feats["n_units"]))
        return actions.decode(tokens, feats["n_units"])

    def arrays(self):
        return {key: np.stack([hour[i] for hour in self.hours]) for i, key in enumerate(("tiles", "units", "globals", "tokens"))} | {"n_units": np.array([hour[4] for hour in self.hours], dtype=np.uint8)}


def replay(job):
    # Replay a seat and return its manifest entry or error.
    row = {key: job[key] for key in ("episode", "seat", "team_id", "team", "submission_id", "rewards", "split")}
    try:
        doc = json.load(gzip.open(job["path"]))
        recorded, row["seed"] = doc["actions"][1:], int(doc["seed"])
        if game.play([lambda obs: recorded[obs.step][0], lambda obs: recorded[obs.step][1]], row["seed"]) != job["rewards"]:
            return {**row, "error": "the engine does not reproduce the recorded rewards"}
        recorder = Recorder(recorded, job["seat"])
        if game.play([recorder if seat == job["seat"] else (lambda obs, seat=seat: recorded[obs.step][seat]) for seat in range(2)], row["seed"]) != job["rewards"]:
            return {**row, "error": "the encoded actions do not reproduce the recorded rewards"}
        path = Path(job["out_dir"]) / f"ep{job['episode']}-seat{job['seat']}.npz"
        np.savez_compressed(path, seat=np.int16(job["seat"]), money=np.asarray(job["rewards"], dtype=np.float64), **recorder.arrays())
        return {**row, "file": path.name}
    except Exception as exc:  # Skip broken replays.
        return {**row, "error": f"{type(exc).__name__}: {exc}"}


def pack(files, team_indices, labels, out, row_lists, meta):
    # Pack seats into arrays; row_lists supplies per-seat repetition counts.
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as pool:
        lengths = list(pool.map(lambda f: len(np.load(f)["n_units"]), files))
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    arrays = {key: np.lib.format.open_memmap(out / f"{key}.npy", mode="w+", dtype=dtype, shape=(int(offsets[-1]), *shape)) for key, (shape, dtype) in SHAPES.items()}

    def fill(i):
        data = np.load(files[i])
        span = slice(int(offsets[i]), int(offsets[i + 1]))
        for key in ("tiles", "units", "globals", "tokens", "n_units"):
            arrays[key][span] = data[key]
        seat = int(data["seat"])
        arrays["margin"][span] = (data["money"][seat] - data["money"][1 - seat]) / 1e5
        arrays["team"][span] = team_indices[i]

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(fill, range(len(files))))
    for array in arrays.values():
        array.flush()
    counts = {}
    for name, repeats in row_lists.items():
        np.save(out / name, np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in range(len(files)) for _ in range(repeats(i))] or [np.zeros(0, np.int64)]).astype(np.int64))
        counts[name] = sum(repeats(i) for i in range(len(files)))
    (out / "meta.json").write_text(json.dumps({"teams": labels, **meta, "files": [os.path.basename(f) for f in files], "hours": int(offsets[-1]), **counts}))
    print(f"{out}: {len(files)} seats, {int(offsets[-1])} hours" + "".join(f", {name} lists {n} seats" for name, n in counts.items()), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Replay, verify, and pack training data; see DATA.md for inputs.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replays", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    replays, out, npz = args.replays.resolve(), args.out.resolve(), args.out.resolve() / "npz"
    index = json.loads((replays / "index.json").read_text())
    episodes = index["episodes"]
    # Clone the index's optional teams list, otherwise every team in its eligible games.
    team_ids = [int(t) for t in index["teams"]] if index.get("teams") else sorted({int(a["team_id"]) for e in filter(eligible, episodes) for a in e["agents"]})
    labels = [f"t{t}" for t in team_ids]

    jobs = select(episodes, dict(zip(team_ids, labels)), replays)
    val_episodes = held_out(jobs)
    for job in jobs:
        job["split"] = "val" if job["episode"] in val_episodes else "train"
        job["out_dir"] = str(npz / job["split"])
        Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)

    manifest = npz / "manifest.jsonl"
    done = {(row["episode"], row["seat"]) for row in map(json.loads, manifest.read_text().splitlines()) if "file" in row} if manifest.exists() else set()
    todo = [job for job in jobs if (job["episode"], job["seat"]) not in done]
    print(f"{len(jobs)} seats of {len(labels)} teams in {len(episodes)} games, {len(val_episodes)} games held out; {len(todo)} seats to replay with {args.workers} workers", flush=True)
    errors, started = 0, time.perf_counter()
    with manifest.open("a") as stream, mp.Pool(args.workers, maxtasksperchild=16) as pool:
        for i, row in enumerate(pool.imap_unordered(replay, todo), 1):
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            errors += "error" in row
            if i % 500 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} seats, {errors} errors, {time.perf_counter() - started:.0f}s", flush=True)

    rows = [row for row in map(json.loads, manifest.read_text().splitlines()) if "file" in row]
    created = {int(e["episode_id"]): e["create_time"] for e in episodes}
    first_game = {}
    for episode in episodes:
        for agent in episode.get("agents", []):
            first_game[int(agent["submission_id"])] = min(first_game.get(int(agent["submission_id"]), "9"), episode["create_time"])
    since = (datetime.fromisoformat(max(created.values())[:19]) - timedelta(days=RECENT_DAYS)).isoformat()
    recent_wins = {row["file"] for row in rows if row["split"] == "train" and row["rewards"][row["seat"]] > row["rewards"][1 - row["seat"]] and created[row["episode"]][:19] >= since}

    # Fine-tune on the index's optional finetune_team, otherwise the team with the most won training seats.
    wins = {label: sum(row["team"] == label and row["split"] == "train" and row["rewards"][row["seat"]] > row["rewards"][1 - row["seat"]] for row in rows) for label in labels}
    finetune_team = f"t{int(index['finetune_team'])}" if index.get("finetune_team") else max(labels, key=lambda label: wins[label])
    newest = sorted({row["submission_id"] for row in rows if row["team"] == finetune_team and row["split"] == "train"}, key=lambda s: first_game[s])[-2:]
    finetune_files = {row["file"] for row in rows if row["team"] == finetune_team and row["split"] == "train" and row["submission_id"] in newest}
    if not finetune_files:
        sys.exit(f"fine-tune team {finetune_team} has no training seats; set finetune_team in index.json to a cloned team with replayable games")
    print(f"{len(rows)} seats replayed; {len(recent_wins)} won training seats since {since} listed {RECENT_WEIGHT} times; fine-tune on {finetune_team} submissions {newest}, {len(finetune_files)} seats", flush=True)

    train_seats = [(str(npz / "train" / row["file"]), labels.index(row["team"])) for row in sorted(rows, key=lambda row: (labels.index(row["team"]), row["file"])) if row["split"] == "train"]
    val_seats = [(str(npz / "val" / row["file"]), labels.index(row["team"])) for label in labels for row in sorted((row for row in rows if row["team"] == label and row["split"] == "val"), key=lambda row: row["file"])[:VAL_SEATS]]
    names = [os.path.basename(f) for f, _ in train_seats]
    pack([f for f, _ in train_seats], [t for _, t in train_seats], labels, out / "train", {"rows.npy": lambda i: RECENT_WEIGHT if names[i] in recent_wins else 1, "finetune.npy": lambda i: int(names[i] in finetune_files)}, {"finetune_team": finetune_team})
    pack([f for f, _ in val_seats], [t for _, t in val_seats], labels, out / "val", {}, {"finetune_team": finetune_team})


if __name__ == "__main__":
    main()
