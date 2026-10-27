# Workers run games; the parent runs policies, with first-seat rows before opponents.
import multiprocessing as mp
import re
import traceback

import numpy as np
import torch

import actions
import features
import game
import rules
from model import HourFormer

OPPONENT_PATTERN = re.compile(r"frozen((?:\+\w+)*):(.+)")
ROLLOUT_KEYS     = ("tiles", "units", "globals", "exists", "tokens", "log_prob", "value")
FEATURE_DTYPES   = {"tiles": np.uint8, "units": np.uint8, "globals": np.float32, "n_units": np.int64}


# Parameter-name prefixes that the Kaggle Models weights use for four layers, and the names model.py uses.
OLD_PREFIXES = {"tile_embed.table.": "tile_embed.categorical.", "unit_embed.table.": "unit_embed.categorical.", "op_unit.": "unit_op_head.", "op_market.": "market_op_head."}


def current_names(weights):
    # Map those prefixes to model.py's names; other names and all tensors are unchanged.
    def rename(key):
        for old, new in OLD_PREFIXES.items():
            if key.startswith(old):
                return new + key[len(old):]
        return key
    return {rename(key): value for key, value in weights.items()}


def load_model(path, device):
    # Load an eval-mode model and its checkpoint.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = HourFormer(**checkpoint["config"]).to(device).eval()
    model.load_state_dict(current_names(checkpoint["model"]), strict=True)
    return model, checkpoint


class Policy:
    # Batched policy with its own sampling generator.

    def __init__(self, model, device, sample=False, temperature=1.0, seed=0):
        self.model, self.device, self.sample, self.temperature = model, torch.device(device), sample, temperature
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    @torch.no_grad()
    def infer(self, feats):
        # Decode features and return device tensors.
        tensors = {key: torch.from_numpy(np.ascontiguousarray(feats[key], dtype=dtype)).to(self.device) for key, dtype in FEATURE_DTYPES.items()}
        exists = torch.arange(features.N_UNITS, device=self.device)[None] < tensors["n_units"][:, None]
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            tokens, log_prob, value = self.model.act(tensors["tiles"], tensors["units"], tensors["globals"], exists, self.sample, self.temperature, self.generator)
        return {"tiles": tensors["tiles"], "units": tensors["units"], "globals": tensors["globals"], "exists": exists, "tokens": tokens, "log_prob": log_prob.float(), "value": value.float()}

    def act(self, observations, seats):
        # Return actions for the observed seats.
        feats = [features.encode(obs, seat) for obs, seat in zip(observations, seats)]
        tokens = self.infer({key: np.stack([f[key] for f in feats]) for key in FEATURE_DTYPES})["tokens"].cpu().numpy()
        return [actions.decode(row, f["n_units"]) for row, f in zip(tokens, feats)]


def parse_opponent(name):
    # Parse frozen[+RULES]:PATH into checkpoint path and rules.
    match = OPPONENT_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"{name!r} is not an opponent (`frozen:PATH` or `frozen+RULES:PATH`)")
    return match.group(2), rules.parse(match.group(1)[1:]) if match.group(1) else ()


def load_opponents(names, device):
    # Load greedy opponents, sharing policies for identical checkpoint paths.
    policies, opponents = {}, {}
    for name in names:
        path, selected_rules = parse_opponent(name)
        if path not in policies:
            policies[path] = Policy(load_model(path, device)[0], device)
        opponents[name] = policies[path], selected_rules
    return opponents


def worker(conn):
    # Run games until the parent closes the connection.
    games, rows, observations, n_units = [], [], [], None
    try:
        while True:
            command, payload = conn.recv()
            if command == "start":  # Payload: (seed, seat, own_rules, opponent_rules).
                games = [game.Game(seed) for seed, _, _, _ in payload]
                rows = [(g, seat, rules.Rules(own_rules)) for g, (_, seat, own_rules, _) in zip(games, payload)] + [(g, 1 - seat, rules.Rules(opponent_rules)) for g, (_, seat, _, opponent_rules) in zip(games, payload)]
            elif command == "step":  # Payload: tokens (rows, 26, 3).
                for (g, seat, rule_set), tokens, n, obs in zip(rows, payload, n_units, observations):
                    action = actions.decode(tokens, n)
                    g.submit(seat, rule_set(game.plain(obs), action) if rule_set.rules else action)
                for g in games:
                    g.advance()
            elif command == "finish":
                conn.send(("outcomes", [{"seat": seat, "margin": g.money()[seat] - g.money()[1 - seat]} for g, seat, _ in rows[:len(games)]]))
                continue
            else:
                return

            margins = np.array([g.money()[seat] - g.money()[1 - seat] for g, seat, _ in rows], dtype=np.float64)
            if not games or games[0].done:
                conn.send(("features", {"margins": margins, "done": True}))
                continue
            observations = [g.observe(seat) for g, seat, _ in rows]
            feats = [features.encode(rule_set.observe(obs), seat) for obs, (_, seat, rule_set) in zip(observations, rows)]
            n_units = np.array([f["n_units"] for f in feats], dtype=np.int64)
            conn.send(("features", {"tiles": np.stack([f["tiles"] for f in feats]).astype(np.uint8), "units": np.stack([f["units"] for f in feats]).astype(np.uint8),
                                    "globals": np.stack([f["globals"] for f in feats]), "n_units": n_units, "margins": margins, "done": False}))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
        raise


class Workers:
    # Game processes controlled through pipes.

    def __init__(self, n):
        ctx = mp.get_context("spawn")
        self.conns, self.procs, self.counts = [], [], []
        for _ in range(n):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=worker, args=(child,), daemon=True)
            proc.start()
            self.conns.append(parent)
            self.procs.append(proc)

    def start(self, games):
        # Distribute games and return initial features.
        n, w = len(games), len(self.conns)
        bounds = [round(i * n / w) for i in range(w + 1)]
        self.counts = [b - a for a, b in zip(bounds, bounds[1:])]
        for conn, a, b in zip(self.conns, bounds, bounds[1:]):
            conn.send(("start", games[a:b]))
        return self.observe()

    def receive(self):
        replies = []
        for conn in self.conns:
            kind, value = conn.recv()
            if kind == "error":
                raise RuntimeError(value)
            replies.append(value)
        return replies

    def observe(self):
        # Collect features, money margins, and completion status.
        parts = self.receive()
        done = parts[0]["done"]
        keys = ("margins",) if done else ("tiles", "units", "globals", "n_units", "margins")
        feats = {key: np.concatenate([p[key][:c] for p, c in zip(parts, self.counts)] + [p[key][c:] for p, c in zip(parts, self.counts)]) for key in keys}
        feats["done"] = done
        return feats

    def step(self, rows):
        # Send action tokens and advance games.
        n, a = sum(self.counts), 0
        for conn, c in zip(self.conns, self.counts):
            conn.send(("step", np.concatenate([rows[a:a + c], rows[n + a:n + a + c]])))
            a += c

    def finish(self):
        # Return outcomes in game order from the first seat's perspective.
        for conn in self.conns:
            conn.send(("finish", None))
        return [row for part in self.receive() for row in part]

    def close(self):
        for conn, proc in zip(self.conns, self.procs):
            if proc.is_alive():
                conn.send(("close", None))
            proc.join(10)
            if proc.is_alive():
                proc.kill()
            conn.close()


def play(worker_sets, games, policy, opponents, reward_scale=1e5):
    # Collect policy rollouts (hours, games, ...) and outcomes, overlapping worker sets.
    bounds = [round(i * len(games) / len(worker_sets)) for i in range(len(worker_sets) + 1)]

    def collector(workers, share):
        n = len(share)
        feats = workers.start([(seed, seat, (), opponents[name][1]) for seed, seat, name in share])
        groups = {}
        for j, (_, _, name) in enumerate(share):
            groups.setdefault(id(opponents[name][0]), (opponents[name][0], []))[1].append(n + j)
        rollout = {key: [] for key in ROLLOUT_KEYS + ("rewards",)}
        previous = torch.from_numpy(feats["margins"][:n])

        while not feats["done"]:
            output = policy.infer({key: feats[key][:n] for key in FEATURE_DTYPES})
            for key in ROLLOUT_KEYS:
                rollout[key].append(output[key])
            pending = [(output["tokens"], slice(0, n))] + [(frozen.infer({key: feats[key][rows] for key in FEATURE_DTYPES})["tokens"], rows) for frozen, rows in groups.values()]
            yield
            tokens = np.empty((2 * n, actions.N_ROWS, 3), dtype=np.int64)
            for chosen, rows in pending:
                tokens[rows] = chosen.cpu().numpy()
            workers.step(tokens)
            yield
            feats = workers.observe()
            current = torch.from_numpy(feats["margins"][:n])
            rollout["rewards"].append(((current - previous) / reward_scale).float())
            previous = current

        rollout["outcomes"] = workers.finish()
        return rollout

    collectors = [collector(workers, games[a:b]) for workers, a, b in zip(worker_sets, bounds, bounds[1:])]
    results, active = [None] * len(worker_sets), list(range(len(worker_sets)))
    while active:
        for i in list(active):
            try:
                next(collectors[i])
            except StopIteration as stop:
                results[i], active = stop.value, [j for j in active if j != i]

    data = {key: torch.cat([torch.stack(r[key]) for r in results], 1) for key in ROLLOUT_KEYS + ("rewards",)}
    return data, [row for r in results for row in r["outcomes"]]
