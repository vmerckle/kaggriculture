# Run the pinned engine one decision hour at a time.
import json
from types import SimpleNamespace

from kaggle_environments.envs.kaggriculture import kaggriculture

with open(kaggriculture.json_path) as stream:
    CONFIG = {name: spec["default"] if isinstance(spec, dict) else spec for name, spec in json.load(stream)["configuration"].items()}
LAST_STEP = CONFIG["episodeSteps"] - 1


class Observation(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def copy(value):
    if isinstance(value, dict):
        return {k: copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [copy(v) for v in value]
    return value


def plain(value):
    # Convert observations to plain Python values.
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__dict__"):
        return {str(k): plain(v) for k, v in vars(value).items()}
    return value


class Game:
    # Observe, submit both seats' actions, then advance.

    def __init__(self, seed):
        self.seed, self.step = int(seed), 0
        self.env = SimpleNamespace(configuration=SimpleNamespace(**CONFIG), done=False, info={"seed": self.seed})
        self.shared = SimpleNamespace(step=0)  # Seat 0 owns shared state.
        self.state = [SimpleNamespace(observation=self.shared if i == 0 else SimpleNamespace(), action=None, status="ACTIVE", reward=None) for i in range(2)]
        kaggriculture.interpreter(self.state, self.env)

    @property
    def done(self):
        return self.step >= LAST_STEP

    def observe(self, seat):
        s = self.shared
        return Observation(player=seat, farms=copy(s.farms), private=copy(self.state[seat].observation.private), market=copy(s.market), town=copy(s.town), day=s.day, hour=s.hour, step=self.step, remainingOverageTime=60)

    def submit(self, seat, action):
        if not isinstance(action, dict):
            raise TypeError(f"seat {seat} returned {type(action).__name__}, not a dict")
        self.state[seat].action = json.loads(json.dumps(action))

    def advance(self):
        self.shared.step = self.step
        kaggriculture.interpreter(self.state, self.env)
        self.step += 1

    def money(self):
        return [float(farm["money"]) for farm in self.shared.farms]


def play(agents, seed):
    # Play two agents and return final money.
    game = Game(seed)
    while not game.done:
        for seat in range(2):
            game.submit(seat, agents[seat](game.observe(seat)))
        game.advance()
    return game.money()
