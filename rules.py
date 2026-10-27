# Action rules: A1 = recovery + melons; D1 = deliver; W1 = wheat trading.
import traceback
from copy import deepcopy
from actions import N_MARKET
from features import FIRST_YIELD, MARKET_I0, SHED_ACCESS, SHED_CAPACITY, engine

RULES       = {}
PRESETS     = {"A1": ("recovery", "melons"), "D1": ("deliver",), "W1": ("wheat",)}
SALE_ORDER  = ("FERTILIZER", "WHEAT", "EGG", "CARROT", "TOMATO", "MELON", "WOOL", "MILK", "STRAWBERRY")  # Recovery sale order.
SHED_COORDS = {x for x, _ in SHED_ACCESS}  # Shed access coordinates.


def parse(text):
    # Expand '+'-joined rules and presets; reject duplicates.
    rules = []
    for name in text.split("+"):
        for rule in PRESETS.get(name, (name,)):
            if rule not in RULES:
                raise ValueError(f"unknown rule {rule!r}; known: {', '.join(RULES)} and the presets {', '.join(PRESETS)}")
            if rule in rules:
                raise ValueError(f"rule {rule!r} repeated in {text!r}")
            rules.append(rule)
    return tuple(rules)


def rule(f):
    RULES[f.__name__] = f
    return f


def count(bag):
    return sum(int(v) for v in bag.values()) if isinstance(bag, dict) else 0


def unit_commands(action):
    # Return unit commands and a setter.
    commands = [action["farmer"], *action["hands"]]

    def set_command(i, value):
        if i == 0:
            action["farmer"] = value
        else:
            action["hands"][i - 1] = value
    return commands, set_command


@rule
def recovery(action, obs, state):
    farm = obs["farms"][int(obs["player"])]
    commands, set_command = unit_commands(action)
    positions = [farm["farmer"], *farm["hands"]]
    shed = dict(obs["private"]["shed"])
    bags = [dict(b) if isinstance(b, dict) else b for b in obs["private"]["inventories"]]
    prices = obs["market"]["prices"]
    n = min(len(commands), len(positions), len(bags))
    at_shed = [int(positions[i][0]) in SHED_COORDS and int(positions[i][1]) in SHED_COORDS for i in range(n)]

    if int(obs["step"]) >= 718:
        for i in range(n):
            if at_shed[i] and count(bags[i]) > 0:
                set_command(i, ["DROP"])

        sales, seen = [], set()
        for order in action["market"]:
            if len(order) >= 3 and order[0] == "SELL":
                sales.append([order[0], order[1], 10000])
                seen.add(order[1])
        action["market"] = sales + [["SELL", item, 10000] for item in SALE_ORDER if item not in seen]
        return

    last_hour = int(obs["hour"]) == 23
    used = set()

    def consume(bag, item):
        if int(bag.get(item, 0)) > 0:
            bag[item] = int(bag[item]) - 1
        elif int(shed.get(item, 0)) > 0:
            shed[item] = int(shed[item]) - 1

    for i in range(n):
        command, bag = commands[i], bags[i]
        if not command:
            continue
        op = command[0]
        x, y = int(positions[i][0]), int(positions[i][1])

        if op == "DROP" and at_shed[i]:
            room = max(0, SHED_CAPACITY - count(shed))
            if last_hour and count(bag) > room:
                set_command(i, ["PASS"])
            elif count(bag) > room:
                best, qty, value = "", 0, -1.0
                for item in sorted(bag):
                    q = min(room, int(bag[item]))
                    v = q * float(prices.get(item, 0.0))
                    if q > 0 and v > value:
                        value, best, qty = v, item, q
                if qty:
                    set_command(i, ["PLACE", best, qty])
                    shed[best] = int(shed.get(best, 0)) + qty
                    bag[best] = int(bag[best]) - qty
                else:
                    set_command(i, ["PASS"])
            else:
                for item in sorted(bag):
                    shed[item] = int(shed.get(item, 0)) + int(bag[item])
                bags[i] = bag = {}
            continue

        if not last_hour:
            continue
        # Predict end-of-hour cargo.
        if op == "PICKUP" and at_shed[i] and len(command) >= 2:
            item = command[1]
            qty = min(int(shed.get(item, 0)), int(command[2]) if len(command) >= 3 else 1)
            shed[item] = int(shed.get(item, 0)) - qty
            bag[item] = int(bag.get(item, 0)) + qty

        tile = farm["tiles"][y][x]
        if not isinstance(tile, dict):
            continue
        if op == "HARVEST" and (y, x, 0) not in used:
            used.add((y, x, 0))
            item = tile.get("crop", "")
            age = int(obs["day"]) - int(tile.get("planted_day", 0))
            if item and age < FIRST_YIELD[item]:
                continue
            if not item and isinstance(tile.get("animal"), str):
                item = engine.ANIMALS[tile["animal"]]["product"]
            if item:
                bag[item] = int(bag.get(item, 0)) + int(tile.get("yield_units", 0))
        if op == "COLLECT_FERTILIZER" and tile.get("fertilizer_available", False) and (y, x, 1) not in used:
            used.add((y, x, 1))
            bag["FERTILIZER"] = int(bag.get("FERTILIZER", 0)) + 1
        if op == "FEED" and not tile.get("fed_today", True) and (y, x, 2) not in used:
            used.add((y, x, 2))
            consume(bag, "WHEAT")
        if op == "FERTILIZE":
            consume(bag, "FERTILIZER")

    if not last_hour:
        return
    # Sell end-of-day overflow; reserve 30 wheat.
    remaining = dict(shed)
    for order in action["market"]:
        if len(order) < 3:
            continue
        op, item = order[0], order[1]
        if op == "SELL":
            remaining[item] = max(0, int(remaining.get(item, 0)) - int(order[2]))
        if op in ("BUY_PRODUCT", "BUY_ANIMAL"):
            remaining[item] = int(remaining.get(item, 0)) + int(order[2])

    excess = count(remaining) - SHED_CAPACITY + sum(count(b) for b in bags)
    if excess <= 0:
        return

    for item in SALE_ORDER:
        qty = min(excess, max(0, int(remaining.get(item, 0)) - (30 if item == "WHEAT" else 0)))
        if not qty:
            continue
        for order in action["market"]:
            if len(order) >= 3 and order[0] == "SELL" and order[1] == item:
                order[2] = int(order[2]) + qty
                break
        else:
            if len(action["market"]) >= N_MARKET:
                continue
            action["market"].append(["SELL", item, qty])
        excess -= qty
        if excess <= 0:
            break


def melon_supply(obs):
    # Estimate melon supply, including stored and growing melons.
    private = obs["private"]
    supply = float(int(obs["market"]["inventory"].get("MELON", 0)) - MARKET_I0 - 10) + int(private["shed"].get("MELON", 0))
    supply += sum(int(bag.get("MELON", 0)) for bag in private["inventories"] if isinstance(bag, dict))
    for farm in obs["farms"]:
        for row in farm["tiles"]:
            for tile in row:
                if isinstance(tile, dict) and tile.get("crop", "") == "MELON":
                    supply += max(6, int(tile.get("yield_units", 0)))
    return max(0.0, supply)


def displaced_value(prices, crop):
    # Estimate revenue from the displaced crop.
    if crop == "WHEAT":
        return 2.0 * (4 * int(prices.get("WHEAT", 0)) - 10)
    if crop == "CARROT":
        return 2.5 * (3 * int(prices.get("CARROT", 0)) - 20)
    if crop == "TOMATO":
        return 4.0 * int(prices.get("TOMATO", 0)) - 50
    return 1e9


def melon_revenue(supply):
    # Revenue from six melons sold after supply others.
    return sum(max(1.0, 250 - .01 * (supply + j) ** 2) for j in range(6))


@rule
def melons(action, obs, state):
    if int(obs["step"]) // 24 != 18:
        return

    prices = obs["market"]["prices"]
    if "melon_target" not in state:
        supply, displaced, target = melon_supply(obs), min(displaced_value(prices, crop) for crop in ("WHEAT", "CARROT", "TOMATO")), 0
        while target < 8 and melon_revenue(supply + 6 * target) - 80 - displaced >= 100:
            target += 1
        state.update(melon_target=target, melon_planted=0, melon_bought=False)
    target = state["melon_target"]
    if not target:
        return

    farm = obs["farms"][int(obs["player"])]
    if not state["melon_bought"] and len(action["market"]) < N_MARKET and farm.get("money", 0) > 1000 + 80 * target:
        action["market"].append(["BUY_SEED", "MELON", target])
        state["melon_bought"] = True

    available = int(obs["private"]["seeds"].get("MELON", 0))
    supply = melon_supply(obs)
    commands, set_command = unit_commands(action)
    used = set()
    for i in range(1 + min(len(farm["hands"]), len(action["hands"]))):
        if not (available > 0 and state["melon_planted"] < target):
            break
        command = commands[i]
        if len(command) < 2 or command[0] != "PLANT" or melon_revenue(supply) - 80 - displaced_value(prices, command[1]) < 100:
            continue
        position = farm["farmer"] if i == 0 else farm["hands"][i - 1]
        x, y = int(position[0]), int(position[1])
        if farm["tiles"][y][x] is not None or (y, x) in used:
            continue
        used.add((y, x))
        set_command(i, ["PLANT", "MELON"])
        supply += 6
        available -= 1
        state["melon_planted"] += 1


@rule
def deliver(action, obs, state):
    step = int(obs["step"])
    if step < 690 or step >= 718:
        return

    farm = obs["farms"][int(obs["player"])]
    bags = obs["private"]["inventories"]
    positions = [farm["farmer"], *farm["hands"]]
    commands, set_command = unit_commands(action)

    for i in range(min(len(commands), len(positions), len(bags))):
        if count(bags[i]) <= 0:
            continue
        x, y = int(positions[i][0]), int(positions[i][1])
        tx, ty = min(max(x, 4), 5), min(max(y, 4), 5)
        distance = abs(x - tx) + abs(y - ty)
        if distance == 0:
            if step >= 717:
                set_command(i, ["DROP"])
        elif distance >= 718 - step:
            set_command(i, ["EAST" if x < tx else "WEST" if x > tx else "SOUTH" if y < ty else "NORTH"])


def wheat_view(obs, state):
    # Hide W1's investment from the network.
    floating = int(state.get("wheat_float", 0))
    if floating <= 0:
        return obs
    view = deepcopy(obs)
    seat = int(obs["player"])
    held = min(floating, int(obs["private"]["shed"].get("WHEAT", 0)))
    view["private"]["shed"]["WHEAT"] = int(obs["private"]["shed"].get("WHEAT", 0)) - held
    view["farms"][seat]["money"] = float(obs["farms"][seat].get("money", 0.0)) + int(state.get("wheat_principal", 0))
    inventory = int(obs["market"]["inventory"].get("WHEAT", 0)) + held
    view["market"]["inventory"]["WHEAT"] = inventory
    view["market"]["prices"]["WHEAT"] = engine.market_price("WHEAT", inventory)
    return view


@rule
def wheat(action, obs, state):
    floating = int(state.get("wheat_float", 0))
    if floating > 0:
        if len(action["market"]) >= N_MARKET:
            del action["market"][N_MARKET - 1]
        action["market"].append(["SELL", "WHEAT", floating])
        state.update(wheat_float=0, wheat_principal=0)
        return

    step = int(obs["step"])
    if step < 288 or step >= 716 or step % 4 or step % 24 == 0 or len(action["market"]) >= N_MARKET - 1:
        return
    farm = obs["farms"][int(obs["player"])]
    if float(farm.get("money", 0.0)) <= 6000:
        return
    if any(order and order[0] in ("BUY_PRODUCT", "BUY_ANIMAL") for order in action["market"]):
        return

    positions = [farm["farmer"], *farm["hands"]]
    commands, _ = unit_commands(action)
    bags = obs["private"]["inventories"]
    shed = obs["private"]["shed"]
    total, reserve, available = count(shed), 0, int(shed.get("WHEAT", 0))
    for command, position, bag in zip(commands, positions, bags):
        if not command:
            continue
        x, y = int(position[0]), int(position[1])
        op = command[0]
        if x in SHED_COORDS and y in SHED_COORDS:
            if op == "DROP":
                total += count(bag)
                available += int(bag.get("WHEAT", 0))
            if op == "PLACE" and len(command) >= 2:
                item = command[1]
                qty = min(int(bag.get(item, 0)), int(command[2]) if len(command) >= 3 else 1)
                total += qty
                if item == "WHEAT":
                    available += qty
            if op == "PICKUP" and len(command) >= 2:
                item = command[1]
                qty = min(int(shed.get(item, 0)), int(command[2]) if len(command) >= 3 else 1)
                total -= qty
                if item == "WHEAT":
                    available -= qty

        if op == "EAST":
            x += 1
        if op == "WEST":
            x -= 1
        if op == "NORTH":
            y -= 1
        if op == "SOUTH":
            y += 1
        if x in SHED_COORDS and y in SHED_COORDS:
            reserve += count(bag) + 8

    qty = min(100, max(0, SHED_CAPACITY - total - reserve))
    if not qty:
        return
    inventory = int(obs["market"]["inventory"].get("WHEAT", 0))
    for order in action["market"]:
        if len(order) >= 3 and order[0] == "SELL" and order[1] == "WHEAT":
            sold = min(available, int(order[2]))
            inventory += sold
            available -= sold

    cost = sum(engine.market_price("WHEAT", inventory - n) for n in range(1, qty + 1))
    if float(farm.get("money", 0.0)) < cost + 3000:
        return
    action["market"].append(["BUY_PRODUCT", "WHEAT", qty])
    state.update(wheat_float=qty, wheat_principal=cost)


class Rules:
    # Apply stateful rules; reset on new games and keep the action if a rule fails, reporting the first failure per process.
    reported = False

    def __init__(self, rules):
        self.rules, self.state, self.last_step = rules, {}, -1

    def observe(self, obs):
        if "wheat" not in self.rules:
            return obs
        if int(obs["step"]) <= self.last_step:
            self.state = {}
        return wheat_view(obs, self.state)

    def __call__(self, obs, action):
        step = int(obs["step"])
        if step <= self.last_step:
            self.state = {}
        self.last_step = step
        edited = {"farmer": list(action["farmer"]), "hands": [list(h) for h in action["hands"]], "market": [list(o) for o in action["market"]]}
        try:
            for name in self.rules:
                if name != "wheat":
                    RULES[name](edited, obs, self.state)
            if "wheat" in self.rules:
                wheat(edited, obs, self.state)
        except Exception:
            if not Rules.reported:
                Rules.reported = True
                traceback.print_exc()
            return action
        return edited
