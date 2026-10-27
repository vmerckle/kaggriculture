# Encode observations as tiles (2, 100, 24), units (16, 31), and globals (157).
import numpy as np

from game import CONFIG, kaggriculture as engine

CROPS, ANIMALS, PRODUCTS = list(engine.CROPS), list(engine.ANIMALS), list(engine.PRODUCTS)
ITEMS = PRODUCTS + ANIMALS
SHOP_PRODUCTS = engine.SHOPS
SHOPS = list(SHOP_PRODUCTS)
FIRST_YIELD = {name: spec["first_yield_day"] for name, spec in (engine.CROPS | engine.ANIMALS).items()}  # Days to first yield.
ONGOING = {name for name, spec in engine.CROPS.items() if spec["ongoing"]}
BASE_PRICE = {name: spec["base"] for name, spec in engine.MARKET_PARAMS.items()}
LAND_PRICES, MARKET_I0 = engine.LAND_PRICES, engine.MARKET_I0
HIRE_COSTS = [engine._hire_cost(n) for n in range(20)]
BOARD_SIZE, N_STEPS, HOURS_PER_DAY, SHED_CAPACITY = CONFIG["boardSize"], CONFIG["episodeSteps"], CONFIG["turnsPerDay"], CONFIG["shedCapacity"]
SHED_ACCESS = set(engine._shed_access_tiles(BOARD_SIZE))

N_TILES, N_UNITS = BOARD_SIZE * BOARD_SIZE, 16
EMPTY, LOCKED, WEED, PLANT, COOP, PASTURE = range(6)
ITEM_INDEX = {name: i for i, name in enumerate(ITEMS)}

TILE_COLUMNS = [("kind", 6), ("crop", 6), ("animal", 4), ("age", 32), ("watered", 2), ("unwatered", 3), ("yield", 8), ("lifespan", 0), ("fert_left", 4), ("cared", 2), ("fert_avail", 2), ("care_bonus", 8),
                ("units_here", 0), ("farmer_here", 2), ("shed_access", 2), ("quadrant", 4), ("x", 10), ("y", 10), ("harvestable", 2), ("waterable", 2), ("feedable", 2), ("careable", 2), ("first_yield_in", 16), ("is_ongoing", 2)]
UNIT_COLUMNS = [("exists", 2), ("is_farmer", 2), ("x", 10), ("y", 10), ("shed_access", 2)] + [(f"cargo_{name}", 0) for name in ITEMS] + [("cargo_total", 0), ("slot", 16), ("under_kind", 6), ("under_crop", 6), ("under_animal", 4),
                ("under_yield", 8), ("under_watered", 2), ("under_harvestable", 2), ("under_cared", 2), ("under_fert_avail", 2), ("under_fert_left", 4), ("carry_wheat", 2), ("carry_fert", 2), ("carry_animal", 2)]
TILE_SCALE = {"lifespan": 200.0, "units_here": 4.0}
UNIT_SCALE = {name: 100.0 for name, card in UNIT_COLUMNS if card == 0}
GLOBALS = (["day", "hour", "step", "remaining", "hour_mod4", "last_hour", "first_hour"] + [f"hour_oh_{i}" for i in range(24)] + [f"day_oh_{i}" for i in range(30)]
           + ["money", "money_log", "opp_money", "opp_money_log", "money_diff"] + [f"shed_{n}" for n in ITEMS] + ["shed_total", "shed_room"] + [f"seed_{n}" for n in CROPS]
           + ["hands", "opp_hands", "hires_today", "opp_hires_today", "next_hire_cost", "quadrants", "opp_quadrants", "next_land_price", "land_affordable", "opp_land_affordable"]
           + [f"price_{n}" for n in PRODUCTS] + [f"inv_{n}" for n in PRODUCTS] + [f"shop_{n}" for n in SHOPS] + ["n_shops"] + [f"demand_{n}" for n in PRODUCTS]
           + [f"plants_{n}" for n in CROPS] + [f"opp_plants_{n}" for n in CROPS] + [f"animals_{n}" for n in ANIMALS] + [f"opp_animals_{n}" for n in ANIMALS]
           + ["weeds", "opp_weeds", "empties", "opp_empties", "structures_free", "opp_structures_free", "harvestable_units", "opp_harvestable_units", "cargo_total", "sellable_value"])
TILE_D, UNIT_D, GLOBAL_D = len(TILE_COLUMNS), len(UNIT_COLUMNS), len(GLOBALS)
G = {name: i for i, name in enumerate(GLOBALS)}


def clip(v, lo, hi):
    return min(max(int(v), lo), hi)


def encode_farm(farm, day, step, tiles):
    # Fill tile features in reading order and return farm counts.
    counts = {"empties": 0, "weeds": 0, "structures_free": 0, "harvestable": 0, "plants": [0] * len(CROPS), "animals": [0] * len(ANIMALS)}
    fx, fy = (int(v) for v in farm["farmer"])
    units_here = {}
    for x, y in [farm["farmer"], *farm["hands"]]:
        units_here[int(x), int(y)] = units_here.get((int(x), int(y)), 0) + 1

    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            location = (units_here.get((x, y), 0), int(x == fx and y == fy), int((x, y) in SHED_ACCESS), 2 * (y >= 5) + (x >= 5), x, y)  # Tile columns 12:18.
            if tile is None:
                counts["empties"] += 1
                tiles[y * BOARD_SIZE + x, 12:18] = location
            elif not isinstance(tile, dict):
                tiles[y * BOARD_SIZE + x, :18] = (LOCKED, *[0] * 11, *location)
            elif tile["kind"] == "WEED":
                counts["weeds"] += 1
                tiles[y * BOARD_SIZE + x, :18] = (WEED, *[0] * 11, *location)
            elif tile["kind"] == "PLANT":
                crop, age, yield_units = tile["crop"], day - int(tile["planted_day"]), int(tile["yield_units"])
                harvestable = int(yield_units > 0 and age >= FIRST_YIELD[crop])
                lifespan = 200 if tile["max_lifespan_step"] < 0 else clip(tile["max_lifespan_step"] - step, 0, 200)
                counts["plants"][CROPS.index(crop)] += 1
                counts["harvestable"] += yield_units * harvestable
                tiles[y * BOARD_SIZE + x] = (PLANT, CROPS.index(crop) + 1, 0, clip(age, 0, 31), int(bool(tile["watered_today"])), clip(tile["consecutive_unwatered"], 0, 2), min(yield_units, 7), lifespan,
                                      clip(int(tile["fertilized_until_day"]) - day + 1, 0, 3), 0, 0, 0, *location, harvestable, int(not tile["watered_today"]), 0, 0, clip(FIRST_YIELD[crop] - age, 0, 15), int(crop in ONGOING))
            elif tile.get("animal") is None:
                counts["structures_free"] += 1
                tiles[y * BOARD_SIZE + x, :18] = (COOP if tile["kind"] == "COOP" else PASTURE, *[0] * 11, *location)
            else:
                animal, age, yield_units = tile["animal"], day - int(tile["placed_day"]), int(tile["yield_units"])
                counts["animals"][ANIMALS.index(animal)] += 1
                counts["harvestable"] += yield_units
                tiles[y * BOARD_SIZE + x] = (COOP if tile["kind"] == "COOP" else PASTURE, 0, ANIMALS.index(animal) + 1, clip(age, 0, 31), int(bool(tile["fed_today"])), clip(tile["consecutive_unfed"], 0, 2), min(yield_units, 7), 0, 0,
                                      int(bool(tile["cared_today"])), int(bool(tile["fertilizer_available"])), clip(tile.get("pending_care_bonus", 0), 0, 7), *location,
                                      int(yield_units > 0), 0, int(not tile["fed_today"]), int(not tile["cared_today"]), clip(FIRST_YIELD[animal] - age, 0, 15), 0)
    return counts


def encode(obs, seat):
    # Encode a seat's observation.
    own, opp = obs["farms"][seat], obs["farms"][1 - seat]
    private, market, town = obs["private"], obs["market"], obs["town"]
    day, hour, step = int(obs["day"]), int(obs["hour"]), int(obs["step"])
    tiles = np.zeros((2, N_TILES, TILE_D), dtype=np.int16)
    own_counts, opp_counts = encode_farm(own, day, step, tiles[0]), encode_farm(opp, day, step, tiles[1])

    positions = [own["farmer"], *own["hands"]][:N_UNITS]
    inventories = private.get("inventories") or []
    units = np.zeros((N_UNITS, UNIT_D), dtype=np.int16)
    cargo = 0
    for slot, (x, y) in enumerate(positions):
        x, y = int(x), int(y)
        bag = inventories[slot] if slot < len(inventories) else {}
        held = {name: int(n) for name, n in bag.items() if name in ITEM_INDEX and n > 0}
        cargo += sum(held.values())
        under = tiles[0, y * BOARD_SIZE + x]
        units[slot, :5] = (1, int(slot == 0), x, y, int((x, y) in SHED_ACCESS))
        for name, n in held.items():
            units[slot, 5 + ITEM_INDEX[name]] = min(n, 100)
        units[slot, 17:] = (min(sum(held.values()), 100), slot, under[0], under[1], under[2], under[6], under[4], under[18], under[9], under[10], under[8],
                            int(bag.get("WHEAT", 0) > 0), int(bag.get("FERTILIZER", 0) > 0), int(any(bag.get(a, 0) > 0 for a in ANIMALS)))

    g = np.zeros(GLOBAL_D, dtype=np.float32)
    money, opp_money = float(own["money"]), float(opp["money"])
    shed = private.get("shed") or {}
    seeds = private.get("seeds") or {}
    shops = [s for s in town.get("unlocked_shops") or [] if s in SHOPS]
    hires = int(own.get("hires_today", 0))
    quadrants = {"own": len(own["unlocked_quadrants"]), "opp": len(opp["unlocked_quadrants"])}
    next_price = {side: LAND_PRICES[n - 1] if n <= len(LAND_PRICES) else 0 for side, n in quadrants.items()}

    g[G["day"]], g[G["hour"]], g[G["step"]], g[G["remaining"]] = day / 30, hour / 24, step / N_STEPS, (N_STEPS - step) / N_STEPS
    g[G["hour_mod4"]], g[G["last_hour"]], g[G["first_hour"]] = (hour % 4) / 4, hour == HOURS_PER_DAY - 1, hour == 0
    g[G[f"hour_oh_{hour}"]] = g[G[f"day_oh_{min(day, 29)}"]] = 1

    g[G["money"]], g[G["opp_money"]], g[G["money_diff"]] = money / 1e4, opp_money / 1e4, (money - opp_money) / 1e4
    g[G["money_log"]], g[G["opp_money_log"]] = np.log1p(max(money, 0.0)) / 12, np.log1p(max(opp_money, 0.0)) / 12
    for name in ITEMS:
        g[G[f"shed_{name}"]] = int(shed.get(name, 0)) / 100
    g[G["shed_total"]] = sum(int(shed.get(name, 0)) for name in ITEMS) / 100
    g[G["shed_room"]] = max(SHED_CAPACITY - sum(int(shed.get(name, 0)) for name in ITEMS), 0) / 100
    for name in CROPS:
        g[G[f"seed_{name}"]] = int(seeds.get(name, 0)) / 10

    g[G["hands"]], g[G["opp_hands"]] = len(own["hands"]) / 15, len(opp["hands"]) / 15
    g[G["hires_today"]], g[G["opp_hires_today"]], g[G["next_hire_cost"]] = hires / 15, int(opp.get("hires_today", 0)) / 15, HIRE_COSTS[min(hires, 19)] / 1000
    g[G["quadrants"]], g[G["opp_quadrants"]], g[G["next_land_price"]] = quadrants["own"] / 4, quadrants["opp"] / 4, next_price["own"] / 4000
    g[G["land_affordable"]], g[G["opp_land_affordable"]] = bool(next_price["own"] and money >= next_price["own"]), bool(next_price["opp"] and opp_money >= next_price["opp"])

    for name in PRODUCTS:
        g[G[f"price_{name}"]] = float(market["prices"][name]) / BASE_PRICE[name]
        g[G[f"inv_{name}"]] = (float(market["inventory"][name]) - MARKET_I0) / 1000
    g[G["sellable_value"]] = sum(float(market["prices"][name]) * int(shed.get(name, 0)) for name in PRODUCTS) / 1e4
    for shop in shops:
        g[G[f"shop_{shop}"]] += 0.5
        for product in SHOP_PRODUCTS[shop]:
            g[G[f"demand_{product}"]] += (2 if len(SHOP_PRODUCTS[shop]) == 1 else 1) / 4
    g[G["n_shops"]] = len(town.get("unlocked_shops") or []) / 8

    for prefix, counts in (("", own_counts), ("opp_", opp_counts)):
        for name, n in zip(CROPS, counts["plants"]):
            g[G[f"{prefix}plants_{name}"]] = n / 100
        for name, n in zip(ANIMALS, counts["animals"]):
            g[G[f"{prefix}animals_{name}"]] = n / 100
        for key in ("weeds", "empties", "structures_free"):
            g[G[prefix + key]] = counts[key] / 100
        g[G[f"{prefix}harvestable_units"]] = counts["harvestable"] / 100
    g[G["cargo_total"]] = cargo / 100

    return {"tiles": tiles, "units": units, "globals": g, "n_units": len(positions)}
