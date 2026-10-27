# Encode/decode 10 market and 16 unit rows; -1 marks absent fields.
import numpy as np

from features import ANIMALS, CONFIG, CROPS, ITEMS, N_UNITS, PRODUCTS

UNIT_OPS = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "PICKUP", "PLACE", "DROP", "FEED", "CARE", "COLLECT_FERTILIZER"]
MARKET_OPS = ["END", "NOOP", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"]
N_MARKET, N_QTY, N_ITEMS = CONFIG["maxMarketOrdersPerTurn"], 101, len(ITEMS)
N_ROWS = N_MARKET + N_UNITS
ITEMS_OF = {"PLANT": CROPS, "PICKUP": ITEMS, "PLACE": ITEMS, "BUY_SEED": CROPS, "BUY_PRODUCT": ["WHEAT", "FERTILIZER"], "BUY_ANIMAL": ANIMALS, "SELL": PRODUCTS}
QTY_OPS = {"PICKUP", "PLACE", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}


def support(ops):
    # Valid items per operation.
    table = np.zeros((len(ops), N_ITEMS), dtype=bool)
    for i, op in enumerate(ops):
        table[i, [ITEMS.index(name) for name in ITEMS_OF.get(op, [])]] = True
    return table


UNIT_ITEMS, MARKET_ITEMS = support(UNIT_OPS), support(MARKET_OPS)
UNIT_NEEDS_ITEM, MARKET_NEEDS_ITEM = UNIT_ITEMS.any(1), MARKET_ITEMS.any(1)
UNIT_NEEDS_QTY = np.array([op in QTY_OPS for op in UNIT_OPS])
MARKET_NEEDS_QTY = np.array([op in QTY_OPS for op in MARKET_OPS])


def clamp_qty(value, default):
    try:
        return min(max(int(value), 0), N_QTY - 1)
    except (TypeError, ValueError):
        return default


def unit_row(row):
    # Encode a unit action; invalid actions become PASS.
    if not (isinstance(row, list) and row and row[0] in UNIT_OPS):
        return 0, -1, -1
    op = row[0]
    if op not in ITEMS_OF:
        return UNIT_OPS.index(op), -1, -1
    if len(row) < 2 or row[1] not in ITEMS_OF[op]:
        return 0, -1, -1
    return UNIT_OPS.index(op), ITEMS.index(row[1]), (clamp_qty(row[2], 1) if len(row) > 2 else 1) if op in QTY_OPS else -1


def market_row(row):
    # Encode a market order; invalid orders become NOOP.
    if not (isinstance(row, list) and row and row[0] in MARKET_OPS[2:]):
        return 1, -1, -1
    op = row[0]
    if op not in ITEMS_OF:
        return MARKET_OPS.index(op), -1, -1
    qty = clamp_qty(row[2], -1) if len(row) > 2 else -1
    if len(row) < 2 or row[1] not in ITEMS_OF[op] or qty <= 0:
        return 1, -1, -1
    return MARKET_OPS.index(op), ITEMS.index(row[1]), qty


def encode(action, n_units):
    # Encode an action as (26, 3) int16 tokens.
    tokens = np.full((N_ROWS, 3), -1, dtype=np.int16)
    action = action if isinstance(action, dict) else {}
    orders = action.get("market", [])
    orders = list(orders)[:N_MARKET] if isinstance(orders, list) else []
    for i, row in enumerate(orders):
        tokens[i] = market_row(row)
    if len(orders) < N_MARKET:
        tokens[len(orders)] = (0, -1, -1)
    hands = action.get("hands", [])
    units = [action.get("farmer", ["PASS"]), *(hands if isinstance(hands, list) else [])]
    for slot in range(min(n_units, N_UNITS)):
        tokens[N_MARKET + slot] = unit_row(units[slot] if slot < len(units) else None)
    return tokens


def decode(tokens, n_units):
    # Decode tokens into an engine action.
    orders = []
    for op, item, qty in tokens[:N_MARKET].tolist():
        if op <= 0:
            break
        name = MARKET_OPS[op]
        if name == "NOOP":
            orders.append([])
        elif name in ITEMS_OF:
            orders.append([name, ITEMS[item], qty] if item >= 0 and qty > 0 else [])
        else:
            orders.append([name])
    units = []
    for op, item, qty in tokens[N_MARKET:N_MARKET + min(n_units, N_UNITS)].tolist():
        name = UNIT_OPS[max(op, 0)]
        if name not in ITEMS_OF:
            units.append([name])
        elif item < 0:
            units.append(["PASS"])
        elif name in QTY_OPS:
            units.append([name, ITEMS[item], qty if qty >= 0 else 1])
        else:
            units.append([name, ITEMS[item]])
    return {"farmer": units[0] if units else ["PASS"], "hands": units[1:], "market": orders}
