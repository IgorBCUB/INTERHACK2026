"""
Warehouse preparation algorithm — pallet-level planning.

Rules:
  1. Each warehouse trip = one FULL pallet (no partial trips)
  2. Each pallet belongs to ONE vehicle only
  3. Loading order = LIFO (last delivery loaded first → deepest in truck)
  4. Each pallet reserves RETURN_RESERVE fraction for next-day empties/returns
  5. Pallets grouped by zone when possible (minimize zone changes per trip)

Output: results_warehouse.json
"""
from __future__ import annotations
import json, os, sys, math
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import openpyxl
import config
from warehouse.classifier import classify
from data.loader import load_pallet_density, load_time_windows

DAYS     = ["16/03/2026","17/03/2026","18/03/2026","19/03/2026","20/03/2026"]
WEEKDAYS = ["Lunes","Martes","Miércoles","Jueves","Viernes"]

RETURN_RESERVE   = 0.15   # 15% of each pallet reserved for next-day returns
MAX_USABLE_FRAC  = 1.0 - RETURN_RESERVE   # 85% usable
MIXED_THRESHOLD  = 0.08   # category fraction below this → goes to mixed zone

# Weight profile per category (higher = heavier = goes to bottom of pallet/stack)
CATEGORY_WEIGHT = {
    "beer": 5, "water": 5, "softdrink": 4, "wine": 3,
    "dairy": 3, "coffee": 2, "generic": 2, "sweetener": 1,
    "packaging": 1, "empty_pkg": 1, "utensil": 1,
}

ZONE_COLORS = {
    "A":"#6366f1","B":"#06b6d4","C":"#10b981","D":"#f59e0b",
    "E":"#f43f5e","F":"#a855f7","G":"#ec4899","H":"#14b8a6",
}
CAT_LABEL = {
    "beer":"Cerveza","water":"Agua","wine":"Vino","softdrink":"Refresco",
    "dairy":"Lácteo","coffee":"Café","sweetener":"Edulcorante","packaging":"Embalaje",
    "empty_pkg":"Envase vacío","generic":"Genérico","utensil":"Utensilio",
}
POSITION_LABEL = {1:"Fons (darrere)", 2:"Mig-Fons", 3:"Mig", 4:"Mig-Davant", 5:"Davant"}


# ── Load today's orders per client ────────────────────────────────────────────

def load_day_orders(date_str: str, client_zones: dict, density: dict, tw_data: dict) -> dict:
    """Returns {client_id: order_dict} for the given day."""
    dt  = datetime.strptime(date_str, "%d/%m/%Y")
    dow = dt.isoweekday()

    orders: dict = {}
    wb = openpyxl.load_workbook(config.HACKATON_XLSX, read_only=True, data_only=True)
    ws = wb["Detalle entrega"]
    for row in ws.iter_rows(min_row=2, values_only=True):
        if str(row[0]) != date_str:
            continue
        cid   = str(row[10]) if row[10] else None
        mat   = str(row[6])  if row[6]  else ""
        denom = str(row[7])  if row[7]  else ""
        qty   = float(row[8]) if row[8] else 0.0
        if not cid or qty <= 0:
            continue
        if cid not in orders:
            tw_open, tw_close = 0, 86399
            if cid in tw_data and dow in tw_data[cid]:
                tw_open, tw_close = tw_data[cid][dow]
            orders[cid] = {
                "client_id":  cid,
                "name":       str(row[11]) if row[11] else cid,
                "city":       str(row[15]) if row[15] else "",
                "zone":       client_zones.get(cid, {}).get("zone", "H"),
                "cats":       defaultdict(float),
                "boxes":      defaultdict(float),
                "total_boxes": 0.0,
                "fragility":  0,
                "weight":     0,
                "tw_open":    tw_open,
                "tw_close":   tw_close,
            }
        cls = classify(mat, denom)
        pal = qty / (density.get(mat, 80) or 80)
        cat = cls["category"]
        orders[cid]["cats"][cat]   += pal
        orders[cid]["boxes"][cat]  += qty
        orders[cid]["total_boxes"] += qty
        orders[cid]["fragility"]    = max(orders[cid]["fragility"], cls["fragility_score"])
        orders[cid]["weight"]       = max(orders[cid]["weight"],    cls["weight_score"])
    wb.close()

    # Compute pallet fraction and zone
    for cid, o in orders.items():
        o["pallet_frac"] = sum(o["cats"].values())
        o["needs_pallet"] = o["pallet_frac"] >= 0.3
        # Determine primary and mixed categories
        total_pal = o["pallet_frac"] or 1.0
        o["main_cats"]  = {c: v for c, v in o["cats"].items() if v / total_pal >= MIXED_THRESHOLD}
        o["mixed_cats"] = {c: v for c, v in o["cats"].items() if v / total_pal <  MIXED_THRESHOLD}

    return orders


# ── Pallet planning per vehicle trip ─────────────────────────────────────────

def _primary_cat(order: dict) -> str:
    """Most voluminous category of an order."""
    cats = order.get("cats", {})
    return max(cats, key=cats.get) if cats else "generic"

def _stop_weight(stop: dict) -> int:
    """Stacking weight score for a stop (higher = heavier = goes to bottom layer)."""
    order_weight = stop.get("weight", 1) * 10
    top_cat = max(stop.get("main_cats", {"generic": 1}),
                  key=lambda k: stop.get("main_cats", {}).get(k, 0),
                  default="generic")
    return order_weight + CATEGORY_WEIGHT.get(top_cat, 2)


def plan_pallets_for_trip(stops_delivery_order: list, orders: dict, vehicle_cap: int = 6) -> list:
    """
    Given stops in DELIVERY order, plan pallets in LOADING order (LIFO).

    Stacking rules:
      1. Same product category stays on the same pallet (box↔box, can↔can).
      2. When a pallet must be mixed (overflow forced it), heavier products go to
         the bottom layer (sorted by weight score DESC within the pallet stop list).
      3. One full pallet slot is reserved for first-stop returns pickups.
      4. 15% of each pallet space reserved for next-day returns (RETURN_RESERVE).
    """
    # Reserve 1 physical pallet slot in the truck for return pickups at first stop
    effective_cap = max(1, vehicle_cap - 1)

    # LIFO: last delivery stop is loaded first (deepest in truck)
    stops_lifo = list(reversed(stops_delivery_order))

    pallets = []
    current_pallet: dict | None = None

    def new_pallet(pcat: str, zone: str) -> dict:
        return {
            "primary_cat": pcat,
            "zone":        zone,
            "used_frac":   0.0,
            "stops":       [],
            "fragile":     False,
            "heavy":       False,
            "categories":  defaultdict(float),
            "total_boxes": 0.0,
        }

    def flush(p: dict):
        if not (p and p["stops"]):
            return
        # Sort stops within pallet: heaviest first → goes to bottom layer
        p["stops"].sort(key=_stop_weight, reverse=True)
        pallets.append(p)

    for stop in stops_lifo:
        cid = stop["client_id"]
        o   = orders.get(cid)
        if not o:
            continue

        pcat      = _primary_cat(o)
        zone      = o["zone"]
        stop_frac = max(0.05, o["pallet_frac"])   # min 0.05 for box-only orders

        if current_pallet is None:
            current_pallet = new_pallet(pcat, zone)

        cat_change     = current_pallet["primary_cat"] != pcat
        would_overflow = current_pallet["used_frac"] + stop_frac > MAX_USABLE_FRAC

        if would_overflow and current_pallet["stops"]:
            # Pallet full → flush and start fresh (same-order rule always respected:
            # the whole client stop goes to the new pallet, never split mid-order)
            flush(current_pallet)
            current_pallet = new_pallet(pcat, zone)
        elif cat_change and current_pallet["used_frac"] >= 0.20:
            # Different primary category + pallet has some content →
            # prefer keeping same product type together (lower threshold = cleaner grouping)
            flush(current_pallet)
            current_pallet = new_pallet(pcat, zone)

        stop_entry = {
            "stop_num":    stop.get("stop_num", 0),
            "client_id":   cid,
            "name":        o["name"],
            "city":        o["city"],
            "boxes":       round(o["total_boxes"], 1),
            "pallet_frac": round(stop_frac, 3),
            "fragility":   o["fragility"],
            "weight":      o["weight"],
            "tw_open":     o["tw_open"],
            "tw_close":    o["tw_close"],
            "main_cats":   {c: round(v, 3) for c, v in o["main_cats"].items()},
            "mixed_cats":  {c: round(v, 3) for c, v in o["mixed_cats"].items()},
        }
        current_pallet["stops"].append(stop_entry)
        current_pallet["used_frac"]   += stop_frac
        current_pallet["total_boxes"] += o["total_boxes"]
        current_pallet["fragile"]      = current_pallet["fragile"] or o["fragility"] >= 2
        current_pallet["heavy"]        = current_pallet["heavy"]   or o["weight"]    >= 2
        for cat, vol in o["cats"].items():
            current_pallet["categories"][cat] += vol

    flush(current_pallet)

    # Finalize pallets: number (1 = deepest = first loaded = last delivery)
    n = len(pallets)
    for i, p in enumerate(pallets):
        depth = i + 1
        pos_bucket = math.ceil(depth / max(n, 1) * 5)
        # Detect mixed pallet (more than one primary category present)
        cats_present = [c for c, v in p["categories"].items() if v > 0]
        is_mixed = len(cats_present) > 1
        # Stacking warning: fragile AND heavy on same pallet
        stack_warn = p["fragile"] and p["heavy"]

        p["pallet_num"]       = depth
        p["loading_depth"]    = depth
        p["loading_position"] = POSITION_LABEL.get(pos_bucket, "Mig")
        p["reserved_frac"]    = round(RETURN_RESERVE, 2)
        p["used_frac"]        = round(p["used_frac"], 3)
        p["total_frac"]       = round(p["used_frac"] + RETURN_RESERVE, 3)
        p["total_boxes"]      = round(p["total_boxes"], 1)
        p["categories"]       = {c: round(v, 3) for c, v in p["categories"].items()}
        p["primary_cat"]      = max(p["categories"], key=p["categories"].get) if p["categories"] else "generic"
        p["is_mixed"]         = is_mixed
        p["stack_warning"]    = stack_warn
        p["stack_order"]      = "weight_asc"   # stops already sorted heavy→bottom
        p["color"]            = ZONE_COLORS.get(p["zone"], "#475569")
        p["return_reserve_slot"] = (i == n - 1)   # last pallet is nearest door, next to reserve slot

    return pallets


# ── Mixed zone aggregation ────────────────────────────────────────────────────

def build_mixed_zone(orders: dict) -> dict:
    """Aggregate all minor-category items across all orders → mixed zone."""
    by_cat: dict = defaultdict(lambda: {"boxes": 0.0, "pallets": 0.0, "clients": []})
    items = []
    for cid, o in orders.items():
        for cat, vol in o["mixed_cats"].items():
            boxes = o["boxes"].get(cat, 0.0)
            by_cat[cat]["boxes"]   += boxes
            by_cat[cat]["pallets"] += vol
            by_cat[cat]["clients"].append(o["name"])
            items.append({
                "client": o["name"], "city": o["city"], "zone": o["zone"],
                "category": cat, "boxes": round(boxes, 1), "pallets": round(vol, 3),
            })

    return {
        "n_items":       len(items),
        "total_boxes":   round(sum(i["boxes"] for i in items), 1),
        "total_pallets": round(sum(v["pallets"] for v in by_cat.values()), 2),
        "by_category": {
            cat: {
                "boxes":   round(v["boxes"],   1),
                "pallets": round(v["pallets"], 3),
                "n_clients": len(v["clients"]),
            }
            for cat, v in sorted(by_cat.items(), key=lambda x: -x[1]["boxes"])
        },
        "items": items[:60],
    }


# ── Zone overview (for global warehouse view) ─────────────────────────────────

def build_zone_overview(orders: dict, zone_profiles: dict) -> dict:
    zone_orders: dict = defaultdict(list)
    for cid, o in orders.items():
        zone_orders[o["zone"]].append(o)

    zones_out = {}
    for zone in "ABCDEFGH":
        zorders = zone_orders.get(zone, [])
        cat_pallets: dict = defaultdict(float)
        cat_boxes:   dict = defaultdict(float)
        for o in zorders:
            for cat, vol in o["main_cats"].items():
                cat_pallets[cat] += vol
                cat_boxes[cat]   += o["boxes"].get(cat, 0.0)

        total_pal = sum(cat_pallets.values())
        cat_summary = [
            {"category": c, "pallets": round(cat_pallets[c], 2), "boxes": round(cat_boxes[c], 1),
             "pct": round(cat_pallets[c] / total_pal * 100, 1) if total_pal else 0}
            for c in sorted(cat_pallets, key=cat_pallets.get, reverse=True)
        ]
        zones_out[zone] = {
            "zone":            zone,
            "color":           ZONE_COLORS[zone],
            "n_clients_total": zone_profiles.get(zone, {}).get("n_clients", 0),
            "n_clients_today": len(zorders),
            "n_pallets":       round(total_pal, 1),
            "has_fragile":     any(o["fragility"] >= 2 for o in zorders),
            "has_heavy":       any(o["weight"]    >= 2 for o in zorders),
            "stable_primary":  zone_profiles.get(zone, {}).get("primary_cats", []),
            "stable_fracs":    zone_profiles.get(zone, {}).get("volume_fracs", {}),
            "cat_summary":     cat_summary,
        }
    return zones_out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading stable client zones...")
    with open("data/client_zones.json") as f:
        zone_data = json.load(f)
    client_zones  = zone_data["clients"]
    zone_profiles = zone_data["zone_profiles"]

    print("Loading clustering routes...")
    with open("results_clustering.json") as f:
        clustering = json.load(f)

    density = load_pallet_density()
    tw_data = load_time_windows(config.HORARIOS_XLSX)

    results = []

    for day_idx, day_clust in enumerate(clustering):
        date_str = day_clust["date"]
        weekday  = WEEKDAYS[day_idx]
        print(f"\n  {date_str} ({weekday})")

        # Load enriched orders for this day
        orders = load_day_orders(date_str, client_zones, density, tw_data)

        # Zone overview for the global warehouse map
        zone_overview = build_zone_overview(orders, zone_profiles)
        mixed_zone    = build_mixed_zone(orders)

        # ── Plan pallets per vehicle ─────────────────────────────────────────
        vehicle_plans = []
        total_pallets_planned = 0

        for route in day_clust["routes"]:
            vid   = route["vehicle_id"]
            vtype = route["vehicle_type"]
            cap   = route["capacity_pallets"]

            trip_plans = []
            for trip in route["trips"]:
                # Stops in delivery order (as they appear in the route)
                stops_delivery = [
                    {"client_id": s["client_id"], "stop_num": i + 1, **s}
                    for i, s in enumerate(trip["stops"])
                ]

                pallets = plan_pallets_for_trip(stops_delivery, orders, cap)
                trip_plans.append({
                    "trip_num":       trip["trip_number"],
                    "is_reload":      trip["is_reload"],
                    "warehouse_zone": trip.get("warehouse_zone", "?"),
                    "n_stops":        len(stops_delivery),
                    "n_pallets":      len(pallets),
                    "pallets":        pallets,
                })
                total_pallets_planned += len(pallets)

            # Summary stats for this vehicle
            all_pallets = [p for t in trip_plans for p in t["pallets"]]
            vehicle_plans.append({
                "vehicle_id":    vid,
                "vehicle_type":  vtype,
                "driver_name":   route["driver_name"],
                "capacity_pallets": cap,
                "n_trips":       len(trip_plans),
                "n_pallets":     len(all_pallets),
                "trips":         trip_plans,
            })

        # Day summary
        total_orders = sum(z["n_clients_today"] for z in zone_overview.values())
        active_zones = sum(1 for z in zone_overview.values() if z["n_clients_today"] > 0)

        print(f"    {total_orders} pedidos · {total_pallets_planned} palets planificats · "
              f"{active_zones}/8 zones · {mixed_zone['n_items']} items zona mixta")

        results.append({
            "date":                date_str,
            "weekday":             weekday,
            "total_orders":        total_orders,
            "total_pallets":       total_pallets_planned,
            "active_zones":        active_zones,
            "zones":               zone_overview,
            "vehicle_plans":       vehicle_plans,
            "mixed_zone":          mixed_zone,
        })

    with open("results_warehouse.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n✓ results_warehouse.json")


if __name__ == "__main__":
    main()
