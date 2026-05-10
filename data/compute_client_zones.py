"""
Stable client clustering based on PRODUCT MIX — computed ONCE for all 1203 clients.

Features: what each client orders (category volume fractions across all history).
Result: X permanent clusters, each client always in the same cluster.

Output: data/client_zones.json
Run once:  python3 data/compute_client_zones.py
"""
from __future__ import annotations
import json, sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from collections import defaultdict

import openpyxl
import config
from warehouse.classifier import classify
from data.loader import load_pallet_density

ZONE_LABELS = list("ABCDEFGH")
K = 8

# Product categories used as clustering features
CATEGORIES = [
    "beer", "water", "wine", "softdrink", "dairy",
    "coffee", "sweetener", "packaging", "empty_pkg",
]

# Threshold: if a category represents < this fraction of a client's total → minor (→ mixed zone)
MIXED_THRESHOLD = 0.08


def build_client_profiles() -> tuple[list, np.ndarray, dict, dict]:
    """Returns (client_ids, feature_matrix, geo, raw_cat_volumes)."""
    with open("data/geocodes.json") as f:
        geo = json.load(f)

    density = load_pallet_density()

    # Aggregate ALL historical data per client
    cat_vol: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total_boxes: dict[str, float] = defaultdict(float)

    wb = openpyxl.load_workbook(config.HACKATON_XLSX, read_only=True, data_only=True)
    ws = wb["Detalle entrega"]
    for row in ws.iter_rows(min_row=2, values_only=True):
        cid   = str(row[10]) if row[10] else None
        mat   = str(row[6])  if row[6]  else ""
        denom = str(row[7])  if row[7]  else ""
        qty   = float(row[8]) if row[8] else 0.0
        if not cid or cid not in geo or qty <= 0:
            continue
        cls = classify(mat, denom)
        pal = qty / (density.get(mat, 80) or 80)
        cat_vol[cid][cls["category"]] += pal
        total_boxes[cid] += qty
    wb.close()

    # Build feature matrix: category FRACTIONS (what % of each client's volume is each category)
    client_ids = [c for c in geo if c in cat_vol]   # only clients with order history
    n = len(client_ids)
    X = np.zeros((n, len(CATEGORIES)), dtype=np.float64)
    for i, cid in enumerate(client_ids):
        total = sum(cat_vol[cid].values()) or 1.0
        for j, cat in enumerate(CATEGORIES):
            X[i, j] = cat_vol[cid].get(cat, 0.0) / total

    return client_ids, X, geo, cat_vol, total_boxes


def kmeans_pp(X: np.ndarray, k: int, seed: int = 42, n_init: int = 15) -> np.ndarray:
    """K-Means++ with multiple restarts. Returns label array."""
    rng = np.random.default_rng(seed)
    n = len(X)
    best_labels, best_inertia = None, np.inf

    for _ in range(n_init):
        # K-Means++ seeding
        centers = [X[rng.integers(n)].copy()]
        for _ in range(k - 1):
            dists = np.array([
                min(float(np.sum((x - c) ** 2)) for c in centers)
                for x in X
            ])
            probs = dists / (dists.sum() + 1e-12)
            centers.append(X[rng.choice(n, p=probs)].copy())
        centers = np.array(centers)

        labels = np.zeros(n, dtype=int)
        for _ in range(300):
            dists = np.linalg.norm(X[:, None] - centers[None], axis=2)  # (n, k)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centers[j] = X[mask].mean(axis=0)

        inertia = float(sum(
            np.sum((X[labels == j] - centers[j]) ** 2) for j in range(k)
        ))
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.copy()

    return best_labels


def describe_cluster(cid_list: list, cat_vol: dict) -> dict:
    """Compute aggregate product profile for a cluster."""
    total: dict[str, float] = defaultdict(float)
    for cid in cid_list:
        for cat, vol in cat_vol[cid].items():
            total[cat] += vol
    grand = sum(total.values()) or 1.0
    fracs = {cat: round(vol / grand, 3) for cat, vol in total.items() if vol > 0}
    primary = sorted(fracs, key=fracs.get, reverse=True)[:3]
    return {"volume_fracs": fracs, "primary_cats": primary, "total_pallets": round(grand, 1)}


def main():
    print("Loading all historical order data...")
    client_ids, X, geo, cat_vol, total_boxes = build_client_profiles()
    print(f"  {len(client_ids)} clients with order history")

    print(f"Running K-Means++ (K={K}) on product-mix features...")
    labels = kmeans_pp(X, K)

    # Sort clusters by their dominant category (beer-heavy first, then water, wine…)
    # so labels are human-readable and stable
    cat_order = {c: i for i, c in enumerate(CATEGORIES)}
    cluster_primary: list[tuple] = []
    for k in range(K):
        mask = labels == k
        if not mask.any():
            cluster_primary.append((999, k))
            continue
        centroid = X[mask].mean(axis=0)
        dom = int(np.argmax(centroid))
        cluster_primary.append((dom, k))
    cluster_primary.sort()
    relabel = {old_k: ZONE_LABELS[new_k] for new_k, (_, old_k) in enumerate(cluster_primary)}

    # Build output per client
    result: dict = {}
    cluster_clients: dict[str, list] = {z: [] for z in ZONE_LABELS}

    for i, cid in enumerate(client_ids):
        zone = relabel[labels[i]]
        total_pal = sum(cat_vol[cid].values()) or 1.0
        fracs = {cat: round(vol / total_pal, 3) for cat, vol in cat_vol[cid].items() if vol > 0}
        primary = sorted(fracs, key=fracs.get, reverse=True)[:2]
        minor   = [cat for cat, frac in fracs.items() if frac < MIXED_THRESHOLD]

        result[cid] = {
            "zone":          zone,
            "lat":           geo[cid]["lat"],
            "lon":           geo[cid]["lon"],
            "name":          geo[cid].get("name", ""),
            "city":          geo[cid].get("city", ""),
            "primary_cats":  primary,
            "cat_fractions": fracs,
            "minor_cats":    minor,
            "total_boxes":   round(total_boxes[cid], 1),
        }
        cluster_clients[zone].append(cid)

    # Compute stable zone profiles
    zone_profiles: dict = {}
    for zone in ZONE_LABELS:
        cids = cluster_clients[zone]
        profile = describe_cluster(cids, cat_vol)
        zone_profiles[zone] = {
            "n_clients":    len(cids),
            "primary_cats": profile["primary_cats"],
            "volume_fracs": profile["volume_fracs"],
            "total_pallets_historical": profile["total_pallets"],
        }

    out = {
        "clients":      result,
        "zone_profiles": zone_profiles,
        "k": K,
        "method": "product_mix_kmeans",
        "categories": CATEGORIES,
        "mixed_threshold": MIXED_THRESHOLD,
    }
    with open("data/client_zones.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n✓ data/client_zones.json")
    print(f"\n{'Zona':<6} {'Clientes':>8} {'Categorías principales':<40} {'Palets hist.'}")
    print("─" * 75)
    for zone in ZONE_LABELS:
        p  = zone_profiles[zone]
        cats = " + ".join(p["primary_cats"])
        pals = p["total_pallets_historical"]
        print(f"  {zone}     {p['n_clients']:>6}   {cats:<40} {pals:.0f}P")

    # Clients not in order history (no orders in data)
    all_geo = set(geo.keys())
    with_orders = set(result.keys())
    without = all_geo - with_orders
    if without:
        print(f"\n  ⚠ {len(without)} clients in geocodes.json with no order history → not clustered")


if __name__ == "__main__":
    main()
