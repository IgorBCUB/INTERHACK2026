"""
Unit tests for the logistics clustering module.

Run with:
  python3 -m pytest clustering/tests.py -v
  python3 clustering/tests.py   (standalone)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from vrp.models import Order, Vehicle
from clustering.models import Cluster, ClusterSolution
from clustering.distance import _logistics_penalty, build_logistics_matrix
from clustering.algorithm import cluster_orders
from clustering.warehouse_zones import assign_warehouse_zones


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _make_order(oid, city, pallet_frac, fragility=0, weight=0,
                tw_open=0, tw_close=86399, returnables=False, priority=3) -> Order:
    return Order(
        id=oid, client_id=oid, client_name=city,
        address="", postal_code="", city=city,
        lat=0.0, lon=0.0,
        volume_boxes=pallet_frac * 60,
        tw_open=tw_open, tw_close=tw_close,
        priority=priority,
        fragility_score=fragility,
        weight_score=weight,
        has_returnables=returnables,
        all_stackable=(fragility < 2),
        est_pallet_fraction=pallet_frac,
        needs_pallet=(pallet_frac >= 0.3),
    )


def _make_vehicle(vid, cap=6, vtype="truck_6") -> Vehicle:
    return Vehicle(
        id=vid, driver_name=vid,
        vehicle_type=vtype,
        capacity_pallets=cap,
        has_side_access=(vtype != "van_3"),
        lifo_penalty_multiplier=3.0 if vtype == "van_3" else 1.0,
    )


def _simple_time_matrix(n: int, seconds_per_hop=600.0) -> list:
    """NxN matrix: T[i][j] = |i-j| * seconds_per_hop."""
    return [[abs(i - j) * seconds_per_hop for j in range(n)] for i in range(n)]


def _simple_fuel_matrix(n: int, litres_per_hop=1.0) -> list:
    return [[abs(i - j) * litres_per_hop for j in range(n)] for i in range(n)]


# ── Test 1: logistics penalty ──────────────────────────────────────────────────

def test_logistics_penalty():
    fragile = _make_order("O_frag", "Vic", 1.0, fragility=3, weight=0)
    heavy   = _make_order("O_heavy", "Vic", 1.0, fragility=0, weight=3)
    normal  = _make_order("O_norm", "Vic", 1.0, fragility=0, weight=0)

    pen_frag_heavy = _logistics_penalty(fragile, heavy)
    pen_norm_norm  = _logistics_penalty(normal, normal)
    pen_return     = _logistics_penalty(
        _make_order("R", "V", 0.5, returnables=True), normal
    )

    assert pen_frag_heavy >= 1800, f"Expected ≥1800, got {pen_frag_heavy}"
    assert pen_norm_norm == 0.0, f"Expected 0, got {pen_norm_norm}"
    assert pen_return >= 300, f"Expected ≥300, got {pen_return}"

    print("✓ test_logistics_penalty")


# ── Test 2: cluster capacity constraint ───────────────────────────────────────

def test_cluster_capacity():
    """A truck_6 (6 pallets) should not accept an order that would exceed capacity."""
    orders = [_make_order(f"O{i}", "BCN", 2.0) for i in range(5)]  # 5×2=10 pallets needed
    vehicles = [_make_vehicle("V1", cap=6), _make_vehicle("V2", cap=6)]

    n = len(orders)
    # Simple symmetric time+fuel matrices (n+1 × n+1, +1 for depot)
    size = n + 1
    TM = _simple_time_matrix(size)
    FM = _simple_fuel_matrix(size)

    matrix_idx = {o.id: i + 1 for i, o in enumerate(orders)}
    order_pos  = {o.id: i for i, o in enumerate(orders)}
    depot_d = np.array([float((i + 1) * 600) for i in range(n)])

    D = build_logistics_matrix(TM, FM, orders, matrix_idx)
    solution = cluster_orders(orders, vehicles, D, order_pos, depot_d)

    for cluster in solution.clusters:
        if cluster.vehicle:
            assert cluster.n_pallets <= cluster.vehicle.capacity_pallets, (
                f"Cluster {cluster.id}: {cluster.n_pallets} pallets > "
                f"{cluster.vehicle.capacity_pallets} capacity"
            )

    print("✓ test_cluster_capacity")


# ── Test 3: K-Medoids convergence ─────────────────────────────────────────────

def test_kmedoids_convergence():
    """10 orders, 2 vehicles → converges and all orders are assigned."""
    orders = [_make_order(f"O{i}", f"City{i % 3}", 0.5) for i in range(10)]
    vehicles = [_make_vehicle(f"V{i}", cap=6) for i in range(2)]

    n = len(orders)
    size = n + 1
    TM = _simple_time_matrix(size)
    FM = _simple_fuel_matrix(size)

    matrix_idx = {o.id: i + 1 for i, o in enumerate(orders)}
    order_pos  = {o.id: i for i, o in enumerate(orders)}
    depot_d = np.ones(n) * 600.0

    D = build_logistics_matrix(TM, FM, orders, matrix_idx)
    solution = cluster_orders(orders, vehicles, D, order_pos, depot_d)

    total_assigned = sum(len(c.orders) for c in solution.clusters)
    total = total_assigned + len(solution.unassigned)
    assert total == n, f"Lost orders: expected {n}, got {total}"
    print(f"✓ test_kmedoids_convergence ({len(solution.clusters)} clusters, "
          f"{len(solution.unassigned)} unassigned)")


# ── Test 4: Example from the plan (5 orders, 2 vehicles) ─────────────────────

def test_example_5_orders():
    """
    Vic cluster (O1+O2) and Granollers cluster (O3+O4+O5) should form.
    O1=fragile beer, O2=water (Vic), O3=beer (Gran), O4=fragile wine (Gran), O5=dairy (Mollet)
    """
    O1 = _make_order("O1", "Vic",        2.0, fragility=2, weight=1, priority=2)
    O2 = _make_order("O2", "Vic",        1.0, fragility=0, weight=2)
    O3 = _make_order("O3", "Granollers", 3.0, fragility=0, weight=1)
    O4 = _make_order("O4", "Granollers", 2.0, fragility=3, weight=1)
    O5 = _make_order("O5", "Mollet",     1.0, fragility=0, weight=0)

    orders = [O1, O2, O3, O4, O5]
    vehicles = [_make_vehicle("VA", cap=6), _make_vehicle("VB", cap=6)]

    # Custom time matrix: Vic ↔ Vic = 600s, Vic ↔ Gran = 1800s, etc.
    # Indices: 0=depot, 1=O1, 2=O2, 3=O3, 4=O4, 5=O5
    TM = [
        [0,    1800, 1800, 1800, 1800, 300 ],  # depot
        [1800, 0,    600,  1800, 1800, 1500],  # O1 (Vic)
        [1800, 600,  0,    1800, 1800, 1500],  # O2 (Vic)
        [1800, 1800, 1800, 0,    300,  1500],  # O3 (Granollers)
        [1800, 1800, 1800, 300,  0,    1500],  # O4 (Granollers)
        [300,  1500, 1500, 1500, 1500, 0   ],  # O5 (Mollet)
    ]
    FM = [[t / 600.0 * 1.0 for t in row] for row in TM]

    matrix_idx = {o.id: i + 1 for i, o in enumerate(orders)}
    order_pos  = {o.id: i for i, o in enumerate(orders)}
    depot_d = np.array([float(TM[0][i + 1]) for i in range(len(orders))])

    D = build_logistics_matrix(TM, FM, orders, matrix_idx)
    solution = cluster_orders(orders, vehicles, D, order_pos, depot_d)

    total = sum(len(c.orders) for c in solution.clusters) + len(solution.unassigned)
    assert total == 5, f"Lost orders: {total}/5"
    for c in solution.clusters:
        if c.vehicle:
            assert c.n_pallets <= c.vehicle.capacity_pallets

    print(f"✓ test_example_5_orders ({len(solution.clusters)} clusters)")
    for c in solution.clusters:
        ids = [o.id for o in c.orders]
        print(f"  {c.id}: {ids}  → pallets={c.n_pallets}  zone={c.warehouse_zone}")


# ── Test 5: Warehouse zone assignment ─────────────────────────────────────────

def test_warehouse_zone_assignment():
    """16 clusters → zones cycling A–H twice."""
    clusters = []
    for i in range(16):
        c = Cluster(id=f"C{i:03d}")
        c.orders = [_make_order(f"X{i}", "City", 0.5, tw_open=i * 1800)]
        c.update_stats()
        clusters.append(c)

    assign_warehouse_zones(clusters)
    zones = [c.warehouse_zone for c in clusters]
    assert len(set(zones)) == 8, f"Expected 8 distinct zones, got {set(zones)}"
    # First cluster (earliest TW) should be zone A
    earliest = min(clusters, key=lambda c: c.orders[0].tw_open)
    assert earliest.warehouse_zone == "A", f"Earliest should be A, got {earliest.warehouse_zone}"
    print(f"✓ test_warehouse_zone_assignment  zones={sorted(set(zones))}")


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_logistics_penalty,
        test_cluster_capacity,
        test_kmedoids_convergence,
        test_example_5_orders,
        test_warehouse_zone_assignment,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
