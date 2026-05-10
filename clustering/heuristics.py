"""
Post-clustering refinement heuristics.

move_order    — move one order from src to dst cluster if it reduces cost
swap_orders   — exchange two orders between clusters
split_cluster — split an overloaded cluster into two (mini K-Medoids with K=2)
merge_clusters — merge two clusters if result is feasible and cheaper
refine_clusters — iterate improvement passes until no change
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import numpy as np

from vrp.models import Order, Vehicle
from vrp.savings import estimate_pallets_byref
from clustering.models import Cluster, ClusterSolution
from clustering.cost import cluster_intra_cost
import config


# ── Helpers ────────────────────────────────────────────────────────────────────

def _max_capacity(cluster: Cluster, vehicles: List[Vehicle]) -> int:
    """Return the capacity of the vehicle assigned to this cluster, or max fleet cap."""
    if cluster.vehicle is not None:
        return cluster.vehicle.capacity_pallets
    return max(v.capacity_pallets for v in vehicles) if vehicles else 8


def _is_feasible(orders: List[Order], capacity: int) -> bool:
    return estimate_pallets_byref(orders) <= capacity


def _intra_delta(
    cluster: Cluster,
    extra_orders: List[Order],
    removed_orders: List[Order],
    D: np.ndarray,
    order_pos: Dict[str, int],
) -> float:
    """
    Approximate change in intra-cluster cost when orders are added/removed.
    Uses the existing medoid as reference (fast approximation).
    """
    mid_pos = order_pos.get(cluster.medoid_order_id, -1)
    if mid_pos < 0:
        return 0.0
    delta = 0.0
    for o in extra_orders:
        pos = order_pos.get(o.id, -1)
        if pos >= 0:
            delta += D[pos][mid_pos]
    for o in removed_orders:
        pos = order_pos.get(o.id, -1)
        if pos >= 0:
            delta -= D[pos][mid_pos]
    return delta


# ── Move ───────────────────────────────────────────────────────────────────────

def move_order(
    order: Order,
    src: Cluster,
    dst: Cluster,
    D: np.ndarray,
    order_pos: Dict[str, int],
    vehicles: List[Vehicle],
) -> bool:
    """
    Move `order` from `src` to `dst` if it reduces intra-cluster cost and
    dst remains feasible. Updates cluster stats in-place.
    Returns True if the move was executed.
    """
    if order not in src.orders:
        return False
    if len(src.orders) <= 1:
        return False  # don't empty a cluster

    # Feasibility check on dst
    dst_cap = _max_capacity(dst, vehicles)
    new_dst_orders = dst.orders + [order]
    if not _is_feasible(new_dst_orders, dst_cap):
        return False

    # Cost delta (intra only — fast proxy)
    delta = (
        _intra_delta(dst, [order], [], D, order_pos)
        + _intra_delta(src, [], [order], D, order_pos)
    )

    if delta < -1.0:  # strict improvement
        src.orders.remove(order)
        dst.orders.append(order)
        src.update_stats()
        dst.update_stats()
        return True

    return False


# ── Swap ───────────────────────────────────────────────────────────────────────

def swap_orders(
    oi: Order,
    oj: Order,
    ci: Cluster,
    cj: Cluster,
    D: np.ndarray,
    order_pos: Dict[str, int],
    vehicles: List[Vehicle],
) -> bool:
    """
    Swap `oi` (from ci) with `oj` (from cj) if it reduces combined intra cost
    and both clusters remain feasible.
    """
    if oi not in ci.orders or oj not in cj.orders:
        return False

    ci_cap = _max_capacity(ci, vehicles)
    cj_cap = _max_capacity(cj, vehicles)

    new_ci = [o for o in ci.orders if o is not oi] + [oj]
    new_cj = [o for o in cj.orders if o is not oj] + [oi]

    if not _is_feasible(new_ci, ci_cap) or not _is_feasible(new_cj, cj_cap):
        return False

    delta = (
        _intra_delta(ci, [oj], [oi], D, order_pos)
        + _intra_delta(cj, [oi], [oj], D, order_pos)
    )

    if delta < -1.0:
        ci.orders = new_ci
        cj.orders = new_cj
        ci.update_stats()
        cj.update_stats()
        return True

    return False


# ── Split ──────────────────────────────────────────────────────────────────────

def split_cluster(
    cluster: Cluster,
    D: np.ndarray,
    order_pos: Dict[str, int],
    vehicles: List[Vehicle],
    next_id: int = 0,
) -> Tuple[Cluster, Cluster]:
    """
    Split a cluster into two using mini K-Medoids with K=2.
    Returns (sub1, sub2). Does NOT mutate the original cluster.
    """
    orders = cluster.orders
    n = len(orders)
    cap = _max_capacity(cluster, vehicles) // 2 + 1

    if n <= 1:
        return cluster, Cluster(id=f"C{next_id:03d}", orders=[], medoid_order_id="")

    # Seed: two most distant orders
    best_pair = (0, 1)
    best_d = -1.0
    for i in range(n):
        for j in range(i + 1, n):
            pi = order_pos.get(orders[i].id, -1)
            pj = order_pos.get(orders[j].id, -1)
            if pi >= 0 and pj >= 0:
                d = D[pi][pj]
                if d > best_d:
                    best_d = d
                    best_pair = (i, j)

    m1_idx, m2_idx = best_pair
    members1: List[int] = [m1_idx]
    members2: List[int] = [m2_idx]

    for idx in range(n):
        if idx in (m1_idx, m2_idx):
            continue
        pi = order_pos.get(orders[idx].id, -1)
        pm1 = order_pos.get(orders[m1_idx].id, -1)
        pm2 = order_pos.get(orders[m2_idx].id, -1)
        d1 = D[pi][pm1] if pi >= 0 and pm1 >= 0 else float("inf")
        d2 = D[pi][pm2] if pi >= 0 and pm2 >= 0 else float("inf")
        if d1 <= d2:
            members1.append(idx)
        else:
            members2.append(idx)

    o1 = [orders[i] for i in members1]
    o2 = [orders[i] for i in members2]

    c1 = Cluster(id=cluster.id, orders=o1, medoid_order_id=orders[m1_idx].id)
    c2 = Cluster(id=f"C{next_id:03d}", orders=o2, medoid_order_id=orders[m2_idx].id)
    c1.update_stats()
    c2.update_stats()
    return c1, c2


# ── Merge ──────────────────────────────────────────────────────────────────────

def merge_clusters(
    ci: Cluster,
    cj: Cluster,
    D: np.ndarray,
    order_pos: Dict[str, int],
    vehicles: List[Vehicle],
) -> Optional[Cluster]:
    """
    Merge ci and cj if the combined cluster is feasible and has lower intra cost.
    Returns the merged Cluster, or None if infeasible / not beneficial.
    """
    combined = ci.orders + cj.orders
    cap = max(_max_capacity(ci, vehicles), _max_capacity(cj, vehicles))
    if not _is_feasible(combined, cap):
        return None

    # Compare intra costs
    before = cluster_intra_cost(ci, D, order_pos) + cluster_intra_cost(cj, D, order_pos)

    # New medoid = order minimising sum of distances to all combined members
    best_cost = np.inf
    best_mid = combined[0].id
    for o in combined:
        po = order_pos.get(o.id, -1)
        if po < 0:
            continue
        cost = sum(
            D[po][order_pos[oo.id]]
            for oo in combined
            if oo.id in order_pos and oo.id != o.id
        )
        if cost < best_cost:
            best_cost = cost
            best_mid = o.id

    merged = Cluster(id=ci.id, orders=combined, medoid_order_id=best_mid)
    merged.update_stats()

    after = cluster_intra_cost(merged, D, order_pos)

    if after < before - 1.0:
        merged.vehicle = ci.vehicle or cj.vehicle
        return merged

    return None


# ── Refinement loop ────────────────────────────────────────────────────────────

def refine_clusters(
    solution: ClusterSolution,
    D: np.ndarray,
    order_pos: Dict[str, int],
    vehicles: List[Vehicle],
    max_passes: int | None = None,
) -> ClusterSolution:
    """
    Iteratively improve cluster assignments using move and swap moves.
    Stops when no improvement is found or max_passes is reached.
    """
    if max_passes is None:
        max_passes = config.CLUSTERING_REFINE_PASSES

    clusters = solution.clusters

    for pass_num in range(max_passes):
        improved = False

        # ── Move: try moving each order to every other cluster ───────────────
        for src in list(clusters):
            for order in list(src.orders):
                for dst in clusters:
                    if dst is src:
                        continue
                    if move_order(order, src, dst, D, order_pos, vehicles):
                        improved = True
                        break

        # ── Swap: try swapping each pair ─────────────────────────────────────
        for i, ci in enumerate(clusters):
            for j, cj in enumerate(clusters):
                if j <= i:
                    continue
                for oi in list(ci.orders):
                    for oj in list(cj.orders):
                        if swap_orders(oi, oj, ci, cj, D, order_pos, vehicles):
                            improved = True
                            break

        if not improved:
            break

    # Try to re-insert unassigned orders
    if solution.unassigned:
        still_unassigned: List[Order] = []
        for order in solution.unassigned:
            placed = False
            for dst in sorted(clusters, key=lambda c: c.n_pallets):
                cap = _max_capacity(dst, vehicles)
                if _is_feasible(dst.orders + [order], cap):
                    dst.orders.append(order)
                    dst.update_stats()
                    placed = True
                    break
            if not placed:
                still_unassigned.append(order)
        solution.unassigned = still_unassigned

    return solution
