"""
Constrained K-Medoids clustering for VRP.

Algorithm:
  1. KMedoids++ initialisation (capacity-aware seeding)
  2. Assignment with capacity constraints (pallets ≤ vehicle.capacity_pallets)
  3. Medoid update (minimise intra-cluster distances)
  4. Iterate until convergence or max_iter
  5. Repair unassigned orders (greedy insertion)
  6. Assign vehicles to clusters (van_3 last, truck_8 for large, truck_6 default)
"""
from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from vrp.models import Order, Vehicle
from vrp.savings import estimate_pallets_byref
from clustering.models import Cluster, ClusterSolution
import config


# ── Vehicle assignment ─────────────────────────────────────────────────────────

def _assign_vehicles_to_clusters(
    clusters: List[Cluster],
    vehicles: List[Vehicle],
) -> None:
    """
    Assign vehicles to clusters, spreading load evenly across the whole fleet.
    Uses trip count (not time) as the load metric to avoid over-conservative estimates.
    """
    vehicle_trip_counts: dict = {v.id: 0 for v in vehicles}
    n_clusters = len(clusters)
    n_vehicles = len(vehicles)
    # Each vehicle gets ceil(n_clusters / n_vehicles) + 1 trips max to absorb any imbalance
    max_trips_per_vehicle = max(2, int(np.ceil(n_clusters / max(n_vehicles, 1))) + 1)

    sorted_clusters = sorted(clusters, key=lambda c: -c.n_pallets)

    for cluster in sorted_clusters:
        v = _pick_vehicle_reload(cluster, vehicles, vehicle_trip_counts, max_trips_per_vehicle)
        if v is not None:
            cluster.vehicle = v
            vehicle_trip_counts[v.id] += 1
        else:
            cluster.vehicle = None


def _pick_vehicle_reload(
    cluster: Cluster,
    vehicles: List[Vehicle],
    vehicle_trip_counts: dict,
    max_trips: int,
) -> Optional[Vehicle]:
    """
    Pick the best vehicle for this cluster using trip count as load metric.
    Sorted by trip count ascending → least-busy vehicle first.
    """
    n = cluster.n_pallets
    sorted_by_trips = sorted(vehicles, key=lambda x: vehicle_trip_counts.get(x.id, 0))

    # van_3: only tiny priority non-fragile
    if n <= 2 and cluster.has_priority and not cluster.has_fragile:
        for v in sorted_by_trips:
            if v.vehicle_type == "van_3" and n <= v.capacity_pallets:
                if vehicle_trip_counts.get(v.id, 0) < max_trips:
                    return v

    # Prefer truck_6 for ≤6 pallets
    if n <= 6:
        for v in sorted_by_trips:
            if v.vehicle_type == "truck_6" and n <= v.capacity_pallets:
                if vehicle_trip_counts.get(v.id, 0) < max_trips:
                    return v

    # truck_8 for larger or fallback
    for v in sorted_by_trips:
        if v.vehicle_type == "truck_8" and n <= v.capacity_pallets:
            if vehicle_trip_counts.get(v.id, 0) < max_trips:
                return v

    # Any vehicle with capacity under limit
    for v in sorted_by_trips:
        if n <= v.capacity_pallets and vehicle_trip_counts.get(v.id, 0) < max_trips:
            return v

    # Force-assign to least-busy vehicle (overloaded — cost function penalises it)
    for v in sorted_by_trips:
        if vehicle_trip_counts.get(v.id, 0) < max_trips:
            return v

    # Last resort: any vehicle regardless of trip count
    return sorted_by_trips[0] if sorted_by_trips else None


def _pick_vehicle_for_cluster(
    cluster: Cluster,
    vehicles: List[Vehicle],
) -> Optional[Vehicle]:
    """Choose the most appropriate available vehicle for this cluster."""
    n = cluster.n_pallets

    # van_3: only small, priority, non-fragile routes
    if n <= 2 and cluster.has_priority and not cluster.has_fragile:
        for v in vehicles:
            if v.vehicle_type == "van_3" and n <= v.capacity_pallets:
                return v

    # truck_8 for large routes
    if n > 6:
        for v in sorted(vehicles, key=lambda x: x.capacity_pallets, reverse=True):
            if v.vehicle_type == "truck_8" and n <= v.capacity_pallets:
                return v
        # Fallback to any vehicle that fits
        for v in sorted(vehicles, key=lambda x: x.capacity_pallets, reverse=True):
            if n <= v.capacity_pallets:
                return v
        return None

    # truck_6 default
    for v in vehicles:
        if v.vehicle_type == "truck_6" and n <= v.capacity_pallets:
            return v

    # Final fallback
    for v in sorted(vehicles, key=lambda x: x.capacity_pallets, reverse=True):
        if n <= v.capacity_pallets:
            return v

    return None


# ── K-Medoids core ─────────────────────────────────────────────────────────────

def _kmedoids_init(
    D: np.ndarray,
    depot_d: np.ndarray,
    K: int,
    rng: random.Random,
) -> List[int]:
    """
    KMedoids++ initialisation.
    First medoid = order farthest from depot.
    Subsequent medoids: sample proportional to squared min-distance to existing medoids.
    """
    n = len(depot_d)
    if K >= n:
        return list(range(n))

    medoids: List[int] = []
    # Seed: order farthest from depot
    medoids.append(int(np.argmax(depot_d)))

    for _ in range(K - 1):
        # Min distance to nearest medoid for each non-medoid
        min_d = np.full(n, np.inf)
        for m in medoids:
            min_d = np.minimum(min_d, D[:, m])
        min_d[medoids] = 0.0

        probs = min_d ** 2
        total = probs.sum()
        if total < 1e-10:
            # All remaining are equidistant — pick randomly
            remaining = [i for i in range(n) if i not in medoids]
            medoids.append(rng.choice(remaining))
        else:
            probs /= total
            medoids.append(int(rng.choices(range(n), weights=probs.tolist())[0]))

    return medoids


def _assign_with_capacity(
    D: np.ndarray,
    orders: List[Order],
    medoid_indices: List[int],
    capacity_per_cluster: List[int],
) -> Tuple[Dict[int, List[int]], List[int]]:
    """
    Assign each order to its nearest feasible cluster.

    Capacity check uses raw pallet fraction sum (not categorical ceiling) to
    avoid inflation from multiple categories blocking assignment. The actual
    categorical pallet count is enforced at vehicle-assignment time.
    """
    K = len(medoid_indices)
    assignments: Dict[int, List[int]] = {k: [] for k in range(K)}
    cluster_raw_frac: Dict[int, float] = {k: 0.0 for k in range(K)}
    unassigned: List[int] = []

    # Largest raw fractions first — fills clusters efficiently
    order_indices = sorted(range(len(orders)), key=lambda i: -orders[i].est_pallet_fraction)

    for oi in order_indices:
        o = orders[oi]
        dists = [(D[oi][medoid_indices[k]], k) for k in range(K)]
        dists.sort()

        placed = False
        for _, k in dists:
            # Use raw fraction sum as capacity proxy — avoids categorical ceiling inflation
            new_frac = cluster_raw_frac[k] + o.est_pallet_fraction
            if new_frac <= capacity_per_cluster[k]:
                assignments[k].append(oi)
                cluster_raw_frac[k] = new_frac
                placed = True
                break

        if not placed:
            unassigned.append(oi)

    return assignments, unassigned


def _update_medoids(
    D: np.ndarray,
    assignments: Dict[int, List[int]],
) -> List[int]:
    """For each cluster, choose the order that minimises sum of distances to all others."""
    new_medoids: List[int] = []
    for k, members in assignments.items():
        if not members:
            new_medoids.append(-1)
            continue
        if len(members) == 1:
            new_medoids.append(members[0])
            continue
        best_cost = np.inf
        best_m = members[0]
        for candidate in members:
            cost = sum(D[candidate][j] for j in members if j != candidate)
            if cost < best_cost:
                best_cost = cost
                best_m = candidate
        new_medoids.append(best_m)
    return new_medoids


# ── Repair ────────────────────────────────────────────────────────────────────

def _repair_unassigned(
    unassigned_idx: List[int],
    orders: List[Order],
    assignments: Dict[int, List[int]],
    D: np.ndarray,
    capacity_per_cluster: List[int],
    medoid_indices: List[int],
) -> Tuple[Dict[int, List[int]], List[int]]:
    """Greedily insert unassigned orders using raw fraction capacity check."""
    still_unassigned: List[int] = []

    # Pre-compute raw fractions per cluster
    cluster_raw: Dict[int, float] = {
        k: sum(orders[j].est_pallet_fraction for j in members)
        for k, members in assignments.items()
    }

    for oi in unassigned_idx:
        o = orders[oi]
        best_k = None
        best_delta = np.inf

        for k, members in assignments.items():
            if cluster_raw[k] + o.est_pallet_fraction > capacity_per_cluster[k]:
                continue
            mid = medoid_indices[k]
            delta = D[oi][mid] if mid >= 0 else 0.0
            if delta < best_delta:
                best_delta = delta
                best_k = k

        if best_k is not None:
            assignments[best_k].append(oi)
            cluster_raw[best_k] += o.est_pallet_fraction
        else:
            still_unassigned.append(oi)

    return assignments, still_unassigned


# ── Main entry ────────────────────────────────────────────────────────────────

def cluster_orders(
    orders: List[Order],
    vehicles: List[Vehicle],
    D: np.ndarray,
    order_pos: Dict[str, int],
    depot_d: np.ndarray | None = None,
    max_iter: int | None = None,
    seed: int = 42,
) -> ClusterSolution:
    """
    Cluster orders using constrained K-Medoids (PAM-style).

    Parameters
    ----------
    orders     : list of Order objects for the day
    vehicles   : available vehicles (fleet)
    D          : NxN logistics distance matrix (from clustering.distance)
    order_pos  : dict mapping order.id → index in D (and orders list)
    depot_d    : 1D array of distances from depot to each order (for seeding)
    max_iter   : max K-Medoids iterations (default: config.CLUSTERING_MAX_ITER)
    seed       : random seed for reproducibility

    Returns
    -------
    ClusterSolution with clusters assigned to vehicles
    """
    if max_iter is None:
        max_iter = config.CLUSTERING_MAX_ITER

    rng = random.Random(seed)
    n = len(orders)
    n_vehicles = len(vehicles)

    if n == 0:
        return ClusterSolution(clusters=[], unassigned=[])

    if n_vehicles == 0:
        return ClusterSolution(clusters=[], unassigned=list(orders))

    # Depot distances for seeding
    if depot_d is None:
        depot_d = np.ones(n, dtype=np.float64) * 1800.0

    # K = number of vehicles (one cluster per vehicle initially).
    # No capacity constraint during K-Medoids — we want geographically compact clusters.
    # Oversized clusters are split after convergence (see Phase 3b below).
    max_cap = max(v.capacity_pallets for v in vehicles)
    K = min(n_vehicles, n)

    # Unconstrained capacity during clustering (we'll split later)
    _BIG = 10 ** 9
    capacity_per_cluster = [_BIG] * K

    # ── Phase 1: KMedoids++ initialisation ───────────────────────────────────
    K_eff = K
    medoid_indices = _kmedoids_init(D, depot_d, K_eff, rng)

    # ── Phase 2: Iterate assignment + medoid update ───────────────────────────
    assignments: Dict[int, List[int]] = {k: [] for k in range(K_eff)}
    unassigned_idx: List[int] = []

    for iteration in range(max_iter):
        old_medoids = list(medoid_indices)

        assignments, unassigned_idx = _assign_with_capacity(
            D, orders, medoid_indices, capacity_per_cluster
        )
        new_medoids = _update_medoids(D, assignments)

        # Replace -1 medoids (empty clusters) with old medoids
        for k in range(K_eff):
            if new_medoids[k] < 0:
                new_medoids[k] = old_medoids[k]

        medoid_indices = new_medoids

        if medoid_indices == old_medoids:
            break

    # ── Phase 3: Repair unassigned (shouldn't happen with unconstrained cap) ──
    assignments, unassigned_idx = _repair_unassigned(
        unassigned_idx, orders, assignments, D, capacity_per_cluster, medoid_indices
    )

    # Force-assign any remaining unassigned orders
    if unassigned_idx:
        overflow_batch: List[int] = []
        next_cluster_id = max(assignments.keys()) + 1 if assignments else K

        for oi in unassigned_idx:
            overflow_batch.append(oi)
            batch_orders = [orders[j] for j in overflow_batch]
            if estimate_pallets_byref(batch_orders) > max_cap:
                if len(overflow_batch) > 1:
                    assignments[next_cluster_id] = overflow_batch[:-1]
                    medoid_indices.append(overflow_batch[0])
                    next_cluster_id += 1
                    overflow_batch = [overflow_batch[-1]]
                else:
                    assignments[next_cluster_id] = [oi]
                    medoid_indices.append(oi)
                    next_cluster_id += 1
                    overflow_batch = []

        if overflow_batch:
            assignments[next_cluster_id] = overflow_batch
            medoid_indices.append(overflow_batch[0])

        unassigned_idx = []

    # ── Phase 4: Build Cluster objects ────────────────────────────────────────
    clusters: List[Cluster] = []
    for k, members in assignments.items():
        if not members:
            continue
        cluster_orders_list = [orders[i] for i in members]
        mid_global = medoid_indices[k]
        mid_id = orders[mid_global].id if 0 <= mid_global < n else cluster_orders_list[0].id

        c = Cluster(
            id=f"C{k:03d}",
            orders=cluster_orders_list,
            medoid_order_id=mid_id,
        )
        c.update_stats()
        # Intra cost
        mid_pos = order_pos.get(mid_id, -1)
        if mid_pos >= 0:
            c.intra_dist = float(
                sum(D[order_pos[o.id]][mid_pos] for o in cluster_orders_list if o.id in order_pos)
            )
        clusters.append(c)

    unassigned_orders = [orders[i] for i in unassigned_idx]

    # ── Phase 4b: Split clusters exceeding max vehicle capacity ──────────────
    final_clusters: List[Cluster] = []
    sid = 1000  # sub-cluster counter

    for c in clusters:
        if c.n_pallets <= max_cap:
            final_clusters.append(c)
            continue

        # Greedy split: accumulate orders until adding the next one would exceed max_cap
        batch: List[Order] = []
        for o in c.orders:
            tentative = batch + [o]
            if batch and estimate_pallets_byref(tentative) > max_cap:
                # Flush current batch
                sub = Cluster(id=f"S{sid:03d}", orders=batch[:], medoid_order_id=batch[0].id)
                sub.update_stats()
                final_clusters.append(sub)
                sid += 1
                batch = [o]
            else:
                batch.append(o)

        if batch:
            sub = Cluster(id=f"S{sid:03d}", orders=batch[:], medoid_order_id=batch[0].id)
            sub.update_stats()
            final_clusters.append(sub)
            sid += 1

    clusters = final_clusters

    # ── Phase 5: Assign vehicles to clusters ──────────────────────────────────
    _assign_vehicles_to_clusters(clusters, vehicles)

    return ClusterSolution(clusters=clusters, unassigned=unassigned_orders)
