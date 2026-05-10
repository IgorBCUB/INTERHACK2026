"""
Cost functions for logistics clustering.

cluster_intra_cost   — compactness: sum of distances to medoid
cluster_route_cost   — TSP nearest-neighbor + warehouse coupling (reuses vrp/cost.py)
cluster_fuel_litres  — total fuel for a cluster's route
cluster_co2_kg       — CO2 equivalent
total_solution_cost  — complete cost (route + warehouse + fuel + CO2 + intra + unserved)
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np

from vrp.models import Order, Vehicle, Trip
from vrp.cost import trip_cost
from clustering.models import Cluster, ClusterSolution
from clustering.tsp import nearest_neighbor_tsp, two_opt_improve
import config


# ── Constants ──────────────────────────────────────────────────────────────────

FUEL_TO_SECONDS  = 310.0          # s equivalent per litre
CO2_KG_PER_LITRE = 2.68           # kg CO2 per litre diesel (DEFRA 2023)
BIG_M            = 1_000_000      # penalty for unserved order


def cluster_intra_cost(
    cluster: Cluster,
    D: np.ndarray,
    order_pos: Dict[str, int],
) -> float:
    """Sum of logistics distances from each order to the cluster medoid."""
    if len(cluster.orders) <= 1 or not cluster.medoid_order_id:
        return 0.0
    mid_pos = order_pos.get(cluster.medoid_order_id, -1)
    if mid_pos < 0:
        return 0.0
    return float(sum(
        D[order_pos[o.id]][mid_pos]
        for o in cluster.orders
        if o.id in order_pos and o.id != cluster.medoid_order_id
    ))


def cluster_fuel_litres(
    cluster: Cluster,
    fuel_matrix: List[List[float]],
    matrix_idx: Dict[str, int],
    stops_order: List[Order] | None = None,
    depot_idx: int = 0,
) -> float:
    """Total fuel (litres) for a cluster's route (depot → stops → depot)."""
    route = stops_order if stops_order is not None else cluster.orders
    if not route:
        return 0.0
    total = 0.0
    prev = depot_idx
    for o in route:
        mi = matrix_idx.get(o.client_id, -1)
        if mi < 0:
            continue
        total += fuel_matrix[prev][mi]
        prev = mi
    if prev != depot_idx:
        total += fuel_matrix[prev][depot_idx]
    return total


def cluster_route_cost(
    cluster: Cluster,
    time_matrix: List[List[float]],
    fuel_matrix: List[List[float]],
    matrix_idx: Dict[str, int],
    vehicle: Vehicle,
    use_two_opt: bool = True,
    depot_idx: int = 0,
) -> tuple[float, List[Order], float]:
    """
    Compute routing cost for a cluster using greedy TSP + warehouse coupling.

    Returns (cost_seconds, ordered_stops, fuel_litres).
    """
    if not cluster.orders or vehicle is None:
        return 0.0, [], 0.0

    stops = nearest_neighbor_tsp(cluster.orders, time_matrix, matrix_idx, depot_idx)
    if use_two_opt:
        stops = two_opt_improve(stops, time_matrix, matrix_idx, depot_idx)

    trip = Trip(vehicle_id=vehicle.id, stops=stops, is_reload=False)
    cost = trip_cost(trip, cluster.orders, time_matrix, vehicle)
    fuel = cluster_fuel_litres(cluster, fuel_matrix, matrix_idx, stops, depot_idx)

    return cost, stops, fuel


def total_solution_cost(
    solution: ClusterSolution,
    time_matrix: List[List[float]],
    fuel_matrix: List[List[float]],
    matrix_idx: Dict[str, int],
    D: np.ndarray,
    order_pos: Dict[str, int],
    intra_weight: float = 0.05,
    depot_idx: int = 0,
) -> ClusterSolution:
    """
    Compute full solution cost in-place. Populates solution.* fields.
    Returns the updated solution.
    """
    route_total = 0.0
    wh_total    = 0.0
    fuel_total  = 0.0
    co2_total   = 0.0
    intra_total = 0.0

    for cluster in solution.clusters:
        if cluster.vehicle is None:
            continue
        cost, _, fuel = cluster_route_cost(
            cluster, time_matrix, fuel_matrix, matrix_idx, cluster.vehicle,
            depot_idx=depot_idx,
        )
        route_total += cost
        wh_total    += getattr(cluster, "_trip_warehouse_cost", 0.0)
        fuel_total  += fuel * FUEL_TO_SECONDS
        co2_total   += fuel * CO2_KG_PER_LITRE * config.CO2_PENALTY_WEIGHT
        intra_total += cluster_intra_cost(cluster, D, order_pos)

    solution.route_cost     = route_total
    solution.fuel_cost_equiv = fuel_total
    solution.co2_cost_equiv  = co2_total
    solution.intra_cost      = intra_total * intra_weight
    solution.total_cost = (
        route_total
        + fuel_total
        + co2_total
        + intra_total * intra_weight
        + len(solution.unassigned) * BIG_M
    )
    return solution
