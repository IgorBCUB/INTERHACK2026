"""
TSP heuristics for ordering stops within a cluster.

nearest_neighbor_tsp  — greedy O(n²), good enough for clusters ≤ 20 stops
two_opt_improve       — local 2-opt improvement
"""
from __future__ import annotations
from typing import Dict, List

from vrp.models import Order


def nearest_neighbor_tsp(
    orders: List[Order],
    time_matrix: List[List[float]],
    matrix_idx: Dict[str, int],
    depot_idx: int = 0,
) -> List[Order]:
    """
    Greedy nearest-neighbour TSP starting from depot.
    Returns orders in visit order.
    """
    if not orders:
        return []
    if len(orders) == 1:
        return list(orders)

    unvisited = list(orders)
    route: List[Order] = []
    current = depot_idx

    while unvisited:
        best_t = float("inf")
        best_o = None
        for o in unvisited:
            mi = matrix_idx.get(o.client_id, -1)
            t = time_matrix[current][mi] if mi >= 0 else float("inf")
            if t < best_t:
                best_t = t
                best_o = o
        route.append(best_o)
        unvisited.remove(best_o)
        current = matrix_idx.get(best_o.client_id, depot_idx)

    return route


def two_opt_improve(
    route: List[Order],
    time_matrix: List[List[float]],
    matrix_idx: Dict[str, int],
    depot_idx: int = 0,
    max_passes: int = 5,
) -> List[Order]:
    """
    2-opt local search on a fixed set of stops.
    Returns improved route.
    """
    if len(route) <= 2:
        return route

    def route_time(r: List[Order]) -> float:
        t = time_matrix[depot_idx][matrix_idx.get(r[0].client_id, depot_idx)]
        for k in range(len(r) - 1):
            a = matrix_idx.get(r[k].client_id, depot_idx)
            b = matrix_idx.get(r[k + 1].client_id, depot_idx)
            t += time_matrix[a][b]
        t += time_matrix[matrix_idx.get(r[-1].client_id, depot_idx)][depot_idx]
        return t

    best = list(route)
    best_t = route_time(best)

    for _ in range(max_passes):
        improved = False
        n = len(best)
        for i in range(n - 1):
            for j in range(i + 2, n):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                ct = route_time(candidate)
                if ct < best_t - 1.0:
                    best = candidate
                    best_t = ct
                    improved = True
        if not improved:
            break

    return best
