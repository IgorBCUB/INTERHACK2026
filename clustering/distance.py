"""
Logistics distance matrix — combines Google Maps time, fuel and logistic penalties.

D[i,j] = α×T[i,j] + β×F[i,j]×FUEL_TO_S + γ×dist_equiv[i,j] + δ×pen(i,j)

Where:
  T[i,j]   = travel time in seconds (from cached time_matrix)
  F[i,j]   = fuel in litres (from cached fuel_matrix)
  dist_equiv = time_s / 3600 × 35 km/h × 60  (minutes equivalent)
  pen(i,j) = logistic incompatibility penalty (fragile+heavy, TW gap, returnables)
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np

from vrp.models import Order
import config


# 1 litre diesel × 1.55 €/L ÷ 18 €/h × 3600 s/h ≈ 310 s of driver time
FUEL_TO_SECONDS = 310.0


def build_logistics_matrix(
    time_matrix: List[List[float]],
    fuel_matrix: List[List[float]],
    orders: List[Order],
    matrix_idx: Dict[str, int],
    alpha: float = None,
    beta: float = None,
    gamma: float = None,
    delta: float = None,
) -> np.ndarray:
    """
    Build an NxN logistics distance matrix over the day's orders.

    matrix_idx maps client_id → index in the global cached matrices.
    Returns float64 ndarray, shape (N, N), diagonal = 0.
    """
    alpha = alpha if alpha is not None else config.CLUSTERING_ALPHA
    beta  = beta  if beta  is not None else config.CLUSTERING_BETA
    gamma = gamma if gamma is not None else config.CLUSTERING_GAMMA
    delta = delta if delta is not None else config.CLUSTERING_DELTA

    n = len(orders)
    D = np.zeros((n, n), dtype=np.float64)

    for i, oi in enumerate(orders):
        mi = matrix_idx.get(oi.client_id, -1)
        for j, oj in enumerate(orders):
            if i == j:
                continue
            mj = matrix_idx.get(oj.client_id, -1)
            if mi < 0 or mj < 0:
                # Fallback: haversine-based estimate
                D[i][j] = _haversine_seconds(oi, oj)
                continue

            t = float(time_matrix[mi][mj])
            f = float(fuel_matrix[mi][mj]) * FUEL_TO_SECONDS
            # distance proxy: time × 35 km/h → km → minutes equivalent
            d_min = (t / 3600.0) * 35.0 * 60.0
            pen = _logistics_penalty(oi, oj)

            D[i][j] = alpha * t + beta * f + gamma * d_min + delta * pen

    return D


def build_depot_distances(
    time_matrix: List[List[float]],
    fuel_matrix: List[List[float]],
    orders: List[Order],
    matrix_idx: Dict[str, int],
    depot_idx: int = 0,
    alpha: float = None,
    beta: float = None,
    gamma: float = None,
) -> np.ndarray:
    """
    Returns 1D array of logistics distances from depot to each order.
    Used for KMedoids++ initialisation.
    """
    alpha = alpha if alpha is not None else config.CLUSTERING_ALPHA
    beta  = beta  if beta  is not None else config.CLUSTERING_BETA
    gamma = gamma if gamma is not None else config.CLUSTERING_GAMMA

    n = len(orders)
    depot_d = np.zeros(n, dtype=np.float64)

    for i, o in enumerate(orders):
        mi = matrix_idx.get(o.client_id, -1)
        if mi < 0:
            depot_d[i] = 3600.0  # unknown → assume 1h
            continue
        t = float(time_matrix[depot_idx][mi])
        f = float(fuel_matrix[depot_idx][mi]) * FUEL_TO_SECONDS
        d_min = (t / 3600.0) * 35.0 * 60.0
        depot_d[i] = alpha * t + beta * f + gamma * d_min

    return depot_d


def _logistics_penalty(oi: Order, oj: Order) -> float:
    """
    Logistic incompatibility penalty between two orders (seconds equivalent).

    Components:
    - Time-window gap: if their windows don't overlap, penalise the gap
    - Fragile + heavy mix: risk of damage if loaded together
    - Returnables: reduce effective capacity
    """
    pen = 0.0

    # Time-window gap: max(0, how much they DON'T overlap)
    tw_gap = max(0, oi.tw_open - oj.tw_close, oj.tw_open - oi.tw_close)
    pen += tw_gap * 0.5

    # Fragile × heavy incompatibility (stacking risk)
    if oi.is_fragile and oj.is_heavy:
        pen += 1800.0
    if oj.is_fragile and oi.is_heavy:
        pen += 1800.0

    # Returnables reduce effective pallet capacity
    if oi.has_returnables or oj.has_returnables:
        pen += 300.0

    return pen


def _haversine_seconds(oi: Order, oj: Order, speed_kmh: float = 35.0) -> float:
    """Fallback distance estimate when matrix entry is missing."""
    import math
    R = 6371.0
    lat1, lon1 = math.radians(oi.lat), math.radians(oi.lon)
    lat2, lon2 = math.radians(oj.lat), math.radians(oj.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    dist_km = 2 * R * math.asin(math.sqrt(a))
    return (dist_km / speed_kmh) * 3600.0
