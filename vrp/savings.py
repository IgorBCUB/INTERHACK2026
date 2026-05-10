"""
Clarke-Wright Savings heuristic for initial VRP solution — vehicle-aware.

For each pair of orders (i, j):
    saving(i, j) = d(depot, i) + d(depot, j) - d(i, j)

Merge routes greedily from highest saving, respetando capacity_pallets de la flota.
Asigna vehículos según tamaño y prioridad (van_3 → rutas pequeñas con prioridad sin frágiles).
"""
from __future__ import annotations
from collections import defaultdict
import math
from typing import List, Dict, Tuple, Set
import copy

from vrp.models import Order, Vehicle, Trip, Solution
import config


# ── Helpers ────────────────────────────────────────────────────────────────────

def _trip_duration(stops: List[Order], matrix: List[List[float]], order_idx: Dict[str, int]) -> float:
    """Estimate total route duration (travel + parking + unload)."""
    if not stops:
        return 0.0
    dur = 0.0
    prev = 0
    for s in stops:
        idx = order_idx[s.id]
        dur += matrix[prev][idx] + s.parking_time + s.service_time
        prev = idx
    dur += matrix[prev][0]
    return dur


def estimate_pallets_byref(stops: List[Order]) -> int:
    """
    Estima palets bajo BY_REFERENCE (agregación por categoría).
    Es la estrategia más eficiente, usada como umbral de capacidad en CW.
    """
    if not stops:
        return 0
    frac_per_cat: Dict[str, float] = defaultdict(float)
    n_box_orders = 0
    for s in stops:
        if not s.needs_pallet:
            n_box_orders += 1
            continue
        if not s.categories:
            frac_per_cat["__nocat__"] += s.est_pallet_fraction
            continue
        share = s.est_pallet_fraction / len(s.categories)
        for cat in s.categories:
            frac_per_cat[cat] += share
    n_pallets = sum(math.ceil(f) for f in frac_per_cat.values())
    if n_box_orders > 0:
        n_pallets += 1   # mini-palet "varios"
    return n_pallets


# ── Clarke-Wright ─────────────────────────────────────────────────────────────

def clarke_wright(
    orders: List[Order],
    vehicles: List[Vehicle],
    matrix: List[List[float]],
) -> Solution:
    """Build an initial feasible solution using Clarke-Wright Savings (vehicle-aware)."""
    order_idx: Dict[str, int] = {o.id: i + 1 for i, o in enumerate(orders)}

    # ── 1. Compute savings ────────────────────────────────────────────────────
    savings: List[Tuple[float, Order, Order]] = []
    for i, oi in enumerate(orders):
        for j, oj in enumerate(orders):
            if j <= i:
                continue
            si = order_idx[oi.id]
            sj = order_idx[oj.id]
            saving = matrix[0][si] + matrix[0][sj] - matrix[si][sj]
            savings.append((saving, oi, oj))
    savings.sort(key=lambda x: -x[0])

    # ── 2. Initialise: each order is its own route ────────────────────────────
    route_of: Dict[str, int] = {o.id: i for i, o in enumerate(orders)}
    routes: Dict[int, List[Order]] = {i: [o] for i, o in enumerate(orders)}

    # Capacidad máxima de palets en la flota (umbral para CW; la asignación a
    # vehículo concreto se hace después y es donde se respeta la capacidad real)
    max_capacity_pallets = max((v.capacity_pallets for v in vehicles), default=8)

    def can_merge(ra: List[Order], rb: List[Order]) -> bool:
        merged = ra + rb
        # Limit por palets (modo by_reference — el más eficiente)
        if estimate_pallets_byref(merged) > max_capacity_pallets:
            return False
        # Limit por tiempo de ruta
        if _trip_duration(merged, matrix, order_idx) > config.DEFAULT_MAX_ROUTE_SECONDS:
            return False
        return True

    # ── 3. Greedily merge ─────────────────────────────────────────────────────
    for _, oi, oj in savings:
        ri = route_of[oi.id]
        rj = route_of[oj.id]
        if ri == rj:
            continue

        ra = routes[ri]
        rb = routes[rj]

        # oi must be at the END of ra, oj at the START of rb (or reversed)
        edge_ok = (ra[-1].id == oi.id and rb[0].id == oj.id) or \
                  (ra[-1].id == oj.id and rb[0].id == oi.id) or \
                  (ra[0].id == oi.id and rb[-1].id == oj.id) or \
                  (ra[0].id == oj.id and rb[-1].id == oi.id)
        if not edge_ok:
            continue

        if not can_merge(ra, rb):
            continue

        # Merge rb into ra
        if ra[-1].id == oi.id and rb[0].id == oj.id:
            new_route = ra + rb
        elif ra[-1].id == oj.id and rb[0].id == oi.id:
            new_route = ra + rb
        elif ra[0].id == oi.id and rb[-1].id == oj.id:
            new_route = rb + ra
        else:
            new_route = rb + ra

        routes[ri] = new_route
        del routes[rj]
        for o in rb:
            route_of[o.id] = ri

    # ── 4. Assign routes to vehicles (vehicle-aware, multiple trips OK) ───────
    trips: List[Trip] = []
    unserved: List[Order] = []
    vehicle_trip_counts: Dict[str, int] = {v.id: 0 for v in vehicles}
    vehicle_time_used: Dict[str, float] = {v.id: 0.0 for v in vehicles}

    # Ordenar rutas: las más grandes (más palets) primero → mejor asignación
    route_list = sorted(routes.values(), key=lambda r: -estimate_pallets_byref(r))

    for route_stops in route_list:
        n_palets = estimate_pallets_byref(route_stops)
        has_priority = any(o.is_priority for o in route_stops)
        has_fragile  = any(o.is_fragile  for o in route_stops)
        load_boxes = sum(o.volume_boxes for o in route_stops)
        dur = _trip_duration(route_stops, matrix, order_idx)

        candidate = _pick_vehicle(
            vehicles, n_palets, has_priority, has_fragile,
            vehicle_time_used, dur,
        )

        if candidate is not None:
            is_reload = vehicle_trip_counts[candidate.id] > 0
            trips.append(Trip(
                vehicle_id=candidate.id,
                stops=route_stops,
                total_time_seconds=dur,
                total_load_boxes=load_boxes,
                is_reload=is_reload,
            ))
            vehicle_time_used[candidate.id] += dur + (
                config.DEPOT_RELOAD_PENALTY_SECONDS if is_reload else 0
            )
            vehicle_trip_counts[candidate.id] += 1
            continue

        # No cabe en ningún vehículo: dividir
        sub_trips = _split_into_sub_trips(
            route_stops, vehicles, matrix, order_idx,
            vehicle_time_used, vehicle_trip_counts,
        )
        if sub_trips:
            trips.extend(sub_trips)
        else:
            unserved.extend(route_stops)

    return Solution(trips=trips, unserved_orders=unserved)


def _pick_vehicle(
    vehicles: List[Vehicle],
    n_palets: int,
    has_priority: bool,
    has_fragile: bool,
    vehicle_time_used: Dict[str, float],
    dur: float,
) -> Vehicle | None:
    """
    Selecciona el mejor vehículo para una ruta:
    - van_3 → solo rutas pequeñas (≤2 palets) con prioridad y SIN frágiles
    - truck_8 → rutas grandes (>6 palets)
    - truck_6 → resto
    Respeta capacity_pallets como restricción dura.
    """
    # Furgoneta: solo si encaja perfectamente
    if n_palets <= 2 and has_priority and not has_fragile:
        for v in vehicles:
            if v.vehicle_type == "van_3":
                if (vehicle_time_used[v.id] + dur <= v.max_daily_seconds
                        and n_palets <= v.capacity_pallets):
                    return v

    # truck_8 para rutas grandes
    if n_palets > 6:
        for v in sorted(vehicles, key=lambda x: vehicle_time_used[x.id]):
            if v.vehicle_type == "truck_8":
                if (vehicle_time_used[v.id] + dur <= v.max_daily_seconds
                        and n_palets <= v.capacity_pallets):
                    return v
        # Si no caben en ningún truck_8 disponible, probar el siguiente más grande
        for v in sorted(vehicles, key=lambda x: -x.capacity_pallets):
            if (vehicle_time_used[v.id] + dur <= v.max_daily_seconds
                    and n_palets <= v.capacity_pallets):
                return v
        return None

    # truck_6 (default), por orden de menor uso
    for v in sorted(vehicles, key=lambda x: vehicle_time_used[x.id]):
        if v.vehicle_type == "truck_6":
            if (vehicle_time_used[v.id] + dur <= v.max_daily_seconds
                    and n_palets <= v.capacity_pallets):
                return v

    # Fallback: cualquier vehículo donde quepa
    for v in sorted(vehicles, key=lambda x: -x.capacity_pallets):
        if (vehicle_time_used[v.id] + dur <= v.max_daily_seconds
                and n_palets <= v.capacity_pallets):
            return v
    return None


def _split_into_sub_trips(
    stops: List[Order],
    vehicles: List[Vehicle],
    matrix: List[List[float]],
    order_idx: Dict[str, int],
    vehicle_time_used: Dict[str, float],
    vehicle_trip_counts: Dict[str, int],
) -> List[Trip]:
    """Split an oversize route into capacity-feasible sub-trips."""
    trips: List[Trip] = []

    # Vehículo con más capacidad de palets y menor uso
    best_v = max(vehicles, key=lambda v: (v.capacity_pallets, -vehicle_time_used[v.id]))

    current_stops: List[Order] = []
    current_palets = 0

    for stop in stops:
        # Estimación rápida añadiendo este stop (usa BY_REFERENCE)
        tentative = current_stops + [stop]
        tentative_palets = estimate_pallets_byref(tentative)
        if tentative_palets > best_v.capacity_pallets and current_stops:
            dur = _trip_duration(current_stops, matrix, order_idx)
            is_reload = vehicle_trip_counts[best_v.id] > 0
            trips.append(Trip(
                vehicle_id=best_v.id,
                stops=current_stops[:],
                total_time_seconds=dur,
                total_load_boxes=sum(s.volume_boxes for s in current_stops),
                is_reload=is_reload,
            ))
            vehicle_time_used[best_v.id] += dur + (
                config.DEPOT_RELOAD_PENALTY_SECONDS if is_reload else 0
            )
            vehicle_trip_counts[best_v.id] += 1
            current_stops = [stop]
        else:
            current_stops.append(stop)

    if current_stops:
        dur = _trip_duration(current_stops, matrix, order_idx)
        is_reload = vehicle_trip_counts[best_v.id] > 0
        trips.append(Trip(
            vehicle_id=best_v.id,
            stops=current_stops,
            total_time_seconds=dur,
            total_load_boxes=sum(s.volume_boxes for s in current_stops),
            is_reload=is_reload,
        ))
        vehicle_time_used[best_v.id] += dur + (
            config.DEPOT_RELOAD_PENALTY_SECONDS if is_reload else 0
        )
        vehicle_trip_counts[best_v.id] += 1

    return trips
