"""
Cost function for the VRP — extendido con acoplado parcial al almacén.

Total cost = Σ_trips [ travel + parking + service×unload_mult + TW penalties + reload ]
           + WH_COUPLING_WEIGHT × Σ_trips trip_warehouse_estimate(trip, vehicle)
           + daily_overrun_penalty per vehicle
           + BIG_M × unserved_orders

Time accumulation per trip (feeds daily cap):
  travel + wait_for_TW + parking + service + return_to_depot + reload_time
"""
from __future__ import annotations
from collections import defaultdict
from typing import List

from vrp.models import Order, Vehicle, Trip, Solution
import config

BIG_M = 1_000_000     # seconds equivalent for unserved order


def trip_cost(
    trip: Trip,
    orders: List[Order],
    matrix: List[List[float]],
    vehicle: Vehicle,
) -> float:
    """Compute the cost of a single trip (route + warehouse partial coupling)."""
    if not trip.stops:
        return 0.0

    # ── 1) Estimador almacén (cachea unload_multiplier y chosen_strategy) ─────
    wh_total = 0.0
    if config.WH_COUPLING_ENABLED:
        from warehouse.fast_estimate import trip_warehouse_estimate
        wh_total = trip_warehouse_estimate(trip, vehicle)
        unload_mult = trip.unload_multiplier
    else:
        unload_mult = 1.0
        trip.chosen_strategy = "by_order"
        trip.unload_multiplier = 1.0

    # ── 2) Coste de ruta tradicional con service_time escalado ────────────────
    order_idx = {o.id: i + 1 for i, o in enumerate(orders)}

    cost = 0.0
    prev = 0  # depot
    current_time = 0.0

    for stop in trip.stops:
        idx = order_idx.get(stop.id, 0)
        travel = matrix[prev][idx]
        cost += travel
        current_time += travel

        # Parking search time (pre-computed per client, deterministic)
        current_time += stop.parking_time
        cost += stop.parking_time

        # Ventana horaria
        arrival = current_time
        if arrival < stop.tw_open:
            wait = stop.tw_open - arrival
            current_time += wait          # wait for window (driver waits, no extra cost)
        elif arrival > stop.tw_close:
            overdue = arrival - stop.tw_close
            priority_factor = 4 - stop.priority   # 1→3, 2→2, 3→1
            cost += overdue * priority_factor * config.TW_VIOLATION_MULTIPLIER

        # Unload time (proportional to order size, scaled by warehouse strategy)
        scaled_service = stop.service_time * unload_mult
        current_time += scaled_service
        cost += scaled_service
        prev = idx

    # ── Stacking / load-order penalty ────────────────────────────────────────
    # LIFO: stop at delivery position i=0 (first delivery) is loaded LAST → near door.
    # Fragile stops buried deep in the truck (high loading depth) are penalised.
    # Heavy stops near the door (loaded last on top of lighter goods) are also penalised.
    n_stops = len(trip.stops)
    if n_stops > 1:
        for i, s in enumerate(trip.stops):
            loading_depth = n_stops - 1 - i  # 0=door (good for fragile), n-1=deepest
            if s.fragility_score >= 2:
                # Fragile deep in truck: penalise proportionally to depth × fragility
                cost += loading_depth * s.fragility_score * config.STACKING_PEN_PER_DEPTH / n_stops
            if s.weight_score >= 3 and i < n_stops // 2:
                # Very heavy near door, potentially over lighter fragile goods deeper in
                cost += (n_stops // 2 - i) * config.STACKING_PEN_PER_DEPTH * 0.5 / n_stops
        # Van without side access: LIFO violations are much worse
        if not vehicle.has_side_access:
            for i, s in enumerate(trip.stops):
                loading_depth = n_stops - 1 - i
                if s.fragility_score >= 2:
                    cost += loading_depth * s.fragility_score * config.WH_PEN_FRAGILE_DEEP_VAN / n_stops

    # Return to depot
    cost += matrix[prev][0]
    current_time += matrix[prev][0]

    # Reload: penalty cost + real time (counts against daily cap)
    if trip.is_reload:
        cost += config.DEPOT_RELOAD_PENALTY_SECONDS
        current_time += config.DEPOT_RELOAD_PENALTY_SECONDS

    # Capacidad en cajas (legado)
    total_load = sum(s.volume_boxes for s in trip.stops)
    if total_load > vehicle.capacity_boxes:
        cost += (total_load - vehicle.capacity_boxes) * 500
    trip.total_load_boxes = total_load

    # Per-trip route overrun (soft penalty — daily cap enforced in evaluate())
    if current_time > vehicle.max_route_seconds:
        cost += (current_time - vehicle.max_route_seconds) * 1.5

    # Store real elapsed time so evaluate() can accumulate daily totals
    trip.total_time_seconds = current_time

    # ── 3) Warehouse coupling term ────────────────────────────────────────────
    cost += config.WH_COUPLING_WEIGHT * wh_total

    return cost


def evaluate(
    solution: Solution,
    orders: List[Order],
    vehicles: List[Vehicle],
    matrix: List[List[float]],
) -> Solution:
    """Fill solution.cost and solution.total_time_seconds in-place."""
    vehicle_map = {v.id: v for v in vehicles}
    total = 0.0
    reload_penalty = 0.0
    # Accumulate real elapsed time per vehicle (travel + parking + unload + reload)
    daily_time: dict[str, float] = defaultdict(float)

    for trip in solution.trips:
        v = vehicle_map.get(trip.vehicle_id)
        if v is None:
            continue
        c = trip_cost(trip, orders, matrix, v)
        total += c
        daily_time[trip.vehicle_id] += trip.total_time_seconds
        if trip.is_reload:
            reload_penalty += config.DEPOT_RELOAD_PENALTY_SECONDS

    # Daily working-day cap: hard constraint via prohibitive penalty
    # 1 min overrun ≈ cost of leaving a client unserved → SA never accepts violations
    for vid, day_t in daily_time.items():
        v = vehicle_map.get(vid)
        if v and day_t > v.max_daily_seconds:
            overrun = day_t - v.max_daily_seconds
            total += overrun * (BIG_M / 3600)   # 1 h overrun = BIG_M cost

    total += len(solution.unserved_orders) * BIG_M

    solution.total_time_seconds = sum(daily_time.values())
    solution.total_reload_penalty = reload_penalty
    solution.daily_time_per_vehicle = dict(daily_time)
    solution.cost = total
    return solution
