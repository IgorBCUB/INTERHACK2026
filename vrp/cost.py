"""
Cost function for the VRP — extendido con acoplado parcial al almacén.

Total cost = Σ_trips [ travel + service × unload_multiplier + TW penalties + reload + capacity ]
           + WH_COUPLING_WEIGHT × Σ_trips trip_warehouse_estimate(trip, vehicle)
           + BIG_M × unserved_orders
"""
from __future__ import annotations
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

        # Ventana horaria
        arrival = current_time
        if arrival < stop.tw_open:
            wait = stop.tw_open - arrival
            current_time += wait          # wait for window (free — driver waits)
        elif arrival > stop.tw_close:
            overdue = arrival - stop.tw_close
            # priority=1 (hospital/clinic) → 3× penalty; priority=3 (normal) → 1×
            priority_factor = 4 - stop.priority   # 1→3, 2→2, 3→1
            cost += overdue * priority_factor * config.TW_VIOLATION_MULTIPLIER

        # Service time escalado por estrategia (ya es proporcional al tamaño del pedido)
        scaled_service = stop.service_time * unload_mult
        current_time += scaled_service
        cost += scaled_service
        prev = idx

    # Vuelta al depot
    cost += matrix[prev][0]
    current_time += matrix[prev][0]

    # Reload penalty
    if trip.is_reload:
        cost += config.DEPOT_RELOAD_PENALTY_SECONDS

    # Capacidad en cajas (legado)
    total_load = sum(s.volume_boxes for s in trip.stops)
    if total_load > vehicle.capacity_boxes:
        cost += (total_load - vehicle.capacity_boxes) * 500
    trip.total_load_boxes = total_load

    # Route length overrun
    if current_time > vehicle.max_route_seconds:
        cost += (current_time - vehicle.max_route_seconds) * 2

    # ── 3) Suma del término almacén con su peso ───────────────────────────────
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

    for trip in solution.trips:
        v = vehicle_map.get(trip.vehicle_id)
        if v is None:
            continue
        c = trip_cost(trip, orders, matrix, v)
        total += c
        if trip.is_reload:
            reload_penalty += config.DEPOT_RELOAD_PENALTY_SECONDS

    total += len(solution.unserved_orders) * BIG_M

    solution.total_time_seconds = total
    solution.total_reload_penalty = reload_penalty
    solution.cost = total
    return solution
