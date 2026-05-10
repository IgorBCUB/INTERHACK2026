"""
Simulated Annealing optimizer for the VRP solution.

Neighbourhood moves:
  - 2-opt:     reverse a sub-sequence within one trip
  - relocate:  move one stop from one trip to another position
  - swap:      exchange one stop between two different trips
  - or-opt:    move a chain of 2-3 consecutive stops to another trip
"""
from __future__ import annotations
import copy
import math
import random
from typing import List, Dict

from vrp.models import Order, Vehicle, Trip, Solution
from vrp.cost import evaluate, trip_cost
import config


def _random_trip_with_stops(trips: List[Trip]) -> Trip | None:
    candidates = [t for t in trips if len(t.stops) > 0]
    return random.choice(candidates) if candidates else None


def _two_opt_move(trip: Trip, orders: List[Order], matrix: List[List[float]]) -> Trip | None:
    """Reverse a random sub-sequence within the trip."""
    if len(trip.stops) < 3:
        return None
    i = random.randint(0, len(trip.stops) - 2)
    j = random.randint(i + 1, len(trip.stops) - 1)
    new_stops = trip.stops[:i] + list(reversed(trip.stops[i:j + 1])) + trip.stops[j + 1:]
    new_trip = copy.copy(trip)
    new_trip.stops = new_stops
    return new_trip


def _relocate_move(
    source: Trip, target: Trip, orders: List[Order], matrix: List[List[float]]
) -> tuple[Trip, Trip] | None:
    """Move a random stop from source to the best position in target."""
    if not source.stops:
        return None
    idx = random.randint(0, len(source.stops) - 1)
    stop = source.stops[idx]

    # Check capacity of target
    target_load = sum(o.volume_boxes for o in target.stops) + stop.volume_boxes
    # We skip capacity check here and rely on cost penalisation for overloads.
    # For a hackathon this is acceptable.

    new_source = copy.copy(source)
    new_source.stops = source.stops[:idx] + source.stops[idx + 1:]

    best_pos = 0
    best_cost = float("inf")
    order_idx = {o.id: i + 1 for i, o in enumerate(orders)}

    for pos in range(len(target.stops) + 1):
        candidate_stops = target.stops[:pos] + [stop] + target.stops[pos:]
        c = _approx_trip_time(candidate_stops, matrix, order_idx)
        if c < best_cost:
            best_cost = c
            best_pos = pos

    new_target = copy.copy(target)
    new_target.stops = target.stops[:best_pos] + [stop] + target.stops[best_pos:]
    return new_source, new_target


def _swap_move(
    t1: Trip, t2: Trip
) -> tuple[Trip, Trip] | None:
    """Swap a random stop between two trips."""
    if not t1.stops or not t2.stops:
        return None
    i1 = random.randint(0, len(t1.stops) - 1)
    i2 = random.randint(0, len(t2.stops) - 1)
    s1 = t1.stops[i1]
    s2 = t2.stops[i2]

    new_t1 = copy.copy(t1)
    new_t1.stops = t1.stops[:i1] + [s2] + t1.stops[i1 + 1:]

    new_t2 = copy.copy(t2)
    new_t2.stops = t2.stops[:i2] + [s1] + t2.stops[i2 + 1:]

    return new_t1, new_t2


def _approx_trip_time(stops: List[Order], matrix: List[List[float]], order_idx: Dict[str, int]) -> float:
    if not stops:
        return 0.0
    t = 0.0
    prev = 0
    for s in stops:
        idx = order_idx.get(s.id, 0)
        t += matrix[prev][idx] + s.service_time
        prev = idx
    t += matrix[prev][0]
    return t


# ── Main SA loop ───────────────────────────────────────────────────────────────

def simulated_annealing(
    initial: Solution,
    orders: List[Order],
    vehicles: List[Vehicle],
    matrix: List[List[float]],
    initial_temp: float = config.SA_INITIAL_TEMP,
    cooling_rate: float = config.SA_COOLING_RATE,
    min_temp: float = config.SA_MIN_TEMP,
    max_iter_per_temp: int = config.SA_MAX_ITER_PER_TEMP,
    verbose: bool = True,
) -> Solution:

    current = evaluate(copy.deepcopy(initial), orders, vehicles, matrix)
    best = copy.deepcopy(current)

    temp = initial_temp
    iteration = 0

    order_idx = {o.id: i + 1 for i, o in enumerate(orders)}

    while temp > min_temp:
        for _ in range(max_iter_per_temp):
            iteration += 1
            candidate = copy.deepcopy(current)
            trips = candidate.trips

            if not trips:
                break

            move = random.choice(["2opt", "relocate", "swap"])

            if move == "2opt":
                t = _random_trip_with_stops(trips)
                if t is None:
                    continue
                new_t = _two_opt_move(t, orders, matrix)
                if new_t is None:
                    continue
                idx = next(i for i, x in enumerate(trips) if x is t)
                trips[idx] = new_t

            elif move == "relocate":
                if len(trips) < 2:
                    continue
                src, tgt = random.sample(trips, 2)
                result = _relocate_move(src, tgt, orders, matrix)
                if result is None:
                    continue
                new_src, new_tgt = result
                for i, t in enumerate(trips):
                    if t is src:
                        trips[i] = new_src
                    elif t is tgt:
                        trips[i] = new_tgt

            elif move == "swap":
                if len(trips) < 2:
                    continue
                t1, t2 = random.sample(trips, 2)
                result = _swap_move(t1, t2)
                if result is None:
                    continue
                new_t1, new_t2 = result
                for i, t in enumerate(trips):
                    if t is t1:
                        trips[i] = new_t1
                    elif t is t2:
                        trips[i] = new_t2

            # Remove empty trips
            candidate.trips = [t for t in trips if t.stops]

            evaluate(candidate, orders, vehicles, matrix)

            delta = candidate.cost - current.cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current = candidate
                if current.cost < best.cost:
                    best = copy.deepcopy(current)

        temp *= cooling_rate
        if verbose and iteration % 5000 == 0:
            print(f"  SA iter={iteration:7d}  T={temp:8.1f}  best={best.cost:,.0f}s  current={current.cost:,.0f}s")

    return best
