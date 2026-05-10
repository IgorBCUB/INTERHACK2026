"""
Main VRP solver pipeline:
  1. Load data
  2. Geocode orders
  3. Build cost matrix (Google Maps or haversine)
  4. Clarke-Wright initial solution
  5. Simulated Annealing optimisation
  6. Return final Solution
"""
from __future__ import annotations
from typing import List, Tuple

from vrp.models import Order, Vehicle, Solution
from vrp.cost import evaluate
from vrp.savings import clarke_wright
from vrp.annealing import simulated_annealing
import config


def solve(
    orders: List[Order],
    vehicles: List[Vehicle],
    matrix: List[List[float]],
    verbose: bool = True,
) -> Solution:
    """Full VRP solve: CW heuristic → SA optimisation."""

    if verbose:
        print(f"\n[Solver] {len(orders)} orders, {len(vehicles)} vehicles")
        print("[Solver] Building initial solution (Clarke-Wright)...")

    initial = clarke_wright(orders, vehicles, matrix)
    evaluate(initial, orders, vehicles, matrix)

    if verbose:
        print(f"[Solver] Initial cost: {initial.cost:,.0f}s  "
              f"trips={initial.num_trips()}  reloads={initial.num_reloads()}  "
              f"unserved={len(initial.unserved_orders)}")
        print("[Solver] Optimising with Simulated Annealing...")

    optimised = simulated_annealing(initial, orders, vehicles, matrix, verbose=verbose)

    if verbose:
        print(f"[Solver] Final cost:   {optimised.cost:,.0f}s  "
              f"trips={optimised.num_trips()}  reloads={optimised.num_reloads()}  "
              f"unserved={len(optimised.unserved_orders)}")

    return optimised
