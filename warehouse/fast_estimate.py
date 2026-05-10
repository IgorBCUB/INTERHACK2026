"""
Fast warehouse estimator (single-pass, vehicle-aware, stack-aware).

Llamado desde vrp/cost.py en cada evaluación del SA.
Estructura:
  - Una sola pasada por trip.stops para recolectar señales agregadas.
  - Calcula coste bajo BY_ORDER, BY_REFERENCE y HYBRID en O(1) cada uno.
  - Elige el mínimo y cachea chosen_strategy + unload_multiplier en el Trip.
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING

import config as cfg

if TYPE_CHECKING:
    from vrp.models import Trip, Vehicle


# ── Helpers ────────────────────────────────────────────────────────────────────

# Categorías que son consideradas "frágiles" como conjunto agregado en BY_REFERENCE
_FRAGILE_CATS = ("wine", "beer")
# Categorías que son consideradas "pesadas" como conjunto agregado
_HEAVY_CATS   = ("beer", "water")


def trip_warehouse_estimate(trip: "Trip", vehicle: "Vehicle") -> float:
    """
    Devuelve coste de almacén+descarga estimado (segundos) para un viaje
    bajo la mejor de las 3 estrategias. Cachea la elección en el Trip.
    """
    if not trip.stops:
        trip.chosen_strategy = "by_order"
        trip.unload_multiplier = 1.0
        trip.warehouse_cost = 0.0
        trip.pallets_used = 0
        trip.n_box_orders = 0
        trip.stack_violations = 0
        trip.fragile_at_back_van = 0
        return 0.0

    n_stops = len(trip.stops)
    is_van = vehicle.vehicle_type == "van_3"
    lifo_mult = vehicle.lifo_penalty_multiplier
    effective_cap_base = vehicle.capacity_pallets

    # ── Acumuladores en single-pass ───────────────────────────────────────────
    cats: set = set()
    cats_normal: set = set()
    has_returnables = False
    n_priority_late = 0
    n_special = 0
    n_box_orders = 0
    n_returns_stops = 0
    max_fragility = 0
    max_weight = 0

    n_fragile_pallets_byorder = 0
    n_heavy_pallets_byorder   = 0
    n_fragile_pallets_special = 0
    n_fragile_at_back_van = 0

    # Fracciones de palet por categoría
    frac_per_cat: dict = {}
    frac_per_cat_normal: dict = {}
    sum_special_pallets = 0     # palets de pedidos especiales (HYBRID)

    for i, s in enumerate(trip.stops):
        cats.update(s.categories)
        has_returnables |= s.has_returnables

        if s.fragility_score > max_fragility:
            max_fragility = s.fragility_score
        if s.weight_score > max_weight:
            max_weight = s.weight_score

        is_sp = s.is_special
        if is_sp:
            n_special += 1
        else:
            cats_normal.update(s.categories)

        if s.is_priority and i > n_stops // 3:
            n_priority_late += 1

        if s.has_returnables:
            n_returns_stops += 1

        if is_van and s.is_fragile and i > 0:
            n_fragile_at_back_van += 1

        if s.needs_pallet:
            pallets_this = math.ceil(s.est_pallet_fraction)
            if s.is_fragile:
                n_fragile_pallets_byorder += pallets_this
                if is_sp:
                    n_fragile_pallets_special += pallets_this
            if s.is_heavy:
                n_heavy_pallets_byorder += pallets_this

            if is_sp:
                sum_special_pallets += pallets_this

            if s.categories:
                share = s.est_pallet_fraction / len(s.categories)
                for cat in s.categories:
                    frac_per_cat[cat] = frac_per_cat.get(cat, 0.0) + share
                    if not is_sp:
                        frac_per_cat_normal[cat] = frac_per_cat_normal.get(cat, 0.0) + share
        else:
            n_box_orders += 1

    # ── Sumarios derivados ────────────────────────────────────────────────────
    qc_total = n_stops * cfg.WH_T_QC_PER_ORDER
    returns_extra = n_returns_stops * cfg.WH_T_RETURNS_PICKUP_PER_STOP
    effective_cap = effective_cap_base - (1 if has_returnables else 0)
    if effective_cap < 1:
        effective_cap = 1
    fragility_scale = cfg.WH_FRAGILITY_PEN_SCALE.get(max_fragility, 0.0)

    # Stack violations: cuántos palets frágiles tienen pesados encima
    def stack_pen(n_frag, n_heavy):
        return min(n_frag, n_heavy) * cfg.WH_PEN_STACK_VIOLATION * fragility_scale

    # ── BY_ORDER ──────────────────────────────────────────────────────────────
    n_pallets_byorder = sum(
        math.ceil(s.est_pallet_fraction) for s in trip.stops if s.needs_pallet
    )
    wh_byorder = (
        n_pallets_byorder * cfg.WH_T_PALLET_BY_ORDER
        + n_box_orders    * cfg.WH_T_BOX_ORDER
        + qc_total
        + n_priority_late * cfg.WH_PEN_LOAD_ORDER * lifo_mult
        + stack_pen(n_fragile_pallets_byorder, n_heavy_pallets_byorder)
    )
    if is_van:
        wh_byorder += n_fragile_at_back_van * cfg.WH_PEN_FRAGILE_DEEP_VAN
    if n_pallets_byorder > effective_cap:
        wh_byorder += cfg.WH_PEN_OVER_CAPACITY * (n_pallets_byorder - effective_cap)

    delivery_extra_byorder = (
        (cfg.WH_UNLOAD_MULT_BY_ORDER - 1.0) * n_stops * cfg.SERVICE_TIME_BASE
        + returns_extra
    )
    total_byorder = wh_byorder + delivery_extra_byorder

    # ── BY_REFERENCE ──────────────────────────────────────────────────────────
    n_pallets_byref = sum(math.ceil(f) for f in frac_per_cat.values())
    if n_box_orders > 0:
        n_pallets_byref += 1   # mini-palet "varios"

    n_fragile_byref = sum(
        math.ceil(frac_per_cat[c]) for c in frac_per_cat if c in _FRAGILE_CATS
    )
    n_heavy_byref = sum(
        math.ceil(frac_per_cat[c]) for c in frac_per_cat if c in _HEAVY_CATS
    )

    wh_byref = (
        n_pallets_byref * cfg.WH_T_PALLET_BY_REFERENCE
        + qc_total
        + n_priority_late * cfg.WH_PEN_LOAD_ORDER * lifo_mult * 1.5
        + stack_pen(n_fragile_byref, n_heavy_byref)
    )
    if max_fragility >= 2 and max_weight >= 2:
        wh_byref += cfg.WH_PEN_FRAGILE_MIX * 2 * fragility_scale
    if has_returnables and n_pallets_byref >= effective_cap:
        wh_byref += cfg.WH_PEN_RETURN_SPACE
    if n_pallets_byref > effective_cap:
        wh_byref += cfg.WH_PEN_OVER_CAPACITY * (n_pallets_byref - effective_cap)
    if is_van and max_fragility >= 2 and n_stops > 1:
        wh_byref += cfg.WH_PEN_FRAGILE_DEEP_VAN * (n_stops - 1)

    delivery_extra_byref = (
        (cfg.WH_UNLOAD_MULT_BY_REFERENCE - 1.0) * n_stops * cfg.SERVICE_TIME_BASE
        + returns_extra
    )
    total_byref = wh_byref + delivery_extra_byref

    # ── HYBRID ────────────────────────────────────────────────────────────────
    n_pallets_normal = sum(math.ceil(f) for f in frac_per_cat_normal.values())
    n_pallets_hybrid = sum_special_pallets + n_pallets_normal
    n_heavy_normal = sum(
        math.ceil(frac_per_cat_normal[c]) for c in frac_per_cat_normal if c in _HEAVY_CATS
    )

    wh_hybrid = (
        sum_special_pallets * cfg.WH_T_PALLET_HYBRID_SPECIAL
        + n_pallets_normal  * cfg.WH_T_PALLET_HYBRID_BULK
        + n_box_orders      * cfg.WH_T_BOX_ORDER
        + qc_total
        + max(0, n_priority_late - n_special) * cfg.WH_PEN_LOAD_ORDER * lifo_mult
        + stack_pen(n_fragile_pallets_special, n_heavy_normal)
    )
    if is_van:
        wh_hybrid += n_fragile_at_back_van * cfg.WH_PEN_FRAGILE_DEEP_VAN
    if has_returnables and n_pallets_hybrid >= effective_cap:
        wh_hybrid += cfg.WH_PEN_RETURN_SPACE
    if n_pallets_hybrid > effective_cap:
        wh_hybrid += cfg.WH_PEN_OVER_CAPACITY * (n_pallets_hybrid - effective_cap)

    delivery_extra_hybrid = (
        (cfg.WH_UNLOAD_MULT_HYBRID - 1.0) * n_stops * cfg.SERVICE_TIME_BASE
        + returns_extra
    )
    total_hybrid = wh_hybrid + delivery_extra_hybrid

    # Recarga: setup adicional en cualquier estrategia
    if trip.is_reload:
        bonus = cfg.WH_T_PALLET_BY_REFERENCE * 0.5
        total_byorder += bonus
        total_byref   += bonus
        total_hybrid  += bonus

    # ── Elegir la mejor ──────────────────────────────────────────────────────
    candidates = (
        (total_byorder, "by_order",     wh_byorder, cfg.WH_UNLOAD_MULT_BY_ORDER,
            n_pallets_byorder, min(n_fragile_pallets_byorder, n_heavy_pallets_byorder)),
        (total_byref,   "by_reference", wh_byref,   cfg.WH_UNLOAD_MULT_BY_REFERENCE,
            n_pallets_byref, min(n_fragile_byref, n_heavy_byref)),
        (total_hybrid,  "hybrid",       wh_hybrid,  cfg.WH_UNLOAD_MULT_HYBRID,
            n_pallets_hybrid, min(n_fragile_pallets_special, n_heavy_normal)),
    )

    best = min(candidates, key=lambda x: x[0])

    trip.chosen_strategy     = best[1]
    trip.warehouse_cost      = best[2]
    trip.unload_multiplier   = best[3]
    trip.pallets_used        = best[4]
    trip.stack_violations    = best[5]
    trip.n_box_orders        = n_box_orders
    trip.fragile_at_back_van = n_fragile_at_back_van if is_van else 0

    return best[0]
