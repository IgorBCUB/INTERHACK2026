"""Domain models for the VRP solver."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class Order:
    """A single delivery stop."""
    id: str                    # entrega ID o client_id (según loader)
    client_id: str
    client_name: str
    address: str
    postal_code: str
    city: str
    lat: float = 0.0
    lon: float = 0.0

    # Load attributes (in uniform 'boxes' unit after normalisation)
    volume_boxes: float = 0.0
    weight_kg: float = 0.0

    # Time window (seconds from midnight)
    tw_open: int = 0
    tw_close: int = 86399

    # Service time at this stop (seconds)
    service_time: int = 600

    # Priority: 1 = highest, 3 = normal
    priority: int = 3

    zone: str = ""

    # Pre-computed parking search time (seconds) — deterministic per client
    parking_time: int = 0

    # ── Warehouse-aware fields (multilevel) ───────────────────────────────────
    n_references: int = 1                   # nº materiales distintos en el pedido
    categories: tuple = ()                  # ej ('beer','water')
    fragility_score: int = 1                # max sobre productos del pedido (0-3)
    weight_score: int = 1                   # max (0-3)
    has_returnables: bool = False
    all_stackable: bool = True              # AND lógico de stackable de cada producto
    est_pallet_fraction: float = 0.0        # suma cajas/Contador
    needs_pallet: bool = False              # fraction ≥ 0.3

    # ── Helpers / aliases ─────────────────────────────────────────────────────
    @property
    def is_fragile(self) -> bool:
        return self.fragility_score >= 2

    @property
    def is_heavy(self) -> bool:
        return self.weight_score >= 2

    @property
    def is_priority(self) -> bool:
        return self.priority < 3

    @property
    def is_special(self) -> bool:
        return self.is_fragile or self.is_priority

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Order) and self.id == other.id


@dataclass
class Vehicle:
    """A delivery vehicle / driver."""
    id: str
    driver_name: str
    capacity_boxes: float = 200.0   # legado / informativo
    max_route_seconds: int = 28800  # 8 h per single trip
    max_daily_seconds: int = 28800  # 8 h total working day
    start_lat: float = 0.0
    start_lon: float = 0.0

    # ── Vehicle-type-aware fields ─────────────────────────────────────────────
    vehicle_type: str = "truck_6"               # "truck_8" | "truck_6" | "van_3"
    capacity_pallets: int = 6                   # restricción dura
    has_side_access: bool = True                # False solo para van_3
    lifo_penalty_multiplier: float = 1.0        # 3.0 para van_3

    # Current state (used during construction)
    current_load: float = field(default=0.0, repr=False)
    time_used: float = field(default=0.0, repr=False)


@dataclass
class Trip:
    """One round-trip: depot → [stops] → depot."""
    vehicle_id: str
    stops: List[Order] = field(default_factory=list)
    total_time_seconds: float = 0.0
    total_load_boxes: float = 0.0
    total_distance_km: float = 0.0
    is_reload: bool = False       # True if this is a 2nd+ trip for this vehicle

    # ── Cacheados por el estimador rápido ────────────────────────────────────
    chosen_strategy: str = "hybrid"             # "by_order" | "by_reference" | "hybrid"
    unload_multiplier: float = 1.0
    warehouse_cost: float = 0.0
    pallets_used: int = 0
    n_box_orders: int = 0
    stack_violations: int = 0
    fragile_at_back_van: int = 0


@dataclass
class Solution:
    """Complete VRP solution: a set of trips across all vehicles."""
    trips: List[Trip] = field(default_factory=list)

    # ── Metrics (populated by evaluate()) ─────────────────────────────────────
    total_time_seconds: float = 0.0
    total_reload_penalty: float = 0.0
    unserved_orders: List[Order] = field(default_factory=list)
    cost: float = float("inf")
    daily_time_per_vehicle: Dict[str, float] = field(default_factory=dict)

    def num_vehicles_used(self) -> int:
        return len({t.vehicle_id for t in self.trips})

    def num_trips(self) -> int:
        return len(self.trips)

    def num_reloads(self) -> int:
        from collections import Counter
        c = Counter(t.vehicle_id for t in self.trips)
        return sum(v - 1 for v in c.values())
