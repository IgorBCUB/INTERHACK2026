"""Data models for logistics clustering."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from vrp.models import Order, Vehicle


@dataclass
class Cluster:
    """A group of orders that will be served by one vehicle in one trip."""
    id: str
    orders: List[Order] = field(default_factory=list)
    vehicle: Optional[Vehicle] = None
    warehouse_zone: str = ""
    medoid_order_id: str = ""

    # Derived / cached
    n_pallets: int = 0
    total_boxes: float = 0.0
    intra_dist: float = 0.0
    has_fragile: bool = False
    has_priority: bool = False
    categories: tuple = ()

    def is_feasible(self, vehicle: Vehicle) -> bool:
        return self.n_pallets <= vehicle.capacity_pallets

    def update_stats(self) -> None:
        """Recompute derived fields from current orders list."""
        from vrp.savings import estimate_pallets_byref
        self.n_pallets = estimate_pallets_byref(self.orders)
        self.total_boxes = sum(o.volume_boxes for o in self.orders)
        self.has_fragile = any(o.is_fragile for o in self.orders)
        self.has_priority = any(o.is_priority for o in self.orders)
        cats: set = set()
        for o in self.orders:
            cats.update(o.categories)
        self.categories = tuple(sorted(cats))


@dataclass
class ClusterSolution:
    """Complete clustering result: a set of clusters with assigned vehicles."""
    clusters: List[Cluster] = field(default_factory=list)
    unassigned: List[Order] = field(default_factory=list)

    # Cost breakdown
    route_cost: float = 0.0
    warehouse_cost: float = 0.0
    fuel_cost_equiv: float = 0.0
    co2_cost_equiv: float = 0.0
    intra_cost: float = 0.0
    total_cost: float = 0.0

    def num_clusters(self) -> int:
        return len(self.clusters)

    def num_orders(self) -> int:
        return sum(len(c.orders) for c in self.clusters)

    def summary(self) -> dict:
        return {
            "n_clusters": self.num_clusters(),
            "n_orders": self.num_orders(),
            "n_unassigned": len(self.unassigned),
            "route_cost_h": round(self.route_cost / 3600, 2),
            "warehouse_cost_h": round(self.warehouse_cost / 3600, 2),
            "fuel_cost_equiv_h": round(self.fuel_cost_equiv / 3600, 2),
            "co2_cost_equiv_h": round(self.co2_cost_equiv / 3600, 2),
            "total_cost_h": round(self.total_cost / 3600, 2),
        }
