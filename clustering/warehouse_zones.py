"""
Warehouse zone assignment for clusters.

Each cluster gets a physical zone in the DDI Mollet warehouse (A–H).
Zone is assigned based on departure time (earliest TW_open cluster → Zone A).
This means warehouse staff prepare Zone A first, then B, etc.
"""
from __future__ import annotations
from typing import List

from clustering.models import Cluster
import config


def assign_warehouse_zones(clusters: List[Cluster]) -> None:
    """
    Assign warehouse zones to clusters in-place.
    Clusters are sorted by mean TW_open (earliest first → Zone A).
    """
    zones = config.WAREHOUSE_ZONES

    def avg_tw_open(c: Cluster) -> float:
        if not c.orders:
            return 86399.0
        return sum(o.tw_open for o in c.orders) / len(c.orders)

    sorted_clusters = sorted(clusters, key=avg_tw_open)

    for i, cluster in enumerate(sorted_clusters):
        cluster.warehouse_zone = zones[i % len(zones)]


def warehouse_zone_report(clusters: List[Cluster]) -> dict:
    """Return a summary of zone assignments for reporting."""
    zones_used: dict = {}
    for c in clusters:
        z = c.warehouse_zone or "?"
        if z not in zones_used:
            zones_used[z] = {"n_clusters": 0, "n_orders": 0, "n_pallets": 0}
        zones_used[z]["n_clusters"] += 1
        zones_used[z]["n_orders"] += len(c.orders)
        zones_used[z]["n_pallets"] += c.n_pallets
    return dict(sorted(zones_used.items()))
