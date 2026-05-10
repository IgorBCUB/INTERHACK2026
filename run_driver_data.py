"""
Generates driver data per vehicle from results_clustering.json.
Output: results_driver.json — one entry per (day, vehicle)
"""
from __future__ import annotations
import json, sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

DAYS = ["16/03/2026","17/03/2026","18/03/2026","19/03/2026","20/03/2026"]
WEEKDAYS = ["Lunes","Martes","Miércoles","Jueves","Viernes"]

CAT_LABEL = {
    "beer":"Cerveza","water":"Agua","wine":"Vino","softdrink":"Refresco",
    "dairy":"Lácteo","coffee":"Café","sweetener":"Edulcorante","packaging":"Embalaje",
    "empty_pkg":"Envase vacío","generic":"Genérico","utensil":"Utensilio",
}

VEHICLE_TYPE_LABEL = {"truck_8":"Camión 8P","truck_6":"Camión 6P","van_3":"Furgoneta 3P"}

def fmt_time(s):
    if not s: return "—"
    h = int(s)//3600; m = (int(s)%3600)//60
    return f"{h}h {m:02d}m"


def build_driver_data():
    with open("results_clustering.json") as f:
        clustering = json.load(f)

    result = []

    for day_idx, day in enumerate(clustering):
        date_str = day["date"]
        weekday  = day["weekday"]
        depot    = day["depot"]

        for route in day["routes"]:
            vid   = route["vehicle_id"]
            vtype = route["vehicle_type"]
            driver= route["driver_name"]
            cap   = route["capacity_pallets"]

            all_stops = []
            total_time = 0.0
            total_fuel = 0.0

            for trip_i, trip in enumerate(route["trips"]):
                for stop_j, stop in enumerate(trip["stops"]):
                    # Build product list from category data
                    products = []
                    # Group by category
                    cats_at_stop = {}
                    # We don't have per-stop breakdown by cat in clustering output
                    # Use boxes as proxy with fragility info
                    products.append({
                        "name": f"{stop['boxes']:.0f} caixes",
                        "type": "caixa",
                        "qty": int(stop["boxes"]),
                        "fragility": stop.get("fragility", 0),
                        "n_pallets": stop.get("n_pallets", 0),
                    })

                    all_stops.append({
                        "stop_num":    len(all_stops) + 1,
                        "trip_num":    trip_i + 1,
                        "client_id":   stop["client_id"],
                        "name":        stop["client_name"],
                        "address":     stop.get("city", "") + ", " + stop.get("zone", ""),
                        "city":        stop["city"],
                        "lat":         stop["lat"],
                        "lon":         stop["lon"],
                        "boxes":       stop["boxes"],
                        "n_pallets":   round(stop.get("n_pallets", 0), 1),
                        "fragility":   stop.get("fragility", 0),
                        "tw_open":     stop.get("tw_open", 0),
                        "tw_close":    stop.get("tw_close", 86399),
                        "products":    products,
                        "has_returnables": False,
                        "is_reload_trip": trip.get("is_reload", False),
                        "warehouse_zone": trip.get("warehouse_zone", "?"),
                    })

                total_time += trip.get("total_time_seconds", 0)
                total_fuel += trip.get("total_fuel_litres", 0)

            if not all_stops:
                continue

            # Route stats
            total_km = round(total_fuel / 0.08, 1)  # approx: 8L/100km diesel
            co2_kg   = round(total_fuel * 2.68, 1)
            co2_saved = round(co2_kg * 0.22, 1)  # estimate vs non-optimized

            result.append({
                "date":         date_str,
                "weekday":      weekday,
                "vehicle_id":   vid,
                "vehicle_type": vtype,
                "vehicle_label": VEHICLE_TYPE_LABEL.get(vtype, vtype),
                "capacity_pallets": cap,
                "driver_name":  driver,
                "n_stops":      len(all_stops),
                "n_trips":      len(route["trips"]),
                "total_time_s": round(total_time, 0),
                "total_time_fmt": fmt_time(total_time),
                "total_fuel_l": round(total_fuel, 1),
                "total_km":     total_km,
                "co2_kg":       co2_kg,
                "co2_saved_kg": co2_saved,
                "depot": {
                    "name": depot["name"],
                    "lat":  depot["lat"],
                    "lon":  depot["lon"],
                },
                "stops": all_stops,
            })

    with open("results_driver.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    n_routes = len(result)
    n_vehicles = len(set(r["vehicle_id"] for r in result))
    print(f"✓ results_driver.json — {n_routes} rutas, {n_vehicles} vehículos únicos")


if __name__ == "__main__":
    build_driver_data()
