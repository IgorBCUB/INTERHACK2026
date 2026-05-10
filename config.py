"""Central configuration for the VRP hackathon solver."""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Google Maps ────────────────────────────────────────────────────────────────
GMAPS_API_KEY = os.getenv("GMAPS_API_KEY", "")

# ── Depot (DDI Mollet del Vallès) ──────────────────────────────────────────────
DEPOT = {
    "id": "DEPOT",
    "name": "DDI Mollet del Vallès",
    "address": "Carrer de Gaietà Vinzia 16, 08100 Mollet del Vallès, Barcelona",
    "lat": 41.5388,
    "lon": 2.2136,
}

# ── Vehicle fleet defaults ─────────────────────────────────────────────────────
DEFAULT_MAX_ROUTE_SECONDS = 8 * 3600      # 8 h max per route leg
DEFAULT_MAX_DAILY_SECONDS = 8 * 3600      # 8 h max working day (total across all trips)
DEFAULT_SERVICE_TIME_SECONDS = 600        # fallback only; real value set per order
DEPOT_RELOAD_PENALTY_SECONDS = 1800       # 30 min penalisation per depot return

# ── Unload time (proportional to order size) ───────────────────────────────────
UNLOAD_BASE_SECONDS   = 300   # 5 min minimum per stop (any size)
UNLOAD_PER_PALLET_S   = 420   # 7 min per full pallet (linear with pallet fraction)

# ── Parking search time per stop (deterministic per client via hash) ───────────
PARKING_MIN_SECONDS = 60    # 1 min  — easy street parking
PARKING_MAX_SECONDS = 360   # 6 min  — busy urban area

# ── Time window violation penalty ─────────────────────────────────────────────
# Priority derived from window width (loader.py): 1=tight(<1h), 2=medium(1-4h), 3=wide(>4h)
# Penalty = overdue_seconds × (4-priority) × TW_VIOLATION_MULTIPLIER
#   priority 1 (<1h window) → 9× base   (very strict)
#   priority 2 (1-4h)       → 6× base   (moderate)
#   priority 3 (>4h/all day)→ 3× base   (soft)
TW_VIOLATION_MULTIPLIER = 3.0

# ── Simulated Annealing ────────────────────────────────────────────────────────
SA_INITIAL_TEMP = 3000.0
SA_COOLING_RATE = 0.995
SA_MIN_TEMP = 5.0
SA_MAX_ITER_PER_TEMP = 50

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "Hackaton")
HACKATON_XLSX = os.path.join(DATA_DIR, "Hackaton.xlsx")
HORARIOS_XLSX = os.path.join(DATA_DIR, "Horarios Entrega.XLSX")
ZM040_XLSX = os.path.join(DATA_DIR, "ZM040.XLSX")

# ── Demo / fallback ────────────────────────────────────────────────────────────
# When no API key is available, use haversine distance * avg_speed as proxy
AVG_SPEED_KMH = 35  # urban average


# ══════════════════════════════════════════════════════════════════════════════
# Warehouse module — partial coupling
# ══════════════════════════════════════════════════════════════════════════════

# ── Acoplado parcial ──────────────────────────────────────────────────────────
WH_COUPLING_ENABLED = True
WH_COUPLING_WEIGHT  = 1.0          # tunable

# ── Tiempos por palet (anclaje: 8 palets by_ref ≈ 1 h) ───────────────────────
WH_T_PALLET_BY_REFERENCE   = 450    # palet ya hecho, mover + cargar
WH_T_PALLET_BY_ORDER       = 1200   # picking caja a caja + montaje + carga
WH_T_PALLET_HYBRID_SPECIAL = 1200
WH_T_PALLET_HYBRID_BULK    = 450

# ── Pedidos pequeños / QC / retornos ──────────────────────────────────────────
WH_T_BOX_ORDER               = 300   # bultos sueltos por pedido
WH_T_QC_PER_ORDER            = 30    # control de calidad por pedido (todas las estrategias)
WH_T_RETURNS_PICKUP_PER_STOP = 90    # tiempo extra en parada con retornos

# ── Service time base por parada ──────────────────────────────────────────────
SERVICE_TIME_BASE = 600

# ── Multiplicador de descarga por estrategia ──────────────────────────────────
WH_UNLOAD_MULT_BY_ORDER     = 1.00
WH_UNLOAD_MULT_HYBRID       = 1.15
WH_UNLOAD_MULT_BY_REFERENCE = 1.40

# ── LIFO multiplier por tipo de vehículo ──────────────────────────────────────
WH_LIFO_MULTIPLIER = {
    "truck_8": 1.0,
    "truck_6": 1.0,
    "van_3":   3.0,    # crítico: sin acceso lateral
}

# ── Penalizaciones (segundos, escaladas por fragility_score) ──────────────────
WH_PEN_FRAGILE_MIX       = 1500
WH_PEN_STACK_VIOLATION   = 800
WH_PEN_FRAGILE_DEEP_VAN  = 1000
# SA stacking: penalty per unit of depth × fragility for fragile stops buried deep
STACKING_PEN_PER_DEPTH   = 600   # seconds per depth unit × fragility_score
WH_PEN_LOAD_ORDER       = 200
WH_PEN_OVER_CAPACITY    = 5000
WH_PEN_RETURN_SPACE     = 400

WH_FRAGILITY_PEN_SCALE = {0: 0.0, 1: 0.3, 2: 1.0, 3: 1.5}

# ── Composición de la flota ───────────────────────────────────────────────────
FLEET_COMPOSITION = {"truck_8": 4, "truck_6": 11, "van_3": 1}

# Capacidad de palets por tipo de vehículo
VEHICLE_CAPACITY_PALLETS = {"truck_8": 8, "truck_6": 6, "van_3": 3}

# Acceso lateral por tipo de vehículo
VEHICLE_HAS_SIDE_ACCESS = {"truck_8": True, "truck_6": True, "van_3": False}

# ── Umbral palet vs caja ──────────────────────────────────────────────────────
WH_PALLET_THRESHOLD = 0.3        # fraction < 0.3 → bultos sueltos

# ── SA almacén (refinamiento final, opcional fase 2/3) ────────────────────────
WH_SA_INIT_TEMP    = 500
WH_SA_COOLING_RATE = 0.99
WH_SA_MIN_TEMP     = 5
WH_SA_ITER_PER_TEMP = 30

# ── Paths warehouse ───────────────────────────────────────────────────────────
PRODUCT_PROFILES_PATH = os.path.join(os.path.dirname(__file__), "data", "product_profiles.json")


# ══════════════════════════════════════════════════════════════════════════════
# Logistics clustering
# ══════════════════════════════════════════════════════════════════════════════

# Weights for logistics distance: D = α×T + β×F×310 + γ×dist_proxy + δ×penalty
CLUSTERING_ALPHA = 0.50   # travel time weight
CLUSTERING_BETA  = 0.30   # fuel cost weight
CLUSTERING_GAMMA = 0.10   # distance proxy weight
CLUSTERING_DELTA = 0.10   # logistics penalty weight

CLUSTERING_MAX_ITER    = 50    # K-Medoids iterations
CLUSTERING_REFINE_PASSES = 10  # post-clustering refinement passes

# Warehouse zones (8 zones, assigned by departure time order)
WAREHOUSE_ZONES = ["A", "B", "C", "D", "E", "F", "G", "H"]

# CO2 factor for cost function
CO2_KG_PER_LITRE   = 2.68   # kg CO2 per litre diesel (DEFRA 2023)
CO2_PENALTY_WEIGHT = 50.0   # seconds equivalent per kg CO2 (ESG factor)
