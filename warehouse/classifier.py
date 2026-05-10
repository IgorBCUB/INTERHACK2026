"""
Clasificador de productos: prefijo + denominación → fragility/weight/category/returnable.

Usa una tabla curada (data/product_profiles.json) si existe, si no aplica reglas heurísticas.
"""
from __future__ import annotations
import json
import os
from functools import lru_cache
from typing import Dict, Optional


# ── Mapeo prefijo → categoría ─────────────────────────────────────────────────
PREFIX_CATEGORY: Dict[str, str] = {
    "0CF":  "coffee",
    "0AG":  "water",
    "0RF":  "softdrink",
    "0VE":  "wine",
    "0AM":  "sweetener",
    "0LM":  "dairy",
    "0ZU":  "juice",
    "ED":   "beer",
    "CJ":   "empty_pkg",
    "3ENV": "packaging",
    "XI":   "icecream",
    "UE":   "utensil",
}

# ── Keywords sobre denominación ───────────────────────────────────────────────
ROBUST_KW    = ("BARRIL", "KEG", "PET", "PLASTICO", "LATA", "CAN ")
GLASS_KW     = ("CRISTAL", "VIDRIO")
BOTTLE_KW    = ("BOT.", "BOTELLA")
HEAVY_KW     = ("BARRIL", "KEG")
HEAVY_VOL_KW = ("5L", "8L", "10L", "GARRAFA")
RETURNABLE_KW = ("RET", "RETORN", "GARRAFA", "ENVASE", "/-")


# ── Carga de tabla curada (lazy / cacheada) ───────────────────────────────────
_CURATED: Optional[Dict[str, dict]] = None

def _load_curated() -> Dict[str, dict]:
    global _CURATED
    if _CURATED is not None:
        return _CURATED
    import config as cfg
    path = cfg.PRODUCT_PROFILES_PATH
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                _CURATED = json.load(f)
                return _CURATED
        except Exception:
            pass
    _CURATED = {}
    return _CURATED


# ── Reglas heurísticas ────────────────────────────────────────────────────────

def classify_category(material_code: str) -> str:
    for prefix, cat in PREFIX_CATEGORY.items():
        if material_code.startswith(prefix):
            return cat
    return "generic"


def classify_fragility(material_code: str, denom_upper: str, category: str) -> int:
    """0 = robusto, 1 = normal, 2 = frágil, 3 = muy frágil (vidrio premium)."""
    # Robustos (override)
    if any(k in denom_upper for k in ROBUST_KW):
        return 0
    # Vidrio explícito
    if any(k in denom_upper for k in GLASS_KW):
        return 3
    if any(k in denom_upper for k in BOTTLE_KW):
        return 2
    # Vino → siempre vidrio
    if category == "wine":
        return 3
    # Cerveza: depende de envase
    if category == "beer":
        if "KEG" in denom_upper or "BARRIL" in denom_upper:
            return 0
        if "PET" in denom_upper or "LATA" in denom_upper:
            return 0
        return 2  # default: botella vidrio
    # Refresco vidrio retornable
    if category == "softdrink" and "RET" in denom_upper:
        return 2
    return 1


def classify_weight(material_code: str, denom_upper: str, category: str) -> int:
    """0 = ligero, 1 = medio, 2 = pesado, 3 = muy pesado (kegs, palets pre-formados)."""
    if any(k in denom_upper for k in HEAVY_KW):
        return 3
    if any(t in denom_upper for t in HEAVY_VOL_KW):
        return 2
    if category in ("beer", "water"):
        return 2
    if category in ("softdrink", "juice", "dairy"):
        return 1
    return 1


def classify_returnable(material_code: str, denom_upper: str) -> bool:
    if any(k in denom_upper for k in RETURNABLE_KW):
        return True
    if material_code.startswith(("CJ", "3ENV")):
        return True
    return False


# ── API pública ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def classify(material_code: str, denom: str) -> dict:
    """Devuelve dict con category, fragility_score, weight_score, stackable, is_returnable."""
    if not material_code:
        return {
            "category":        "generic",
            "fragility_score": 1,
            "weight_score":    1,
            "stackable":       True,
            "is_returnable":   False,
        }

    mc = material_code.strip()
    d = (denom or "").upper()

    # 1) Tabla curada
    curated = _load_curated()
    profile = curated.get(mc)
    if profile:
        return {
            "category":        profile["category"],
            "fragility_score": int(profile["fragility"]),
            "weight_score":    int(profile["weight"]),
            "stackable":       bool(profile["stackable"]),
            "is_returnable":   classify_returnable(mc, d),
        }

    # 2) Heurística
    cat = classify_category(mc)
    fragility = classify_fragility(mc, d, cat)
    weight    = classify_weight(mc, d, cat)
    return {
        "category":        cat,
        "fragility_score": fragility,
        "weight_score":    weight,
        "stackable":       fragility < 2,   # frágiles ≥2 no son apilables
        "is_returnable":   classify_returnable(mc, d),
    }


def reset_cache():
    """Útil para tests cuando se cambia data/product_profiles.json."""
    global _CURATED
    _CURATED = None
    classify.cache_clear()
