from __future__ import annotations
import os
from typing import Optional, Dict, Tuple, List
from datetime import datetime

import pandas as pd
import numpy as np
import requests
from scipy.spatial import KDTree


try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except Exception:
    HAS_GEOPANDAS = False


class ValidatedFloodPredictor:
    """
    A service that provides flood predictions based on validated return period 
    data.
    - accepts data_dir
    - exposes predict_by_query() and predict_by_coords()
    - returns JSON-friendly dicts
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.locations_df = self._load_locations()
        self.return_periods_df = self._load_return_periods()
        self.basin_gdf = self._load_basins()
        self._build_indexes()

    # -------- file helpers --------
    def _p(self, *parts): return os.path.join(self.data_dir, *parts)

    def _load_locations(self) -> pd.DataFrame:
        path = self._p("global_hybas_locations.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        return pd.read_csv(path, dtype={"gauge_id": str})

    def _load_return_periods(self) -> pd.DataFrame:
        path = self._p("global_return_periods.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        df = pd.read_csv(path, dtype={"gauge_id": str})
        return df[df["return_period_2"] >= 1].copy()  # keep reasonable ranges

    def _load_basins(self):
        if not HAS_GEOPANDAS:
            return None
        shp = self._p("BasinATLAS_v10_lev07.shp")
        if not os.path.exists(shp):
            return None
        try:
            return gpd.read_file(shp)
        except Exception:
            return None

    def _build_indexes(self):
        coords = self.locations_df[["latitude", "longitude"]].values
        self.location_tree = KDTree(coords)
        self.rp_data: Dict[str, Dict[int, float]] = {}

        for _, row in self.return_periods_df.iterrows():
            gid = row["gauge_id"]
            rp = {}
            for col in self.return_periods_df.columns:
                if col.startswith("return_period_"):
                    try:
                        years = int(col.split("_")[-1])
                        val = float(row[col])
                        if pd.notna(val) and val > 0:
                            rp[years] = val
                    except Exception:
                        pass
            if rp:
                self.rp_data[gid] = rp

    # -------- public entry points --------
    def predict_by_query(self, location: str) -> Optional[Dict]:
        coords = self._geocode(location)
        if not coords:
            return None
        return self._predict_core(coords["lat"], coords["lon"])

    def predict_by_coords(self, lat: float, lon: float) -> Optional[Dict]:
        return self._predict_core(lat, lon)

    # -------- internals --------
    def _geocode(self, location: str):
        try:
            r = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": location, "format": "json", "limit": 1},
                headers={"User-Agent": "ValidatedFloodPredictor"},
                timeout=10,
            )
            if r.status_code == 200 and r.json():
                d = r.json()[0]
                return {
                    "lat": float(d["lat"]),
                    "lon": float(d["lon"]),
                    "name": d["display_name"],
                }
        except Exception:
            pass
        return None

    def _predict_core(self, lat: float, lon: float) -> Optional[Dict]:
        basin = self._find_basin(lat, lon)
        if not basin:
            return None
        main = self._find_main_river_coords(lat, lon)
        if not main:
            return None
        glofas = self._get_glofas_data(*main)
        if not glofas:
            return None
        thresholds = self.rp_data.get(basin["gauge_id"], {})
        return {"basin": basin, "glofas": glofas, "thresholds": thresholds}

    def _find_basin(self, lat: float, lon: float) -> Optional[Dict]:
        # Prefer precise polygon match
        if self.basin_gdf is not None:
            try:
                point = Point(lon, lat)
                matches = self.basin_gdf[
                    self.basin_gdf.geometry.contains(point)
                ]
                if not matches.empty:
                    row = matches.iloc[0]
                    gid = f"hybas_{row['HYBAS_ID']}"
                    if gid in self.rp_data:
                        return {
                            "gauge_id": gid,
                            "distance_km": 0.0,
                            "method": "Point-in-Polygon",
                        }
            except Exception:
                pass
        # Fallback KDTree
        dists, idxs = self.location_tree.query([lat, lon], k=200)
        candidates = []
        for dist, idx in zip(dists, idxs):
            km = float(dist) * 111
            if km > 50:
                break
            gid = self.locations_df.iloc[int(idx)]["gauge_id"]
            if gid in self.rp_data:
                t2 = self.rp_data[gid].get(2, 0)
                candidates.append({
                    "gauge_id": gid,
                    "distance_km": km,
                    "t2": t2
                })
        if not candidates:
            return None
        majors = [c for c in candidates if c["t2"] > 1000]
        best = (
            min(majors, key=lambda c: c["distance_km"])
            if majors
            else max(candidates, key=lambda c: c["t2"])
        )
        return {"gauge_id": best["gauge_id"], "distance_km": best["distance_km"], "method": "KDTree-MajorRiver"}

    def _find_main_river_coords(self, lat: float, lon: float) -> Optional[Tuple[float, float]]:
        offsets = np.linspace(-0.1, 0.1, 5)
        grid = [(lat + dx, lon + dy) for dx in offsets for dy in offsets]
        vals: List[Tuple[float, float, float]] = []
        for la, lo in grid:
            try:
                r = requests.get(
                    "https://flood-api.open-meteo.com/v1/flood",
                    params={"latitude": la, "longitude": lo, "daily": "river_discharge", "past_days": 1, "forecast_days": 7},
                    timeout=6,
                )
                if r.status_code == 200:
                    q = r.json().get("daily", {}).get("river_discharge", [])
                    if q:
                        v = q[-2] if len(q) > 1 else q[-1]
                        if v and v > 0:
                            vals.append((la, lo, float(v)))
            except Exception:
                pass
        if not vals:
            return None
        # simple clustering by distance < 10 km
        clusters = []
        for la, lo, v in sorted(vals, key=lambda x: x[2], reverse=True):
            placed = False
            for c in clusters:
                import math
                if math.sqrt((la - c["lat"])**2 + (lo - c["lon"])**2) * 111 < 10:
                    c["pts"].append((la, lo, v))
                    c["lat"] = float(np.mean([p[0] for p in c["pts"]]))
                    c["lon"] = float(np.mean([p[1] for p in c["pts"]]))
                    c["mean"] = float(np.mean([p[2] for p in c["pts"]]))
                    placed = True
                    break
            if not placed:
                clusters.append({"lat": la, "lon": lo, "mean": v, "pts": [(la, lo, v)]})
        best = max(clusters, key=lambda c: c["mean"])
        return (best["lat"], best["lon"]) if best["mean"] > 5 else None

    def _get_glofas_data(self, lat: float, lon: float) -> Optional[Dict]:
        try:
            r = requests.get(
                "https://flood-api.open-meteo.com/v1/flood",
                params={"latitude": lat, "longitude": lon, "daily": "river_discharge", "forecast_days": 7},
                timeout=10,
            )
            if r.status_code == 200:
                d = r.json().get("daily", {})
                q = d.get("river_discharge", [])
                t = d.get("time", [])
                if q and t:
                    today = datetime.utcnow().strftime("%Y-%m-%d")
                    current = float(q[-2] if len(q) > 1 else q[-1])
                    forecast = [{"date": ti, "discharge": float(qi)} for ti, qi in zip(t, q) if ti > today][:7]
                    return {"current": current, "forecast": forecast}
        except Exception:
            pass
        return None


# Singleton factory (only load data once)
_predictor_singleton: Optional[ValidatedFloodPredictor] = None


def get_predictor() -> ValidatedFloodPredictor:
    global _predictor_singleton
    if _predictor_singleton is None:
        data_dir = os.getenv("FLOOD_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
        _predictor_singleton = ValidatedFloodPredictor(data_dir=os.path.abspath(data_dir))
    return _predictor_singleton
