from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, Dict, Any
import requests, asyncio, re

# ---------------------------
# Data models for location info
# ---------------------------

class LocationInfo(BaseModel):
    """Structured information about a single location."""
    query: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    display_name: Optional[str] = None
    country: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None  # Raw JSON returned by Nominatim


class LocationResult(BaseModel):
    """Wrapper model that holds a LocationInfo object."""
    location: LocationInfo


# ---------------------------
# Low-level synchronous helpers for Nominatim
# ---------------------------

def _nominatim_search_sync(query: str) -> Optional[Dict[str, Any]]:
    """Perform forward geocoding using OpenStreetMap Nominatim API."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1, "addressdetails": 1}
    headers = {
        "User-Agent": "multi-agent-flood-prediction-communication/1.0 (contact: you@example.com)",
        "Accept-Language": "en",
    }
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data[0] if data else None


def _nominatim_reverse_sync(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Perform reverse geocoding (coordinates → address) using Nominatim API."""
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 10, "addressdetails": 1}
    headers = {
        "User-Agent": "multi-agent-flood-prediction-communication/1.0 (contact: you@example.com)",
        "Accept-Language": "en",
    }
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------
# Async wrappers (to avoid blocking)
# ---------------------------

async def _geocode_nominatim(query: str) -> Optional[Dict[str, Any]]:
    """Run forward geocoding asynchronously using a background thread."""
    return await asyncio.to_thread(_nominatim_search_sync, query)


async def _reverse_nominatim(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Run reverse geocoding asynchronously using a background thread."""
    return await asyncio.to_thread(_nominatim_reverse_sync, lat, lon)


# ---------------------------
# Main service function
# ---------------------------

async def get_location_info(q: str) -> LocationResult:
    """
    Retrieve structured location information for a query string.

    Features:
    - Supports parsing "lat,lon" directly and performing reverse geocoding.
    - Otherwise performs normal text-based (forward) geocoding.
    - Returns a LocationResult object compatible with other agents.
    """
    q_str = (q or "").strip()
    loc = LocationInfo(query=q_str)

    # Check if the query is in "lat,lon" format
    m = re.match(r"^\s*([+-]?\d+(\.\d+)?)\s*,\s*([+-]?\d+(\.\d+)?)\s*$", q_str)
    if m:
        lat = float(m.group(1))
        lon = float(m.group(3))
        loc.latitude, loc.longitude = lat, lon
        loc.display_name = f"Coordinates ({lat},{lon})"
        try:
            rev = await _reverse_nominatim(lat, lon)
            if rev:
                # Update display name and extract country if available
                loc.display_name = rev.get("display_name") or loc.display_name
                addr = rev.get("address") or {}
                loc.country = addr.get("country")
                loc.raw = rev
        except Exception as e:
            loc.raw = {"reverse_error": str(e)}
        return LocationResult(location=loc)

    # Otherwise, perform forward geocoding
    try:
        geo = await _geocode_nominatim(q_str)
        if geo:
            loc.latitude = float(geo.get("lat")) if geo.get("lat") else None
            loc.longitude = float(geo.get("lon")) if geo.get("lon") else None
            loc.display_name = geo.get("display_name")
            loc.country = (geo.get("address") or {}).get("country")
            loc.raw = geo
    except Exception as e:
        loc.raw = {"error": str(e)}

    return LocationResult(location=loc)

def canonical_place_string(loc: LocationInfo) -> str:
    raw = (loc.display_name or loc.query or "Canada").strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts[0] if parts else "Canada"


# ---------------------------
# Test script (standalone use)
# ---------------------------

from app.services.location_service import get_location_info, canonical_place_string

async def main():
    # Example 1: Query by city name
    r1 = await get_location_info("Vancouver, BC")
    print("=== Vancouver, BC ===")
    print(r1.model_dump())
    print("canonical place:", canonical_place_string(r1.location))
    print()

    # Example 2: Query by coordinates
    r2 = await get_location_info("49.2827,-123.1207")
    print("=== 49.2827,-123.1207 ===")
    print(r2.model_dump())
    print("canonical place:", canonical_place_string(r2.location))

if __name__ == "__main__":
    asyncio.run(main())
