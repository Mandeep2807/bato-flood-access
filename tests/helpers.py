"""Builders for hand-written test metadata (never real satellite data;
source ids are prefixed "TEST-")."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from bato.contracts.acquisition import Acquisition
from bato.contracts.common import OrbitDirection, Polarization, PolygonGeometry, Sensor

REPO_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
FIXED_TODAY = date(2026, 10, 1)       # injected so date validation is deterministic
EVENT_DATE = date(2026, 8, 26)
FINGERPRINT = "a" * 64

# A small lon/lat box; any valid polygon works for metadata-only tests.
FOOTPRINT = PolygonGeometry(coordinates=((
    (85.20, 28.00), (85.50, 28.00), (85.50, 28.30), (85.20, 28.30), (85.20, 28.00),
),))


def s1(source_id: str, when: datetime, *, direction: OrbitDirection = OrbitDirection.DESCENDING,
       relative_orbit: int = 19, pols: tuple[Polarization, ...] = (Polarization.VV, Polarization.VH)
       ) -> Acquisition:
    """Build a Sentinel-1 acquisition record (metadata only)."""
    return Acquisition(
        source_id=source_id, sensor=Sensor.SENTINEL_1, platform="S1A",
        acquisition_start=when, product_type="IW_GRDH_1S", orbit_direction=direction,
        relative_orbit=relative_orbit, polarizations=pols, footprint=FOOTPRINT,
    )


def utc(y: int, m: int, d: int, h: int = 0, mi: int = 0) -> datetime:
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Synthetic CDSE-like records for Step 2B tests (TEST FIXTURE, not real data)
# ---------------------------------------------------------------------------

TEST_AOI_BBOX = (85.20, 28.00, 85.40, 28.20)
FOOTPRINTS = {
    "full": [[84.70, 27.40], [86.10, 27.62], [85.85, 28.95], [84.42, 28.73], [84.70, 27.40]],
    "south": [[84.70, 27.40], [86.10, 27.62], [86.00, 28.10], [84.60, 28.10], [84.70, 27.40]],
    "north": [[84.60, 28.08], [86.00, 28.08], [85.85, 28.95], [84.42, 28.73], [84.60, 28.08]],
    # covers lon >= 85.23, i.e. ~85 % of the test AOI
    "east85": [[85.23, 27.40], [86.10, 27.40], [86.10, 28.90], [85.23, 28.90], [85.23, 27.40]],
}


def odata_record(platform: str, direction: str, rel: int, start: datetime, *, footprint: str = "full",
                 pols: str = "VV&VH", drop: tuple[str, ...] = ()) -> dict:
    """One synthetic CDSE OData Sentinel-1 IW GRDH record (TEST FIXTURE)."""
    import hashlib
    stop = start.replace(second=min(start.second + 25, 59))
    key = f"{platform}{direction}{rel}{start.isoformat()}{footprint}"
    digest = hashlib.sha256(key.encode()).hexdigest().upper()
    pol_code = {"VV&VH": "DV", "HH&HV": "DH", "VV": "SV", "HH": "SH"}[pols]
    name = (f"S1{platform}_IW_GRDH_1S{pol_code}_{start:%Y%m%dT%H%M%S}_{stop:%Y%m%dT%H%M%S}_"
            f"{int(digest[:5], 16) % 999999:06d}_{digest[5:11]}_{digest[11:15]}.SAFE")
    attrs = {"platformSerialIdentifier": platform, "operationalMode": "IW", "productType": "IW_GRDH_1S",
             "polarisationChannels": pols, "orbitDirection": direction, "relativeOrbitNumber": rel}
    return {
        "Id": f"TEST-FIXTURE-{digest[:12]}", "Name": name,
        "ContentDate": {"Start": start.strftime("%Y-%m-%dT%H:%M:%S.000Z")},
        "GeoFootprint": {"type": "Polygon", "coordinates": [FOOTPRINTS[footprint]]},
        "Attributes": [{"Name": k, "Value": v} for k, v in attrs.items() if k not in drop],
    }
