"""Parse Sentinel-1 product names (ESA naming convention).

Example: ``S1A_IW_GRDH_1SDV_20240101T001520_20240101T001545_051234_062ABC_1F2E.SAFE``

    MMM_BB_TTTR_LFPP_<start>_<stop>_<absolute orbit>_<datatake>_<crc>

  MMM  mission/platform  S1A, S1B, S1C, S1D
  BB   beam mode         IW, EW, WV, S1..S6, ...
  TTT  product type      RAW, SLC, GRD, OCN
  R    resolution class  F, H, M, or _ (none)
  L    processing level  0, 1, 2
  F    product class     S (standard), A (annotation), N (noise)
  PP   polarisation      SH, SV (single HH / VV), DH, DV (dual HH+HV / VV+VH),
                         or HH, VV, HV, VH (partial)

The name is an official, independent statement of a product's identity.
BATO uses it to cross-check catalogue attributes, never to invent values the
catalogue does not provide (orbit direction and relative orbit are NOT in
the name and must come from catalogue attributes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from bato.contracts.common import Polarization
from bato.exceptions import MetadataError

_NAME = re.compile(
    r"^(?P<platform>S1[A-D])_(?P<mode>[A-Z0-9]{2})_(?P<ptype>RAW|SLC|GRD|OCN)(?P<res>[FHM_])_"
    r"(?P<level>[0-2])(?P<pclass>[SAN])(?P<pol>SH|SV|DH|DV|HH|VV|HV|VH)_"
    r"(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_(?P<abs_orbit>\d{6})_(?P<datatake>[0-9A-F]{6})_"
    r"(?P<crc>[0-9A-F]{4})(?P<suffix>_COG)?(?:\.SAFE)?$"
)

POLARIZATION_CODES: dict[str, tuple[Polarization, ...]] = {
    "SH": (Polarization.HH,), "SV": (Polarization.VV,),
    "DH": (Polarization.HH, Polarization.HV), "DV": (Polarization.VV, Polarization.VH),
    "HH": (Polarization.HH,), "VV": (Polarization.VV,),
    "HV": (Polarization.HV,), "VH": (Polarization.VH,),
}


@dataclass(frozen=True)
class S1ProductName:
    name: str
    platform: str            # "S1A"
    mode: str                # "IW"
    product_kind: str        # "GRD"
    resolution: str          # "H"
    level: str               # "1"
    product_class: str       # "S"
    polarization_code: str   # "DV"
    start: datetime          # UTC, whole seconds
    stop: datetime
    absolute_orbit: int
    datatake_id: str
    is_cog: bool

    @property
    def product_type(self) -> str:
        """ESA product type string, e.g. ``IW_GRDH_1S``."""
        return f"{self.mode}_{self.product_kind}{self.resolution}_{self.level}{self.product_class}"

    @property
    def polarizations(self) -> tuple[Polarization, ...]:
        return POLARIZATION_CODES[self.polarization_code]


def _utc(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def parse_s1_product_name(name: str) -> S1ProductName:
    """Parse a Sentinel-1 product name.

    Raises:
        MetadataError: the name does not follow the Sentinel-1 convention.
    """
    match = _NAME.match(name.strip())
    if match is None:
        raise MetadataError(f"product name does not follow the Sentinel-1 naming convention: {name!r}")
    g = match.groupdict()
    try:
        start, stop = _utc(g["start"]), _utc(g["stop"])
    except ValueError as exc:
        raise MetadataError(f"invalid timestamp in product name {name!r}: {exc}") from exc
    if stop < start:
        raise MetadataError(f"product name stop time precedes start time: {name!r}")
    return S1ProductName(
        name=name.strip(), platform=g["platform"], mode=g["mode"], product_kind=g["ptype"],
        resolution=g["res"], level=g["level"], product_class=g["pclass"], polarization_code=g["pol"],
        start=start, stop=stop, absolute_orbit=int(g["abs_orbit"]), datatake_id=g["datatake"],
        is_cog=g["suffix"] is not None,
    )
