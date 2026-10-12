"""Attribution and citation text required by the official PS.

The three ATTRIBUTION strings are copied verbatim from the Track B document
("Required attribution in every submission"). The dashboard, situation report,
README and technical report must all show them.
"""

from __future__ import annotations

ATTRIBUTION_SENTINEL = "Contains modified Copernicus Sentinel data 2026."
ATTRIBUTION_COPERNICUS_DEM = (
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and "
    "Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; "
    "all rights reserved."
)
ATTRIBUTION_OSM = "© OpenStreetMap contributors."

REQUIRED_ATTRIBUTIONS: tuple[str, ...] = (
    ATTRIBUTION_SENTINEL, ATTRIBUTION_COPERNICUS_DEM, ATTRIBUTION_OSM,
)

# Training datasets: cite whichever are actually used.
CITATION_KURO_SIWO = "Bountos et al., 2024 (Kuro Siwo, NeurIPS 2024 Datasets and Benchmarks)."
CITATION_SEN1FLOODS11 = "Bonafilia et al., CVPR Workshops 2020 (Sen1Floods11)."

# Evaluation reference only (never a pipeline input).
CREDIT_COPERNICUS_EMS = "European Union, Copernicus Emergency Management Service data"
