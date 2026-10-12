"""Data acquisition.

STEP 2A (implemented): Sentinel-1 catalogue discovery on the Copernicus Data
Space Ecosystem (CDSE) OData catalogue - search, strict filtering, AOI
coverage, orbit-track grouping. Entry point: ``s1_catalogue.search_sentinel1``.

Modules: ``cdse_odata`` (query + record parsing), ``product_name`` (ESA name
parser), ``coverage`` (AOI coverage fraction), ``cdse_auth`` (tokens for
later downloads), ``http`` (stdlib transport).

STEP 2B (implemented): same-track temporal-set selection -
``temporal_selection.select_temporal_sets`` (passes, PAIRING HEURISTIC, ranked
PRE-1 + PRE-2 + POST candidates per track, PrePostPair enforcement).

STEP 2C (implemented): fetching of the selected Sentinel-1 GRD products and
Copernicus DEM tiles - ``s1_fetch`` (plan + execution + manifest),
``downloader`` (resumable, verified, atomic), ``dem`` (GLO-30 tiles),
``safe_inspect`` (archive structure checks).

NOT IMPLEMENTED YET: mosaicking, Sentinel-2, and any pixel processing
(Step 2D onwards).
"""
