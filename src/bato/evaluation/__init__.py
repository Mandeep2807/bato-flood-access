"""EVALUATION ONLY. Comparison of the frozen BATO result with reference data (Copernicus EMS EMSR927).

Pipeline packages must never import this package; tests/unit/test_data_rules.py enforces it. Modules:

* reference      - ReferenceDataset contract
* acquisition    - EMSR927 activation record + product download with provenance (data/reference/EMSR927/)
* ems_activation - parsing the official activation record (products, versions, URLs, image times)
* ems_product    - read-only EMS zip inspection and layer loading (GeoJSON, shapefile)
* geometry       - WKT, spherical area, convex clipping, pixel-centre rasterisation on the BATO grid
* official       - read-only, hash-verified loading of the official road-rule v1 BATO result
* protocol       - the pre-registered protocol (docs/evaluation/emsr927_protocol.md), mapping checks, freeze
* metrics        - TP/FP/FN/TN, precision, recall, F1, IoU, +/-1 px tolerant variants; NOT CALCULABLE handling
* roads          - optional road-segment comparison (only with an EMS road-damage layer)
* timegap        - EMS image time vs BATO POST under each time-zone reading
* leakage        - scan of pipeline provenance/manifests for reference-data citations
* emsr927        - inspect / freeze / evaluate orchestration
"""
