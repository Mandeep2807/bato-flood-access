# tests/fixtures/real_catalogues/

REAL metadata, not imagery and not synthetic:

- `trishuli_s1.json`, `valencia_s1.json`: Step 2A CDSE catalogue search results
  (live queries run by the team; Copernicus Sentinel data catalogue metadata).
- `trishuli_sets.json`, `valencia_sets.json`: Step 2B temporal-set selections
  produced from them by `scripts/select_temporal_sets.py` (validated).

Used for offline regression tests of Step 2C fetch planning. They contain
CDSE product ids and footprints only; no credentials, no pixels. Production
code never reads this folder.
