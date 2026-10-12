# data/reference/ - EVALUATION ONLY

Published damage maps (Copernicus EMS e.g. EMSR927, UNOSAT) and post-event OSM, used **only** to check results.

Using anything in this folder as a pipeline input, training label or threshold-tuning target breaks the PS data rules and leads to disqualification. Code under `src/bato/` (except `bato.evaluation`) must not read it.

Credit for Copernicus EMS data: "European Union, Copernicus Emergency Management Service data".

## EMSR927 layout (written only by `scripts/fetch_emsr927.py` and `scripts/inspect_emsr927.py`)

- `EMSR927/activation/EMSR927_activation_<UTC>.json`: the official activation record, byte-for-byte.
- `EMSR927/products/*.zip`: product zips downloaded from the record's `downloadPath`.
- `EMSR927/acquisition_manifest.json`: provenance for each file (URL, time, bytes, SHA-256, AOI, product, version, status, image times).
- `EMSR927/inspection/inspection_report.{json,md}`: product contents. This report contains no metric.

Metrics are written to `data/eval/EMSR927/` (see `docs/evaluation/emsr927_protocol.md`).
