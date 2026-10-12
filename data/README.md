# data/

Nothing in `data/` is committed (see `.gitignore`); only these README files are.

| Folder | Contents | Pipeline input? |
|---|---|---|
| `raw/` | Untouched downloads: Sentinel-1/2 and Copernicus DEM chips, pre-event OSM snapshots | Yes |
| `interim/` | Preprocessed rasters and intermediate products | Yes |
| `processed/` | Detection rasters, impact tables, graphs, evidence records | Yes (downstream stages) |
| `reference/` | **EVALUATION ONLY**: published reference maps such as Copernicus EMS EMSR927 | **Never** |
| `eval/` | **EVALUATION ONLY**: labelled evaluation sets and metric outputs | **Never** |

Pipeline code must open files through `bato.data_rules.assert_pipeline_input_allowed`,
which raises `ProhibitedInputError` for anything under `reference/` or `eval/`.
