# dashboard/

Map dashboard of the frozen official result (road rule v1): `app/` is a static web app (MapLibre GL and IBM Plex
vendored under `app/vendor/`, licences included). It reads only the bundle written by `scripts/build_dashboard.py`,
which verifies every hash and recomputes the summary before writing. See [docs/dashboard.md](../docs/dashboard.md).

```bash
python scripts/build_dashboard.py --manifest data/raw/manifests/<aoi>_fetch_manifest.json --title <name>
python scripts/serve_dashboard.py --open
```
