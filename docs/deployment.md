# Public deployment (Render)

The hosted service is the existing standard-library server, `scripts/serve_dashboard.py`. It serves:
- the offline dashboard;
- the official Trishuli bundle;
- the SITREP Copilot.

Nothing about detection, thresholds, road classification, run selection, grounding or attribution changes.

## 1. What is deployed

`data/` is Git-ignored, so a fresh clone has no results. The results are shipped as **one archive**,
`deploy/bato_official_data.tar.gz`, built by `scripts/deploy_data.py pack`.

**What it contains:**
- **Dashboard bundle:** `data/processed/dashboard/<aoi>/<candidate>/`. This is `bundle.json` plus the rendered PNG
  overlays and evidence chips. It comes with an `index.json` that lists only this bundle.
- **Official run:** the road-rule **v1** connectivity run, `data/processed/connectivity/<aoi>/<run>/`. This is its
  provenance plus every output the provenance lists.
- **Provenance files:** the stack provenance and the Step 4C provenance JSON that the run references.
- **Manifest:** `BATO_DEPLOY_MANIFEST.json`, with the SHA-256 and size of every file.

**What it never contains:**
- GeoTIFFs or other rasters;
- raw Sentinel-1 / DEM / OSM downloads;
- superseded runs or road-rule v2 runs;
- the experimental detector outputs;
- anything under `data/reference` or `data/eval` (EMSR927).

`pack` refuses any file outside `data/processed` and `data/interim`, and any type other than
`.json .geojson .jsonl .md .png`.

**Checks before packing:**
- The run must pass the full SITREP integrity chain (`load_facts`): official v1 only, every output hash and the
  published summary verified.
- The bundle's `integrity` block must name the same run and the same provenance hashes.

**Checks on the host:**
- `install` re-checks every SHA-256.
- The server's `--check-evidence` rebuilds and hash-verifies the copilot evidence at start-up.
- `/healthz` reports the result.

**Size:** `pack` prints the size of every file and refuses archives over 95 MiB (GitHub rejects files over 100 MB).

## 2. Build and check the archive (your machine, Windows PowerShell, repo root)

```powershell
python scripts/deploy_data.py pack --aoi aoi-e893707c2beb --candidate ts-80841ae62417
```

Then prove that a fresh clone works with it. Use the folder of your choice; your working copy is not touched:

```powershell
git clone https://github.com/Mandeep2807/bato-flood-access $env:TEMP\bato-fresh
cd $env:TEMP\bato-fresh
git apply <path>\bato_render_deploy.patch
Copy-Item <your repo>\deploy\bato_official_data.tar.gz deploy\
python -m pip install .
python scripts/deploy_data.py install
python scripts/deploy_data.py verify
```

`verify` starts the real server on the installed data and checks:
- `/healthz`, after the official evidence has been verified;
- the dashboard page;
- `index.json`;
- the v1 `bundle.json`;
- the deterministic English SITREP, including the run id and the attribution lines.

It removes `BATO_LLM_*` from the server it starts, so it never calls the model and never prints a key. It must end
with `VERIFIED`.

## 3. Render settings

Commit the code and `deploy/bato_official_data.tar.gz`. Then use one of these two options.

**A. Blueprint.** In Render, choose New → Blueprint, pick the repository, and Render reads `render.yaml`. It asks
for `BATO_LLM_API_KEY`; paste the key there.

**B. Manual web service** (New → Web Service):

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Build command | `pip install . && python scripts/deploy_data.py install` |
| Start command | `python scripts/serve_dashboard.py --host 0.0.0.0 --check-evidence --trust-proxy` |
| Health check path | `/healthz` |
| Instance type | Free works (0.1 CPU, 512 MB; it sleeps after 15 min idle and wakes in about a minute) |

| Environment variable | Value |
|---|---|
| `PYTHON_VERSION` | `3.12.11` |
| `BATO_LLM_PROVIDER` | `openai` |
| `BATO_LLM_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `BATO_LLM_MODEL` | the Gemini model that works for you locally, e.g. `gemini-3.5-flash-lite` |
| `BATO_LLM_API_KEY` | your key, added as a secret value in Render only |
| `BATO_COPILOT_PER_MINUTE` | `6` (per client) |
| `BATO_COPILOT_PER_DAY` | `200` (AI requests per day, all users) |
| `BATO_COPILOT_CONCURRENCY` | `2` (AI requests at once) |

`PORT` is set by Render and read automatically.

**After deploy:**
1. Open `https://<service>.onrender.com/healthz`. Expect `{"status": "ok", "detail": "official evidence verified", ...}`.
   It shows `"starting"` (HTTP 503) while the evidence is being verified, and `"error"` (HTTP 500) with the reason
   if the data is broken.
2. Open the site and ask a question in the Copilot tab.

## 4. Security and limits

- **API key:** it lives only in Render's environment. It is used only by the server's provider module and is never
  returned to the browser or logged; the server prints only provider and model.
- **CSRF protection:** POSTs must carry `X-BATO-Copilot: 1`, which other websites cannot send without a CORS
  preflight that this server never grants.
- **When limits apply:** whenever the server listens on a non-loopback address. A refused request gets HTTP 429 with
  `Retry-After`, and the panel shows "Request limit reached".
- **What the limits count:**
  - questions and the Nepali SITREP count against the daily and concurrency caps;
  - the deterministic English SITREP (no AI) is only subject to the per-client minute limit;
  - with no provider configured, AI requests are answered 503 "Copilot unavailable" before any limit is used.
- **Per-client identity:** with `--trust-proxy` the client is the first `X-Forwarded-For` address. Render forwards
  this header, but a client can forge it, so the per-client limit is best effort. The daily cap and the concurrency
  cap cannot be bypassed this way.
- **Limit state:** it is held in memory, so a restart, a redeploy or a free-tier wake-up resets it.

## 5. Updating the data

After re-running the pipeline and `scripts/build_dashboard.py`:
1. run `pack` again;
2. run `install --force` only on a deployment host, never over local results;
3. commit the new archive.

`install` refuses to overwrite a local file whose contents differ unless `--force` is given.
