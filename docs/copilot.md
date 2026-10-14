# SITREP Copilot (AI component)

| | |
|---|---|
| Server | `src/bato/copilot/` (`evidence.py`, `grounding.py`, `provider.py`, `service.py`, `api.py`); routes in `scripts/serve_dashboard.py` |
| UI | Dashboard rail → **Copilot** (`dashboard/app/js/ui/copilot.js`) |
| Tests | `tests/unit/test_copilot.py`, `tests/integration/test_copilot_server.py` (mocked provider, real HTTP), `tests/integration/test_copilot_real_data.py` (opt-in) |

The copilot is the project's AI component. A large language model does two things:
- it answers questions about the official result, in English or Nepali;
- it translates the SITREP into Nepali.

The model only ever sees verified BATO evidence. Deterministic checks then decide whether its text is shown.

## 1. What the model sees: official v1 evidence only

The evidence pack is built read-only, in two steps:

1. `bato.dashboard.sitrep.load_facts` runs the full SITREP integrity chain:
   - it uses the latest road-rule v1 run; a v2 run is refused;
   - every output is checked against its SHA-256;
   - the stack and Step 4C provenance hashes are checked;
   - the summary is recomputed from the layers, and the stretches and access states are cross-checked.
2. The same run directory is reopened, hash-verified again, to read the per-stretch records and the settlement layer.

The pack has three kinds of item:
- **Core items:** `run`, `dates`, `roads`, `stretches`, `settlements`, `not_assessed`, `method`, `limits`, `priorities`, `attribution`.
- **Detail items:** one per blocked stretch (`stretch:blk-*`) and one per settlement (`settlement:*`).
- **Calculations (`calc:*`):** each states its formula, e.g. the share of road length by state.

Every item names its source file and field.

EMSR927 and reference data are never read. The only EMSR927 sentence is the fixed statement that the frozen protocol
justifies, and it appears only for the evaluated run.

## 2. What is checked before an answer is shown

- **Format:** the model must reply with JSON: `supported`, `answer` and `citations`.
- **Citations:** every citation must name an item that was actually given to the model, and a supported answer must
  cite at least one item.
- **Numbers:** every number in the answer must appear in the items it cites, or in the user's question. Before
  comparing, numbers are normalised:
  - Devanagari digits become Western digits and thousands separators are removed;
  - roundings to 1–2 decimals are accepted;
  - rounding to a whole number is accepted only for values of 100 or more.
- **Language:** a Nepali answer must be written in Devanagari.
- **Failure:** the model gets one corrective retry. If the second answer also fails, its text is **withheld** and the
  verified evidence is shown instead.
- **Unsupported questions:** the model must return `supported: false`, and the panel says "Not supported by BATO
  evidence". Examples are population, casualties, forecasts and current conditions.

## 3. SITREPs

| Language | How it is made | Label in the UI |
|---|---|---|
| English | The deterministic `bato.dashboard.sitrep` template, filled from the verified facts. No model is used. | "Deterministic template · no AI" |
| Nepali | The model translates the English SITREP. It is accepted only if the numbers match the English source exactly and the three required attribution statements are still verbatim. | "AI translation · provider:model" |

If the Nepali translation fails these checks, it is withheld and the English SITREP remains the authoritative text.

## 4. Configure an LLM provider

The provider is configured only through environment variables, in the same way as `OHSOME_API_KEY`. The key stays
in the local server process:
- it is sent only to the configured endpoint;
- it is never returned to the browser, logged or written to disk.

| Variable | Meaning |
|---|---|
| `BATO_LLM_PROVIDER` | `openai` (any OpenAI-compatible `/chat/completions`: OpenAI, Groq, OpenRouter, Together, local Ollama / LM Studio) or `anthropic` |
| `BATO_LLM_MODEL` | Model name, e.g. `gpt-4o-mini`, `claude-sonnet-4-5`, `llama3.1` |
| `BATO_LLM_API_KEY` | API key. Not needed for an OpenAI-compatible server on `localhost`. |
| `BATO_LLM_BASE_URL` | Optional. Defaults to `https://api.openai.com/v1` or `https://api.anthropic.com`. Plain `http` is allowed only for localhost. |
| `BATO_LLM_TIMEOUT_S` | Optional request timeout in seconds (default 60). |
| `BATO_LLM_MAX_TOKENS` | Optional output-token limit (default 8192). Gemini counts its thinking tokens against it. |

Windows PowerShell:

```powershell
$env:BATO_LLM_PROVIDER="openai"; $env:BATO_LLM_MODEL="gpt-4o-mini"; $env:BATO_LLM_API_KEY="<your key>"
python scripts\serve_dashboard.py --open
```

macOS / Linux:

```bash
export BATO_LLM_PROVIDER=anthropic BATO_LLM_MODEL=claude-sonnet-4-5 BATO_LLM_API_KEY=<your key>
python scripts/serve_dashboard.py --open
```

Google Gemini through its OpenAI-compatible endpoint (short model id, no `models/` prefix):

```powershell
$env:BATO_LLM_PROVIDER="openai"; $env:BATO_LLM_MODEL="gemini-3.5-flash-lite"
$env:BATO_LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"; $env:BATO_LLM_API_KEY="<your key>"
```

HTTP 429, 5xx and dropped connections are retried twice with backoff. Errors shown in the panel include the
provider's own reason (for example `UNAVAILABLE: This model is currently experiencing high demand`), with the key
redacted.

A local model keeps everything offline, for example Ollama:

```bash
export BATO_LLM_PROVIDER=openai BATO_LLM_MODEL=llama3.1 BATO_LLM_BASE_URL=http://localhost:11434/v1
```

When the server starts, it prints whether the copilot is available, and why not if it isn't. Without a provider:
- the panel says **"Copilot unavailable"**;
- asking questions and the Nepali SITREP are disabled;
- the English SITREP still works, clearly labelled as not AI;
- the map, the evidence drawer and every other panel work as before.

**Security.** API requests must carry the header `X-BATO-Copilot: 1`. A browser cannot send that header to another
site without a CORS preflight, which this server never grants, so other web pages cannot spend your API quota.

## 5. Limits

- The copilot explains and translates the official result. It adds no measurement, and its quality depends on the model.
- A model can still word things badly even when every number is right. The wording rules in the prompt reduce this
  but cannot rule it out:
  - "access unverified" is not "cut off";
  - "unobservable" is not "open".
- The citations are what a reader should check.
- A cloud provider needs internet access. The offline map does not, and a local model keeps the copilot offline too.
- The evidence pack is cached per server process, so restart the server after re-running the pipeline.
