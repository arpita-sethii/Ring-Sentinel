# Ring Sentinel — Full Stack

A fraud-ring investigation console: a FastAPI backend serving real detection
output (from a GraphSAGE + XGBoost + tiered-agent pipeline) and a frontend
that talks to it live over HTTP — including a genuinely **live** agent
investigation, not an animation of stored history.

## What's inside

```
ring_sentinel/
├── backend/
│   ├── main.py              FastAPI app — all API routes + serves the frontend
│   ├── agent_live.py        The real tiered agent, runnable on demand (generator, streams as it computes)
│   ├── requirements.txt
│   └── data/
│       ├── frontend_bundle.json          Pre-built ring/account views (148 rings, 19,167 accounts)
│       ├── accounts.csv, claims.csv,
│       │   payouts.csv                   Raw data the LIVE agent queries in real time
│       └── xgb_proba.npy, shap_values.npy,
│           feature_names.json,
│           account_index.csv             The trained model's risk scores + SHAP explanations
├── frontend/
│   ├── index.html           The UI — fetches everything from the API, no embedded data
│   └── static/
│       └── vis-network.min.js     Graph visualization library (vendored, no CDN needed)
├── model_training/           The GraphSAGE + XGBoost pipeline that PRODUCED backend/data/'s
│   │                         model artifacts — see model_training/README.md
│   ├── generate_dataset_v7.py
│   ├── stage1_2_gnn_xgboost_v8.py
│   └── kaggle_derived/        Small real-data samples (from Kaggle IEEE-CIS) the generator needs
└── README.md
```

## Run it

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000**. The backend serves both the API and
the frontend from the same origin — nothing else to configure.


## Agent architecture — the two layers, and why they use different reasoning

**Two separate layers, deliberately using different reasoning approaches:**

1. **Investigation** (`agent_live.py`) — decides which accounts belong to
   a ring. **Always deterministic, never an LLM.** This is still a genuine
   agent: it dynamically decides which account to look at next (adaptive,
   priority-ordered by evidence strength — not a fixed script), decides
   continue/promote/stop per step based on what it actually finds, and
   self-corrects. Agentic behavior comes from *making its own decisions*,
   not from which reasoning engine is plugged in. No LLM is involved here
   by design — this is the layer that actually determines fraud, and it
   stays on the same rule-based, validated, zero-dependency logic that
   was proven against the offline pipeline's 148-ring benchmark.

2. **Explanation — "Ring Intelligence"** (`ring_intelligence.py`) —
   answers "why was this ring flagged," AFTER investigation has already
   decided. **LLM-driven when available** (a single-shot call to a local
   Ollama model — not the multi-turn tool-calling loop the investigation
   used to use, which is what caused the earlier reliability problems),
   with a deterministic template fallback when it isn't. This layer never
   changes a risk score or ring membership — `recommended_action` is
   always forced to match the investigation's own verdict, even if the
   LLM's response somehow disagreed.

Both layers are visible in `/api/health`:
```json
{
  "investigation_engine": "adaptive_deterministic (no LLM, by design)",
  "explanation_llm_available": true/false,
  "explanation_llm_status": "..."
}
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Server status, counts loaded |
| `/api/stats` | GET | Top-level dashboard stats |
| `/api/rings` | GET | All investigations (list view) |
| `/api/rings/{ring_id}` | GET | Full detail: graph nodes/edges, historical activity summary, timeline |
| `/api/accounts/{account_id}` | GET | Account risk score + evidence. Returns 404 for accounts investigated but not confirmed — the frontend renders that as a distinct "cleared" state, not an error |
| `/api/rings/{ring_id}/action` | POST | Body: `{"action": "escalate" \| "watch" \| "dismiss"}`. Persists server-side (in-memory) |
| **`/api/investigate/{account_id}/stream`** | GET | **Live** — Server-Sent Events stream. Runs the real agent right now. Optional `?delay=` seconds between events (default 0.35, use 0 for no pacing) |
| `/api/watchlist` | GET | Accounts queued for re-review |

Swagger docs at `/docs` once the server is running.

## "Run Live Investigation" is genuinely live — here's what that means

Clicking the button in the UI opens a Server-Sent Events connection to
`/api/investigate/{account_id}/stream`. The backend (`agent_live.py`) then,
at that moment:

1. Queries real evidence for the seed account — referral chain, payout
   convergence, network proximity, pairwise claim-burst timing — against
   the actual account/claims/payout data in `backend/data/*.csv`.
2. Pulls that account's real GraphSAGE+XGBoost risk score and SHAP
   explanation from the trained model's saved output.
3. Applies the same tiered promotion-gate logic as the offline pipeline
   (strong relationships expand immediately; medium relationships must
   independently justify themselves before being promoted; weak
   relationships aren't chased) to decide who else to investigate.
4. Yields each step to the stream **the instant that step is computed** —
   every event carries a `compute_ms` field showing its actual computation
   time, and the final verdict reports the total wall-clock time for the
   whole investigation.

This is why re-running it against the same seed account reliably
reproduces the same cluster: it's the same logic and the same underlying
data as what originally produced the stored rings, just executed on
demand instead of ahead of time. The `delay` parameter is purely a
readability pause between already-computed events — it never affects what
gets computed, only how fast the results are sent to the browser.

## What's real vs. what's a demo simplification

**Real:**
- All 148 rings, risk scores, and evidence come directly from actual
  pipeline output — nothing is fabricated for the UI.
- The frontend has zero embedded data. Every number on screen came from an
  HTTP request you can watch in the browser's network tab.
- Escalate / Watch / Dismiss actions genuinely POST to the backend and
  persist — refresh the page and the status is still there.
- The live investigation (above) is real computation, not a replay —
  in Ollama mode, the LLM's tool-call decisions are genuinely its own,
  not scripted; in fallback mode, the rule-based promotion logic runs live.

**Demo simplification:**
- Action state lives in memory (a Python dict), not a database — it
  resets when the server restarts. Swap `RING_ACTIONS` in `main.py` for a
  real datastore (Postgres, SQLite, Redis) if this needs to survive restarts.
- The data the live agent queries is a snapshot from one pipeline run, not
  a live-updating feed. The agent's *reasoning* is live; the *data* it
  reasons over is fixed until you replace the CSVs in `backend/data/`.
- No authentication — this is a demo/judge-facing build, not a production
  deployment.

## Ring Intelligence — the explanation layer, in detail

`backend/ring_intelligence.py`, exposed at `GET /api/rings/{ring_id}/intelligence`.

**What it does:** reads the already-completed investigation result for a
ring (evidence, relationships, risk scores — all already computed by the
frozen GraphSAGE+XGBoost+v11 pipeline, untouched by this layer) and
synthesizes a structured "why was this ring flagged" report: primary
pattern, strongest/supporting evidence, per-account role assessment,
weakest member (only when the evidence genuinely supports it), and a
recommended action that's always forced to match the existing
investigation's own verdict — this layer cannot override it, even if an
LLM's response somehow tried to.

**Two ways it can be generated, both fully disclosed in the response:**
- `generated_by: "ollama_llm:<model>"` — a real local LLM call synthesized
  the report (single-shot, no tools, no multi-turn loop).
- `generated_by: "deterministic_template"` — Ollama wasn't available, so
  template rules over the same real evidence counts produced it instead.

Every response also carries `llm_attempted` and `llm_status` so it's never
ambiguous which path ran or why. The UI's source-tag line at the bottom of
the report reflects this dynamically — it never hardcodes one answer.

In the UI: click **🧠 Ring Intelligence** on any ring to open the report.

## Model monitoring, feedback loop, and audit export

Three additions on top of the core detection + investigation system:

### 1. Data drift monitoring (`backend/drift_monitor.py`, `GET /api/drift`)

Same 4-metric approach (PSI, Wasserstein distance, KL divergence, KS test —
all implemented from scratch) as an earlier drift-detection project,
applied here to a fraud-specific question: **have ring operators shifted
tactics between the train period and the test period?**

Two things are monitored, both computed fresh from real data on every
request (not cached/stale):
- **Risk score distribution drift** — has the model's score distribution
  shifted between train-period and test-period accounts?
- **Evidence-type mix drift** — of confirmed rings, what fraction were
  caught via shared payout vs. referral vs. claim burst vs. proximity, in
  the train period vs. the test period? If ring operators start avoiding
  the most-monitored signal (payout convergence) and lean on weaker
  signals instead, this shows up directly here — a more actionable,
  fraud-specific signal than generic score drift alone.

PSI severity follows the same convention as before: <0.1 no significant
drift, 0.1-0.2 moderate (monitor), ≥0.2 severe (retrain recommended). The
"Model Health" badge in the top bar reflects this live; click it for the
full metric breakdown.

**Honest note on the current numbers:** this benchmark's train/test split
wasn't built with a deliberate tactic shift between the two periods, so
the real PSI values come back low ("no drift") — which is the *correct*
answer for this data, not a bug. Interestingly, the KS test still flags
statistical significance (large sample size makes it very sensitive) even
though PSI shows no *practical* drift — this exact tension is why multiple
metrics are used together rather than relying on one.

### 2. AI-vs-analyst feedback loop + real retrain (`GET /api/feedback-stats`, `POST /api/retrain`)

Every `Escalate` / `Watch` / `Dismiss` action records **both** the AI's
own original recommendation (derived from the ring's stored `risk_label`)
**and** the analyst's actual decision, plus whether they disagreed
(`overridden: true/false`). Shown directly in the ring header once a
decision has been made ("AI RECOMMENDED: ESCALATE / ANALYST DECIDED:
DISMISS / ⚠ OVERRIDE").

**Four independent retrain triggers**, each checked and reported
separately (no single one is treated as "the" answer — see
`_evaluate_retrain_triggers()` in `main.py`):

1. **Analyst override rate** > 30%, requiring at least **25 decisions**
   (raised from an initial 5 — 2 overrides out of 5 crossing 30% is noise,
   not a pattern worth acting on).
2. **Performance degradation** — analyst agreement with the AI's own
   ESCALATE verdicts specifically drops below 90% (minimum 10 escalate
   reviews), mirroring the "model catching <90% of fraud" style threshold
   used in published fraud-detection retraining criteria.
3. **Data drift (PSI)** — the drift monitor above reports `severe`.
4. **Scheduled interval** — 7+ days since the last retrain (or since
   server start, if none has run yet), independent of any threshold —
   fraud models are commonly retrained on a roughly weekly cadence in
   production regardless of whether a specific spike was detected,
   because tactics drift even without a clean statistical signal.

**Clicking "Retrain Model" runs a REAL retrain, not a stub or a log
message.** It refits the XGBoost layer (on top of the already-trained,
unchanged GraphSAGE embeddings — retraining the GNN itself needs
torch/torch_geometric and takes minutes, which isn't something to trigger
unattended from a button click) using every recorded analyst override as
a label correction: if the AI escalated a ring the analyst dismissed,
that ring's members get corrected toward "legitimate" for the refit; if
the AI only watched a ring the analyst escalated, they get corrected
toward "fraud." Confirmed on real test runs: 9 overridden rings → 71
label corrections → 147 accounts' verdicts genuinely shifted, in ~3
seconds.

**SESSION-ONLY, by design.** The retrain updates the live agent's
in-memory risk scores immediately (visible in the next live investigation
you run), but **nothing is written to disk** — verified via md5sum before
and after a full retrain run, the underlying `.npy`/`.pkl` files are
byte-for-byte unchanged. Restarting the server always returns to the
exact shipped baseline model. This is intentional: testing/demoing the
retrain button shouldn't permanently corrupt the project's baseline
artifacts. The pre-built ring cards in the sidebar (from
`frontend_bundle.json`) are a separate static snapshot, unaffected either
way — only live investigations reflect a session's retrain.

### 3. Audit report export (`GET /api/rings/{ring_id}/export`, "⬇ Export" button)

Downloads a complete JSON audit report for any ring: risk scores, full
evidence/relationship graph, the deterministic Ring Intelligence
synthesis, per-account blueprints, and the AI-vs-analyst decision record
if one exists. Always uses the deterministic (not LLM) explanation
synthesis specifically so an exported report is reproducible — re-running
the export produces the same content, not something that varies run to
run.

## Regenerating the data

To refresh what the backend serves after a new pipeline run: replace
`backend/data/frontend_bundle.json` (pre-built ring views) and the raw
`accounts.csv` / `claims.csv` / `payouts.csv` / model output files (used by
the live agent) with the new run's outputs, then restart the server.
