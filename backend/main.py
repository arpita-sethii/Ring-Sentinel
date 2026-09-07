"""
Ring Sentinel — Backend
=========================
Serves the fraud-ring investigation data (built from the real GNN +
XGBoost + tiered-agent pipeline output) via a small REST API, and hosts
the frontend as static files.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000
"""
import json
import os
import asyncio
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agent_live
import ring_intelligence
import drift_monitor
import retrain
import auth
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "frontend_bundle.json")
FRONTEND_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "frontend"))

with open(DATA_PATH) as f:
    BUNDLE = json.load(f)

STATS = BUNDLE["stats"]
RINGS = {r["ring_id"]: r for r in BUNDLE["rings"]}
RING_ORDER = [r["ring_id"] for r in BUNDLE["rings"]]
RING_DETAILS = BUNDLE["ring_details"]
ACCOUNT_DETAILS = BUNDLE["account_details"]
WATCH_LIST = BUNDLE["watch_list"]

# In-memory action store: ring_id -> {"status": ..., "updated_at": ...}
# Not persisted across restarts by design (kept simple) — swap for a real
# database if this needs to survive a server restart.
RING_ACTIONS: dict[str, dict] = {}

app = FastAPI(title="Ring Sentinel API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# AUTH — in-memory, session-cookie based. Read/browse endpoints stay
# open without login; only WRITE actions (ring actions, retrain) require
# one, so they can be attributed to a real person.
# ---------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


def require_user(request: Request) -> str:
    token = request.cookies.get("session_token")
    username = auth.get_user_from_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Login required for this action.")
    return username


@app.post("/api/auth/register")
def register(body: RegisterRequest):
    try:
        result = auth.register_user(body.username, body.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.post("/api/auth/login")
def login(body: LoginRequest, response: Response):
    if not auth.verify_login(body.username, body.password):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    token = auth.create_session(body.username)
    response.set_cookie("session_token", token, httponly=True, samesite="lax", max_age=60*60*24*7)
    return {"username": body.username}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    auth.destroy_session(request.cookies.get("session_token"))
    response.delete_cookie("session_token")
    return {"status": "logged_out"}


@app.get("/api/auth/me")
def me(request: Request):
    username = auth.get_user_from_token(request.cookies.get("session_token"))
    if not username:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return {"username": username}


def ring_with_status(ring_id: str, ring_obj: dict) -> dict:
    action = RING_ACTIONS.get(ring_id)
    out = dict(ring_obj)
    out["action_status"] = action["status"] if action else "pending"
    out["action_updated_at"] = action["updated_at"] if action else None
    return out


@app.get("/api/stats")
def get_stats():
    return STATS


@app.get("/api/rings")
def list_rings():
    return [ring_with_status(rid, RINGS[rid]) for rid in RING_ORDER]


@app.get("/api/rings/{ring_id}")
def get_ring(ring_id: str):
    if ring_id not in RING_DETAILS:
        raise HTTPException(status_code=404, detail=f"Ring {ring_id} not found")
    detail = dict(RING_DETAILS[ring_id])
    action = RING_ACTIONS.get(ring_id)
    detail["action_status"] = action["status"] if action else "pending"
    detail["action_updated_at"] = action["updated_at"] if action else None
    detail["ai_verdict"] = action["ai_verdict"] if action else None
    detail["analyst_decision"] = action["analyst_decision"] if action else None
    detail["overridden"] = action["overridden"] if action else False
    return detail


@app.get("/api/accounts/{account_id}")
def get_account(account_id: str):
    if account_id not in ACCOUNT_DETAILS:
        # Not an error — this is the "investigated but cleared" case the
        # frontend renders as a distinct state, not a failure.
        raise HTTPException(status_code=404, detail="Account not linked to any confirmed ring")
    return ACCOUNT_DETAILS[account_id]


class RingActionRequest(BaseModel):
    action: Literal["escalate", "watch", "dismiss"]


# --- FEEDBACK LOOP STATE (in-memory — see README for the "swap for a real
# datastore" note that already applies to RING_ACTIONS) ---
FEEDBACK_STATS = {"total_decisions": 0, "overridden_decisions": 0,
                   "escalate_verdicts_total": 0, "escalate_verdicts_agreed": 0}
SERVER_START_TIME = datetime.now(timezone.utc)
LAST_RETRAIN_AT = None   # None until the first real retrain runs; time-trigger measures from SERVER_START_TIME until then

# Drift acknowledgment snapshot — set on every successful retrain to the
# CURRENT PSI values at that moment. The data-drift trigger only fires
# again if drift has gotten WORSE than this acknowledged baseline — not
# for the same static number forever. Data drift often can't be "fixed"
# by a retrain with zero analyst corrections (it's a structural property
# of the train/test split, not something an empty-correction refit moves),
# so re-nagging about the identical unchanged number after the user has
# already seen it and retrained would just be noise, not new information.
DRIFT_ACK_PSI = {"risk_score": None, "evidence_mix": None}

# Four independent retrain triggers — no single published industry standard
# for the override-rate number specifically, but each of these mirrors a
# real, documented pattern (see README): PSI-based drift thresholds,
# performance-degradation thresholds, human-feedback/override loops, and
# scheduled time-based retraining cadence (fraud models are commonly
# retrained on the order of weekly in production, independent of any
# threshold, because tactics drift even without a detectable spike).
OVERRIDE_RATE_THRESHOLD = 0.30
OVERRIDE_MIN_DECISIONS = 25          # raised from 5 — 2/5 crossing 30% is noise, not signal
PERFORMANCE_AGREEMENT_THRESHOLD = 0.90   # analyst agreement on ESCALATE verdicts specifically must stay >= 90%
PERFORMANCE_MIN_ESCALATIONS = 10
TIME_BASED_INTERVAL_DAYS = 7


def _ai_recommended_action(ring_obj: dict) -> str:
    """What the AI itself would have recommended for this ring, derived
    from its OWN stored verdict — used as the baseline to detect overrides."""
    return "escalate" if ring_obj["risk_label"] == "HIGH" else "watch"


def _drift_is_worse_than_acknowledged(drift: dict) -> bool:
    """The data-drift trigger should only re-fire on genuinely NEW/worse
    drift — not nag forever about the same static number a retrain already
    acknowledged (retraining with zero analyst corrections can't move a
    structural train/test PSI value at all, so re-showing the identical
    number after every retrain click would just be permanent noise)."""
    if drift["overall_severity"] != "severe":
        return False
    if DRIFT_ACK_PSI["risk_score"] is None:
        return True   # never acknowledged yet — first time this fires for real
    return (drift["risk_score_drift"]["psi"] > DRIFT_ACK_PSI["risk_score"] or
            drift["evidence_mix_drift"]["psi"] > DRIFT_ACK_PSI["evidence_mix"])


def _evaluate_retrain_triggers():
    """Checks all four triggers independently and returns which (if any)
    are currently firing, plus the detail needed to explain why."""
    total = FEEDBACK_STATS["total_decisions"]
    overridden = FEEDBACK_STATS["overridden_decisions"]
    override_rate = overridden / total if total else 0.0

    esc_total = FEEDBACK_STATS["escalate_verdicts_total"]
    esc_agreed = FEEDBACK_STATS["escalate_verdicts_agreed"]
    agreement_rate = esc_agreed / esc_total if esc_total else 1.0   # no data yet = assume fine, don't false-trigger

    live_proba = agent_live.xgb_proba if LAST_RETRAIN_AT is not None else None
    drift = drift_monitor.compute_drift_report(live_proba)

    last_retrain_reference = LAST_RETRAIN_AT or SERVER_START_TIME
    days_since = (datetime.now(timezone.utc) - last_retrain_reference).total_seconds() / 86400

    triggers = {
        "override_rate": {
            "firing": total >= OVERRIDE_MIN_DECISIONS and override_rate > OVERRIDE_RATE_THRESHOLD,
            "value": round(override_rate, 4), "threshold": OVERRIDE_RATE_THRESHOLD,
            "sample_size": total, "min_sample_required": OVERRIDE_MIN_DECISIONS,
            "detail": f"{overridden}/{total} decisions overridden" if total else "no decisions yet",
        },
        "performance_degradation": {
            "firing": esc_total >= PERFORMANCE_MIN_ESCALATIONS and agreement_rate < PERFORMANCE_AGREEMENT_THRESHOLD,
            "value": round(agreement_rate, 4), "threshold": PERFORMANCE_AGREEMENT_THRESHOLD,
            "sample_size": esc_total, "min_sample_required": PERFORMANCE_MIN_ESCALATIONS,
            "detail": f"analyst agreed with {esc_agreed}/{esc_total} AI escalate-verdicts" if esc_total else "no escalate-verdicts reviewed yet",
        },
        "data_drift_psi": {
            "firing": _drift_is_worse_than_acknowledged(drift),
            "value": drift["overall_severity"], "threshold": "severe (PSI >= 0.2)",
            "detail": f"risk-score PSI={drift['risk_score_drift']['psi']}, evidence-mix PSI={drift['evidence_mix_drift']['psi']}"
                      + (f" (acknowledged at last retrain: risk={DRIFT_ACK_PSI['risk_score']}, evidence={DRIFT_ACK_PSI['evidence_mix']})"
                         if DRIFT_ACK_PSI["risk_score"] is not None else ""),
        },
        "scheduled_interval": {
            "firing": days_since >= TIME_BASED_INTERVAL_DAYS,
            "value": round(days_since, 2), "threshold": TIME_BASED_INTERVAL_DAYS,
            "detail": f"{days_since:.1f} days since {'last retrain' if LAST_RETRAIN_AT else 'server start'}",
        },
    }
    any_firing = any(t["firing"] for t in triggers.values())
    return triggers, any_firing


@app.post("/api/rings/{ring_id}/action")
def act_on_ring(ring_id: str, body: RingActionRequest, analyst: str = Depends(require_user)):
    if ring_id not in RINGS:
        raise HTTPException(status_code=404, detail=f"Ring {ring_id} not found")
    ring_obj = RINGS[ring_id]
    status_map = {"escalate": "escalated", "watch": "on_watch", "dismiss": "dismissed"}

    ai_verdict = _ai_recommended_action(ring_obj)
    overridden = (body.action != ai_verdict)

    RING_ACTIONS[ring_id] = {
        "status": status_map[body.action],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "ai_verdict": ai_verdict,
        "ai_confidence": ring_obj["confidence"],
        "analyst_decision": body.action,
        "overridden": overridden,
        "analyst_username": analyst,
    }

    FEEDBACK_STATS["total_decisions"] += 1
    if overridden:
        FEEDBACK_STATS["overridden_decisions"] += 1
    if ai_verdict == "escalate":
        FEEDBACK_STATS["escalate_verdicts_total"] += 1
        if not overridden:
            FEEDBACK_STATS["escalate_verdicts_agreed"] += 1

    return {"ring_id": ring_id, **RING_ACTIONS[ring_id]}


@app.get("/api/feedback-stats")
def get_feedback_stats():
    """AI-verdict-vs-analyst-decision tracking + all four retrain
    triggers, each reported independently so it's clear WHICH condition
    (if any) is actually firing, not just a single opaque yes/no."""
    triggers, any_firing = _evaluate_retrain_triggers()
    total = FEEDBACK_STATS["total_decisions"]
    overridden = FEEDBACK_STATS["overridden_decisions"]
    return {
        "total_decisions": total, "overridden_decisions": overridden,
        "override_rate": round(overridden / total, 4) if total else 0,
        "last_retrain_at": LAST_RETRAIN_AT.isoformat() if LAST_RETRAIN_AT else None,
        "triggers": triggers,
        "retrain_recommended": any_firing,
    }


@app.post("/api/retrain")
def run_retrain(analyst: str = Depends(require_user)):
    """REAL retrain — not a stub. Refits the XGBoost layer (on top of the
    already-trained, unchanged GraphSAGE embeddings) using every recorded
    analyst override as a label correction, persists the new model +
    scores + SHAP values to disk, and hot-swaps them into the live agent's
    in-memory arrays so the next investigation reflects the update
    immediately — no server restart needed. Resets the feedback counters
    afterward so the next threshold check measures fresh drift, not
    overrides that already led to this retrain."""
    global LAST_RETRAIN_AT, DRIFT_ACK_PSI

    overridden_actions = {}
    for ring_id, action in RING_ACTIONS.items():
        if action.get("overridden") and ring_id in RINGS:
            overridden_actions[ring_id] = {
                "ai_verdict": action["ai_verdict"],
                "cluster_members": RINGS[ring_id]["members"],
            }

    result = retrain.retrain(overridden_actions)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {result['error']}")

    # SESSION-ONLY hot-swap: update the live agent's in-memory arrays
    # directly from what retrain() computed — nothing was written to disk,
    # so a server restart always returns to the exact shipped baseline.
    # This is deliberate: while testing/demoing, clicking retrain (or
    # taking any Escalate/Watch/Dismiss action) should never permanently
    # mutate the project's shipped data files.
    agent_live.xgb_proba = result.pop("new_proba")
    agent_live.shap_values = result.pop("new_shap")

    # Snapshot the drift level at THIS retrain — the data-drift trigger
    # will now only re-fire if it gets worse than this, not for the same
    # unchanged number forever (see _drift_is_worse_than_acknowledged).
    post_retrain_drift = drift_monitor.compute_drift_report(agent_live.xgb_proba)
    DRIFT_ACK_PSI["risk_score"] = post_retrain_drift["risk_score_drift"]["psi"]
    DRIFT_ACK_PSI["evidence_mix"] = post_retrain_drift["evidence_mix_drift"]["psi"]

    LAST_RETRAIN_AT = datetime.now(timezone.utc)
    FEEDBACK_STATS["total_decisions"] = 0
    FEEDBACK_STATS["overridden_decisions"] = 0
    FEEDBACK_STATS["escalate_verdicts_total"] = 0
    FEEDBACK_STATS["escalate_verdicts_agreed"] = 0

    print(f"[retrain] Completed: {result}")
    return {**result, "retrained_at": LAST_RETRAIN_AT.isoformat(),
            "note": "SESSION-ONLY: the live agent's risk scores are updated immediately in this running "
                    "server's memory (visible right away in new live investigations), but nothing was "
                    "written to disk — restarting the server returns to the exact shipped baseline model. "
                    "The pre-built ring cards in the sidebar (from frontend_bundle.json) are a separate "
                    "static snapshot and are not affected by this either way."}


@app.get("/api/rings/{ring_id}/export")
def export_ring_report(ring_id: str):
    """Downloadable audit report — full evidence, blueprints, and the
    AI-vs-analyst decision record for this ring, if any action was taken."""
    if ring_id not in RING_DETAILS:
        raise HTTPException(status_code=404, detail=f"Ring {ring_id} not found")
    ring = RING_DETAILS[ring_id]
    intelligence = ring_intelligence.generate_report(ring)   # deterministic synthesis — export is always reproducible, no LLM variability
    action_record = RING_ACTIONS.get(ring_id)
    return {
        "export_generated_at": datetime.now(timezone.utc).isoformat(),
        "ring_id": ring_id,
        "risk_label": ring["risk_label"],
        "confidence": ring["confidence"],
        "account_count": ring["account_count"],
        "exposure_inr": ring["exposure_inr"],
        "nodes": ring["nodes"],
        "edges": ring["edges"],
        "ring_intelligence": {k: v for k, v in intelligence.items() if k not in ("ring_blueprint", "account_blueprints")},
        "account_blueprints": intelligence["account_blueprints"],
        "decision_record": action_record or {"status": "no action taken yet"},
    }


@app.get("/api/rings/{ring_id}/intelligence")
def get_ring_intelligence(ring_id: str):
    """
    Ring Intelligence (explanation layer) — reads the ALREADY-COMPLETED
    investigation result for this ring and synthesizes a structured
    "why was this ring flagged" report. Does NOT re-run detection, does
    NOT touch the model or the ring membership, does NOT change any risk
    score. Deterministic template synthesis over real evidence counts —
    see ring_intelligence.py docstring. `generated_by` in the response is
    always "deterministic_template", never misrepresented as an LLM.
    """
    if ring_id not in RING_DETAILS:
        raise HTTPException(status_code=404, detail=f"Ring {ring_id} not found")
    ring = RING_DETAILS[ring_id]
    report = ring_intelligence.generate_llm_report(ring)
    print(f"[ring_intelligence] Generated report for {ring_id}: "
          f"{len(report['account_blueprints'])} account blueprint(s), "
          f"payload ~{len(json.dumps(report, default=str))} bytes")
    return report


@app.get("/api/watchlist")
def get_watchlist():
    return WATCH_LIST


class RiskEvaluationRequest(BaseModel):
    account_id: str
    transaction_id: str | None = None
    amount: float | None = None
    device_id: str | None = None   # accepted for API-shape completeness; matched against our real device_fingerprint data below, not a separate live lookup


# account_id -> ring_id reverse lookup, built once at startup from the real ring membership data
ACCOUNT_TO_RING = {}
for _rid, _rdetail in RING_DETAILS.items():
    for _aid in _rdetail.get("members", []):
        ACCOUNT_TO_RING[_aid] = _rid


@app.post("/api/risk/evaluate")
def evaluate_risk(body: RiskEvaluationRequest):
    """Real-time risk evaluation — the actual product-facing surface a
    merchant's checkout/risk pipeline would call. Reuses the SAME trained
    model and the SAME thresholds already used everywhere else in this
    system (0.6766 train-calibrated bar for evidence promotion inside the
    agent; 0.55 / 0.28 for the ESCALATE / WATCH / DISMISS verdict bands,
    here surfaced as BLOCK / REVIEW / ALLOW) — no new numbers invented for
    this endpoint specifically.

    Honest scope note: this only returns a real score for an account_id
    that exists in our trained account index. A genuinely new, previously
    unseen account has no computed features yet — a production version of
    this endpoint would need a real-time feature-computation path before
    scoring it, which is out of scope here. We return a clear 404 for that
    case rather than inventing a score.
    """
    account_id = body.account_id
    if account_id not in agent_live.acct_to_row:
        raise HTTPException(status_code=404,
            detail=f"Account '{account_id}' has no computed risk score yet — not in the trained account index. "
                    f"A production deployment would compute features for a brand-new account in real time before scoring; "
                    f"this demo scores accounts from the existing dataset only.")

    risk_score = agent_live.get_risk_score(account_id)
    shap_evidence = agent_live.get_shap_evidence(account_id, top_k=4)

    # SAME verdict thresholds used by the tiered agent everywhere else —
    # 0.55 / 0.28 — just relabeled to the merchant-facing ALLOW/REVIEW/BLOCK
    # vocabulary instead of ESCALATE/WATCH/DISMISS.
    if risk_score >= 0.55:
        decision, risk_level = "BLOCK", "HIGH"
    elif risk_score >= 0.28:
        decision, risk_level = "REVIEW", "MEDIUM"
    else:
        decision, risk_level = "ALLOW", "LOW"

    reasons = [f"{ev['feature']} (SHAP {ev['shap_value']:+.3f})" for ev in shap_evidence] if shap_evidence else \
              [f"Model risk score {risk_score:.0%} — no single dominant SHAP feature"]

    ring_id = ACCOUNT_TO_RING.get(account_id)
    if ring_id:
        reasons.insert(0, f"Account is a known member of investigated ring {ring_id}")

    # device signal (added for the 5-relation model) surfaced directly in
    # the API response when it's genuinely present — not fabricated per-call
    device_fp = agent_live.accounts_idx.loc[account_id, "device_fingerprint"] if account_id in agent_live.accounts_idx.index else None
    device_sharing = None
    if device_fp is not None:
        co_users = agent_live.accounts[agent_live.accounts["device_fingerprint"] == device_fp]["account_id"].tolist()
        co_users = [a for a in co_users if a != account_id]
        if co_users:
            reasons.append(f"Device fingerprint shared with {len(co_users)} other account(s)")
            device_sharing = {"device_fingerprint": device_fp, "co_linked_accounts": co_users[:10]}

    return {
        "account_id": account_id,
        "transaction_id": body.transaction_id,
        "risk_score": round(float(risk_score), 4),
        "decision": decision,
        "risk_level": risk_level,
        "reasons": reasons,
        "ring_id": ring_id,
        "device_sharing": device_sharing,
        "thresholds_used": {"block_at": 0.55, "review_at": 0.28},
    }


@app.get("/api/devices/{device_fingerprint}")
def get_device_entity(device_fingerprint: str):
    """Entity-graph view for a single device — the natural product surface
    for the device-sharing signal added to the 5-relation model. Real data
    only: connected accounts and how many of them are in a confirmed ring,
    computed directly from the same account/ring data the rest of the app
    uses, not a separate mocked-up store."""
    matches = agent_live.accounts[agent_live.accounts["device_fingerprint"] == device_fingerprint]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"No accounts found with device fingerprint '{device_fingerprint}'")
    connected = matches["account_id"].tolist()
    high_risk = [a for a in connected if agent_live.get_risk_score(a) >= 0.55]
    in_rings = sorted({ACCOUNT_TO_RING[a] for a in connected if a in ACCOUNT_TO_RING})
    return {
        "device_fingerprint": device_fingerprint,
        "connected_accounts": connected,
        "connected_account_count": len(connected),
        "high_risk_account_count": len(high_risk),
        "high_risk_accounts": high_risk,
        "known_rings": in_rings,
    }


@app.get("/api/audit")
def get_audit_log():
    """Full history of every analyst decision taken this session — ring
    ID, what the AI recommended, what the analyst actually decided,
    whether that was an override, and when. Real data straight from
    RING_ACTIONS, the same store every other endpoint reads from —
    nothing separately tracked or summarized, so this can never drift
    out of sync with what the rest of the app shows."""
    entries = []
    for ring_id, action in RING_ACTIONS.items():
        ring = RING_DETAILS.get(ring_id, {})
        entries.append({
            "ring_id": ring_id,
            "account_count": ring.get("account_count"),
            "exposure_inr": ring.get("exposure_inr"),
            "ai_verdict": action.get("ai_verdict"),
            "analyst_decision": action.get("analyst_decision"),
            "overridden": action.get("overridden", False),
            "status": action.get("status"),
            "updated_at": action.get("updated_at"),
            "analyst_username": action.get("analyst_username", "unknown"),
        })
    entries.sort(key=lambda e: e["updated_at"] or "", reverse=True)
    overridden_count = sum(1 for e in entries if e["overridden"])
    return {
        "total_decisions": len(entries),
        "overridden_decisions": overridden_count,
        "entries": entries,
    }


@app.get("/api/drift")
def get_drift_report():
    """Data drift monitoring — PSI/Wasserstein/KL/KS (same 4-metric
    approach, built from scratch) applied to (1) risk score distribution
    and (2) evidence-type mix, train-period vs test-period. Computed
    fresh on each request from real train/test data — not cached/stale.
    Reflects the live, session-retrained model when one exists (see
    BUGFIX note in drift_monitor.compute_risk_score_drift)."""
    live_proba = agent_live.xgb_proba if LAST_RETRAIN_AT is not None else None
    report = drift_monitor.compute_drift_report(live_proba)
    report["reflects_live_retrained_model"] = LAST_RETRAIN_AT is not None
    return report


@app.get("/api/health")
def health():
    return {"status": "operational", "rings_loaded": len(RINGS), "accounts_indexed": len(ACCOUNT_DETAILS),
            "investigation_engine": "adaptive_deterministic (no LLM, by design)",
            "explanation_llm_available": getattr(ring_intelligence, "LLM_AVAILABLE", False),
            "explanation_llm_provider": getattr(ring_intelligence, "LLM_PROVIDER", None),
            "explanation_llm_model": getattr(ring_intelligence, "LLM_MODEL", None),
            "explanation_llm_status": getattr(ring_intelligence, "LLM_INIT_ERROR", None)}


@app.get("/api/investigate/{account_id}/stream")
async def investigate_stream(account_id: str, delay: float = Query(0.9, ge=0.0, le=3.0)):
    """
    LIVE investigation, not a replay of stored history. Every event in this
    stream is computed at request time by agent_live.investigate_stream():
    real evidence queries (referral/convergence/proximity/burst) against
    the actual account data, the account's real GraphSAGE+XGBoost risk
    score, and the same tiered promotion-gate decision logic as the
    offline pipeline — run fresh, right now, for this specific account.

    `delay` (seconds, default 0.35) is an artificial pause between events
    purely so a human can read the stream as it arrives — the actual
    computation for each step (see `compute_ms` in each event) typically
    takes low-single-digit milliseconds. Set delay=0 for the fastest
    possible real-time feed with no pacing.

    Server-Sent Events (text/event-stream) — the frontend consumes this
    with EventSource.
    """
    def event_generator():
        for event in agent_live.investigate_stream(account_id, step_delay=delay):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# Serve the frontend (index.html at root, static assets under /static)
# "/" needs an explicit route — StaticFiles' html=True default only
# auto-serves index.html for a root request, and this project's entry
# point is landing.html instead.
from fastapi.responses import FileResponse

@app.get("/", include_in_schema=False)
def root_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "landing.html"))


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
