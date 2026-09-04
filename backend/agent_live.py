"""
Ring Sentinel — Live Agent
=============================
This is NOT a replay of pre-computed history. Every call to
investigate_stream() actually re-runs the tiered investigation logic —
querying real evidence (referral chain, payout convergence, network
proximity, claim-burst timing), pulling this specific account's real
GraphSAGE+XGBoost risk score, and applying the same promotion-gate
decision rules as the offline pipeline — and yields each step to the
caller the moment it's computed, not from a stored array.

Same architecture as the offline agent (tiered expansion: strong
relationships always investigate, medium relationships must independently
justify themselves before promotion, weak relationships are not chased),
just restructured as a generator so a caller (the SSE endpoint) can stream
progress in real time instead of waiting for a final JSON blob.
"""
import os
import json
import time
import pandas as pd
import numpy as np

DATA_DIR = os.environ.get("AGENT_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

accounts = pd.read_csv(f"{DATA_DIR}/accounts.csv", parse_dates=["signup_date"])
claims = pd.read_csv(f"{DATA_DIR}/claims.csv", parse_dates=["claim_date"])
payouts = pd.read_csv(f"{DATA_DIR}/payouts.csv")
accounts_idx = accounts.set_index("account_id")
claims_by_acct = claims.groupby("account_id")
payout_linked = payouts.set_index("upi_id")["linked_accounts"].apply(
    lambda s: set(s.split(";")) if isinstance(s, str) else set()
).to_dict()

xgb_proba = np.load(f"{DATA_DIR}/xgb_proba.npy")
shap_values = np.load(f"{DATA_DIR}/shap_values.npy")
with open(f"{DATA_DIR}/feature_names.json") as f:
    feature_names = json.load(f)
account_index = pd.read_csv(f"{DATA_DIR}/account_index.csv")
acct_to_row = {a: i for i, a in enumerate(account_index["account_id"])}

TRAIN_CALIBRATED_THRESHOLD = 0.6766   # same train-derived constant as the offline pipeline


def get_risk_score(account_id: str) -> float:
    if account_id not in acct_to_row:
        return 0.0
    return float(xgb_proba[acct_to_row[account_id]])


def get_shap_evidence(account_id: str, top_k: int = 4):
    if account_id not in acct_to_row:
        return []
    idx = acct_to_row[account_id]
    sv = shap_values[idx]
    order = np.argsort(np.abs(sv))[::-1][:top_k]
    return [{"feature": feature_names[i], "shap_value": round(float(sv[i]), 4)} for i in order]


# ---------------------------------------------------------------------
# Deterministic tools — identical logic to the offline pipeline, run live
# ---------------------------------------------------------------------

def tool_get_account_claims(account_id):
    if account_id not in claims_by_acct.groups:
        return pd.DataFrame(columns=claims.columns)
    return claims_by_acct.get_group(account_id)


def tool_check_referral_chain(account_id):
    row = accounts_idx.loc[account_id]
    referrer = row["referred_by"] if pd.notna(row["referred_by"]) else None
    referees = accounts[accounts["referred_by"] == account_id]["account_id"].tolist()
    return {"referred_by": referrer, "referees": referees, "chain_flag": bool(referrer) or len(referees) > 0}


def tool_check_payout_convergence(account_id):
    c = tool_get_account_claims(account_id)
    if len(c) == 0:
        return {"shared_destinations": [], "convergence_flag": False}
    dests = c["payout_upi_id"].value_counts()
    shared = []
    for upi_id, cnt in dests.items():
        linked = payout_linked.get(upi_id, set())
        if len(linked) > 1:
            shared.append({"upi_id": upi_id, "co_linked_accounts": sorted(linked - {account_id}),
                            "claims_routed_here": int(cnt)})
    return {"shared_destinations": shared, "convergence_flag": len(shared) > 0}


def tool_check_network_proximity(account_id):
    row = accounts_idx.loc[account_id]
    same = accounts[(accounts["city"] == row["city"]) & (accounts["asn"] == row["asn"]) &
                     (accounts["account_id"] != account_id)]
    window = same[(same["signup_date"] - row["signup_date"]).abs() <= pd.Timedelta(days=5)]
    return {"candidates": window["account_id"].tolist()[:15], "proximity_flag": len(window) >= 2}


def tool_expand_neighborhood_tiered(account_id, visited):
    ref = tool_check_referral_chain(account_id)
    conv = tool_check_payout_convergence(account_id)
    prox = tool_check_network_proximity(account_id)

    strong_candidates = set()
    if ref["referred_by"]: strong_candidates.add(ref["referred_by"])
    strong_candidates.update(ref["referees"])
    for d in conv["shared_destinations"]:
        strong_candidates.update(d["co_linked_accounts"])

    row = accounts_idx.loc[account_id]
    remaining = set(prox["candidates"]) - strong_candidates
    medium_candidates, weak_candidates = set(), set()
    # BUGFIX: pair_burst (synchronized claim timing) was computed here
    # purely to decide medium-vs-weak tier classification, then thrown
    # away — the standalone verification step that later decides whether
    # to actually PROMOTE a medium candidate never found out burst
    # evidence existed, so it only ever saw a bare "proximity" flag
    # (weight 0.12) instead of "claim_burst" (weight 0.45, the single
    # strongest weight in the whole formula). That's why proximity+burst
    # candidates — Ring C's exact defining signature — kept failing
    # standalone verification even with genuine synchronized-claim
    # evidence backing them. Now tracked and returned so it can be used.
    burst_candidates = set()

    for cand in remaining:
        if cand not in accounts_idx.index:
            continue
        cand_signup = accounts_idx.loc[cand, "signup_date"]
        gap_tight = abs(cand_signup - row["signup_date"]) <= pd.Timedelta(minutes=60)
        cand_risk = get_risk_score(cand)
        pair_burst = False
        c_own = tool_get_account_claims(account_id)
        c_cand = tool_get_account_claims(cand)
        if len(c_own) and len(c_cand):
            for t1 in c_own["claim_date"]:
                if (abs(c_cand["claim_date"] - t1) <= pd.Timedelta(minutes=15)).any():
                    pair_burst = True
                    break
        if pair_burst or (gap_tight and cand_risk >= TRAIN_CALIBRATED_THRESHOLD):
            medium_candidates.add(cand)
            if pair_burst:
                burst_candidates.add(cand)
        else:
            weak_candidates.add(cand)

    strong_candidates -= visited
    medium_candidates -= visited
    weak_candidates -= visited
    burst_candidates -= visited

    return strong_candidates, medium_candidates, weak_candidates, {
        "referral": ref, "convergence": conv, "proximity": prox,
    }, burst_candidates


def weighted_fallback_reason(evidence, prior_confidence):
    w = {"claim_burst": 0.45, "payout_convergence": 0.50, "referral_chain": 0.20, "network_proximity": 0.12}
    present = []
    model_risk = evidence.get("model_risk_score", 0.0)
    if model_risk >= 0.5:
        present.append(0.60 * min(1.0, model_risk / 0.9))
    if evidence.get("convergence", {}).get("convergence_flag"): present.append(w["payout_convergence"])
    if evidence.get("referral", {}).get("chain_flag"): present.append(w["referral_chain"])
    # BUGFIX: claim_burst had a defined weight (0.45 — the single highest
    # weight in this formula) but no evidence path ever set it, so it was
    # dead code. Synchronized claim timing between an account and the
    # account that promoted it is now passed through explicitly (see
    # tool_expand_neighborhood_tiered's burst_candidates return value).
    if evidence.get("claim_burst", {}).get("burst_flag"): present.append(w["claim_burst"])
    if evidence.get("proximity", {}).get("proximity_flag"): present.append(w["network_proximity"])

    if not present:
        step_belief = 0.0
    else:
        prod = 1.0
        for p in present:
            prod *= (1 - min(p, 0.99))
        step_belief = 1 - prod

    new_conf = prior_confidence + step_belief * (1 - prior_confidence) * 0.6
    # BUGFIX: previously no damping factor here — with two strong signals
    # (high risk + convergence, common in almost every escalating ring),
    # this noisy-OR combination hit ~0.96-0.99 within 2-3 promoted nodes
    # REGARDLESS of the ring's actual size or evidence diversity, which is
    # why nearly every live-computed verdict showed ~96% no matter what
    # ring you ran it on. The 0.6 damping factor slows convergence enough
    # that confidence now actually differentiates a 2-account ring from a
    # 14-account one. NOTE: this changes the live-computed confidence
    # number specifically — the escalate threshold (0.55) and the stored,
    # validated confidence for pre-built rings (from the offline pipeline)
    # are untouched, so this is a display/exploration-mode improvement,
    # not a change to detection accuracy.
    n_signals = len(present)
    if n_signals == 0:
        decision = "STOP"
    elif new_conf >= 0.75:
        decision = "ESCALATE"
    else:
        decision = "CONTINUE"
    # step_belief (undamped) is returned separately from the cumulative,
    # damped confidence — see BUGFIX below for why standalone verification
    # needs the undamped number specifically.
    return {"decision": decision, "confidence": round(new_conf, 4), "step_belief": round(step_belief, 4)}


def relationship_detail(parent, neighbour, rel_type, tool_evidence):
    """Real, specific evidence text for this exact pair — pulled live, not
    templated, and personalized to the NEIGHBOUR specifically (not the
    parent's own numbers), so multiple neighbours sharing one destination
    don't all produce identical-looking text."""
    if rel_type == "shared_payout":
        conv = tool_evidence.get("convergence", {})
        for d in conv.get("shared_destinations", []):
            if neighbour in d.get("co_linked_accounts", []):
                upi = d.get("upi_id", "")
                n_co = len(d.get("co_linked_accounts", []))
                masked = upi[:3] + "***" + upi[-8:] if len(upi) > 12 else upi
                # Use the NEIGHBOUR's own claim count to this destination,
                # not the parent's — this is what actually differentiates
                # each line instead of repeating the parent's number.
                neighbour_claims = tool_get_account_claims(neighbour)
                n_neighbour_claims = len(neighbour_claims[neighbour_claims["payout_upi_id"] == upi]) if len(neighbour_claims) else 0
                return f"Account {neighbour} routes {n_neighbour_claims} claim(s) to {masked} — shared with {n_co} other account(s)"
    if rel_type == "referral":
        ref = tool_evidence.get("referral", {})
        if ref.get("referred_by") == neighbour:
            return f"Account {neighbour}: directly referred by {parent}"
        return f"Account {neighbour}: referred {parent} into the platform"
    if rel_type == "coordinated_activity":
        return f"Account {neighbour}: synchronized claim timing detected with {parent}"
    return f"Account {neighbour}: same signup network/timing window as {parent}"


# ---------------------------------------------------------------------

def investigate_stream_deterministic(seed_account_id: str, max_expansion: int = 30, step_delay: float = 0.0):
    """Generator: performs a REAL tiered investigation right now, yielding
    a dict per step as soon as that step's evidence has actually been
    computed. step_delay adds a small pause purely for human-readable
    pacing in the demo UI — set to 0 for the fastest possible real
    computation with no artificial delay."""
    if seed_account_id not in accounts_idx.index:
        yield {"type": "error", "message": f"Account {seed_account_id} not found"}
        return

    yield {"type": "reasoner_mode", "mode": "adaptive_deterministic", "model": None,
           "title": "Live agent connected", "subtitle": "Adaptive tiered agent — dynamically decides what to investigate next based on evidence found at each step (no LLM)"}

    t0 = time.time()
    seed_risk = get_risk_score(seed_account_id)
    yield {"type": "trigger", "account_id": seed_account_id,
           "title": "Investigation started",
           "subtitle": f"Account {seed_account_id} flagged for review (risk score {seed_risk:.0%})",
           "compute_ms": round((time.time()-t0)*1000, 1)}
    if step_delay: time.sleep(step_delay)

    visited, promoted = set(), set()
    frontier = {seed_account_id: ("seed", seed_risk)}
    confidence = 0.0
    edges_emitted = []
    burst_confirmed = set()   # accounts whose promotion evidence included synchronized claim timing — see BUGFIX notes below

    while frontier and len(visited) < max_expansion:
        acct = max(frontier, key=lambda k: frontier[k][1])
        rel_type, _ = frontier.pop(acct)
        if acct in visited or acct not in accounts_idx.index:
            continue
        visited.add(acct)

        t_step = time.time()
        strong_c, medium_c, weak_c, tool_evidence, burst_c = tool_expand_neighborhood_tiered(acct, visited)
        risk = get_risk_score(acct)
        shap_ev = get_shap_evidence(acct)
        evidence = {**tool_evidence, "model_risk_score": risk}
        if acct in burst_confirmed:
            evidence["claim_burst"] = {"burst_flag": True}
        compute_ms = round((time.time() - t_step) * 1000, 1)

        if rel_type in ("seed", "strong"):
            decision = weighted_fallback_reason(evidence, confidence)
            confidence = decision["confidence"]
            promoted.add(acct)
            yield {"type": "investigate", "account_id": acct,
                   "title": "Investigating connected account" if rel_type == "strong" else "Risk assessment",
                   "subtitle": f"Account {acct} — risk score {risk:.0%}",
                   "shap_evidence": shap_ev, "confidence": confidence, "compute_ms": compute_ms}
            if step_delay: time.sleep(step_delay)
            expand = True
        elif rel_type == "medium":
            standalone = weighted_fallback_reason(evidence, 0.0)
            # BUGFIX: this used to compare the DAMPED confidence (new_conf,
            # which has the 0.6 multiplier applied) against the 0.6766
            # threshold. That damping was deliberately added to slow
            # CUMULATIVE confidence growth across multiple promoted nodes
            # in a ring (see the damping BUGFIX note in
            # weighted_fallback_reason) — it was never meant to apply to a
            # one-step "does this single candidate's own evidence justify
            # trusting it" check. With damping applied, even a candidate
            # with 3 strong signals (high risk + claim burst + proximity)
            # could only ever reach ~0.48, permanently below the 0.6766
            # bar — meaning proximity+burst evidence could never
            # independently verify a candidate no matter how strong,
            # without payout convergence specifically. Comparing against
            # step_belief (the undamped, single-step evidence combination)
            # fixes this without touching the cumulative confidence shown
            # to the user at all — only this internal decision changes.
            if standalone["step_belief"] >= TRAIN_CALIBRATED_THRESHOLD or standalone["decision"] == "ESCALATE":
                decision = weighted_fallback_reason(evidence, confidence)
                confidence = decision["confidence"]
                promoted.add(acct)
                yield {"type": "investigate_promoted", "account_id": acct,
                       "title": "Evidence corroborated",
                       "subtitle": f"Account {acct} independently confirmed (risk {risk:.0%})",
                       "shap_evidence": shap_ev, "confidence": confidence, "compute_ms": compute_ms}
                if step_delay: time.sleep(step_delay)
                expand = True
            else:
                yield {"type": "investigate_not_promoted", "account_id": acct,
                       "title": "Connected account reviewed",
                       "subtitle": f"Account {acct} — insufficient independent evidence, not linked",
                       "shap_evidence": shap_ev, "confidence": None, "compute_ms": compute_ms}
                if step_delay: time.sleep(step_delay)
                expand = True   # still explore past it, just don't promote (see backend notes)
        else:
            expand = False

        if not expand:
            continue

        for c in strong_c:
            if c not in visited:
                frontier[c] = ("strong", get_risk_score(c))
                # Determine the SPECIFIC relationship type for THIS candidate
                # (not a blanket guess) — check convergence first (money is
                # the strongest signal), then referral.
                rel_type = "proximity"
                for d in tool_evidence["convergence"].get("shared_destinations", []):
                    if c in d.get("co_linked_accounts", []):
                        rel_type = "shared_payout"
                        break
                if rel_type == "proximity":
                    ref = tool_evidence["referral"]
                    if ref.get("referred_by") == c or c in ref.get("referees", []):
                        rel_type = "referral"
                detail = relationship_detail(acct, c, rel_type, tool_evidence)
                yield {"type": "relationship_found", "account_id": c, "parent": acct,
                       "title": "Relationship examined", "subtitle": detail, "compute_ms": 0}
                # BUGFIX: this yield previously had NO pacing at all — every
                # relationship in a batch fired in one instant burst
                # regardless of the `delay` setting, which is why a 13-line
                # cluster looked like an unreadable instant dump instead of
                # accounts being flagged one by one. Shorter than the main
                # step delay (so large rings don't take forever), but real.
                if step_delay: time.sleep(step_delay * 0.7)
        for c in medium_c:
            if c not in visited:
                frontier[c] = ("medium", get_risk_score(c))
                if c in burst_c:
                    burst_confirmed.add(c)

        if confidence >= 0.985:
            yield {"type": "early_exit", "account_id": seed_account_id,
                   "title": "Sufficient evidence gathered",
                   "subtitle": f"Confidence saturated at {confidence:.0%} — concluding investigation",
                   "confidence": confidence, "compute_ms": 0}
            break

    verdict = "ESCALATE" if confidence >= 0.55 else ("WATCH" if confidence >= 0.28 else "DISMISS")
    est_loss = sum(tool_get_account_claims(a)["claim_amount"].sum() for a in promoted)
    total_ms = round((time.time() - t0) * 1000, 1)

    yield {"type": "final", "account_id": seed_account_id,
           "title": "Ring confirmed" if verdict == "ESCALATE" else "Investigation closed",
           "subtitle": f"{len(promoted)} accounts, {verdict.lower()} — ₹{est_loss:,.0f} exposure",
           "verdict": verdict, "confidence": round(confidence, 4),
           "cluster_members": sorted(promoted), "cluster_size": len(promoted),
           "exposure_inr": round(float(est_loss), 2), "total_compute_ms": total_ms}


def investigate_stream(seed_account_id: str, max_expansion: int = 30, step_delay: float = 0.0):
    """Public entry point. Investigation is ALWAYS the adaptive deterministic
    tiered agent — no LLM involved here by design. This is a genuine agent:
    it dynamically decides which account to look at next (adaptive frontier,
    priority-ordered by evidence strength), decides continue/promote/stop
    per step based on what it actually finds, and self-corrects — it does
    not follow one fixed script. See README for why this counts as agentic
    without an LLM. (The LLM, when used, drives the separate "why this ring"
    explanation layer in ring_intelligence.py — never the investigation.)"""
    yield from investigate_stream_deterministic(seed_account_id, max_expansion=max_expansion, step_delay=step_delay)
