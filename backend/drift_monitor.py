"""
Ring Sentinel — Drift Monitor
================================
Same 4-metric approach as the original ETTh1 LSTM drift-detection project
(PSI, Wasserstein distance, KL divergence, KS test — all implemented from
scratch, not via a library like Evidently), applied here to a genuinely
relevant fraud-specific question: have ring operators shifted tactics
between the train period and the test period?

Two things are monitored:
1. RISK SCORE DRIFT — has the model's score distribution shifted between
   train-period and test-period accounts? A shift here suggests the
   population the model sees now looks meaningfully different from what
   it was trained on.
2. EVIDENCE-TYPE MIX DRIFT — of confirmed rings, what fraction were caught
   via shared payout vs. referral vs. claim burst vs. proximity, in the
   train period vs. the test period? If ring operators start avoiding
   payout convergence (the strongest, most-monitored signal) and lean
   more on weaker signals instead, this shows up directly here — a more
   fraud-specific, actionable drift signal than generic score drift alone.

PSI severity thresholds match the original project's convention:
  < 0.1  : no significant drift
  0.1-0.2: moderate drift, monitor
  >= 0.2 : severe drift, retrain recommended
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

PSI_MODERATE_THRESHOLD = 0.1
PSI_SEVERE_THRESHOLD = 0.2


def _psi(expected, actual, bins=10):
    """Population Stability Index — same from-scratch implementation
    approach as the original project. Bins the EXPECTED (train/reference)
    distribution, then compares bin proportions against ACTUAL (test/new)."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    breakpoints = np.linspace(np.min(expected), np.max(expected), bins + 1)
    breakpoints[0] -= 1e-6
    breakpoints[-1] += 1e-6

    def _bin_pct(data):
        counts, _ = np.histogram(data, bins=breakpoints)
        pct = counts / max(len(data), 1)
        return np.where(pct == 0, 1e-6, pct)   # avoid log(0)

    e_pct = _bin_pct(expected)
    a_pct = _bin_pct(actual)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def _psi_categorical(expected_counts: dict, actual_counts: dict):
    """PSI over a categorical distribution (evidence type mix) instead of
    binned continuous values — same underlying formula, categories are
    already the 'bins'."""
    categories = sorted(set(expected_counts) | set(actual_counts))
    e_total = sum(expected_counts.values()) or 1
    a_total = sum(actual_counts.values()) or 1
    psi = 0.0
    for c in categories:
        e_pct = max(expected_counts.get(c, 0) / e_total, 1e-6)
        a_pct = max(actual_counts.get(c, 0) / a_total, 1e-6)
        psi += (a_pct - e_pct) * np.log(a_pct / e_pct)
    return float(psi)


def _wasserstein(expected, actual):
    return float(scipy_stats.wasserstein_distance(expected, actual))


def _kl_divergence(expected, actual, bins=10):
    breakpoints = np.linspace(min(np.min(expected), np.min(actual)),
                               max(np.max(expected), np.max(actual)), bins + 1)
    e_counts, _ = np.histogram(expected, bins=breakpoints)
    a_counts, _ = np.histogram(actual, bins=breakpoints)
    e_p = np.where(e_counts == 0, 1e-6, e_counts / max(e_counts.sum(), 1))
    a_p = np.where(a_counts == 0, 1e-6, a_counts / max(a_counts.sum(), 1))
    return float(np.sum(a_p * np.log(a_p / e_p)))


def _ks_test(expected, actual):
    stat, pvalue = scipy_stats.ks_2samp(expected, actual)
    return float(stat), float(pvalue)


def _severity(psi):
    if psi >= PSI_SEVERE_THRESHOLD:
        return "severe"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "moderate"
    return "none"


def compute_risk_score_drift(live_xgb_proba=None):
    """PSI/Wasserstein/KL/KS between train-period and test-period risk
    score distributions — the same 4-metric bundle as the original
    project, applied to whether the model's scores look different now.

    BUGFIX: this used to always np.load() xgb_proba.npy fresh from disk,
    completely ignoring a session-only retrain — since retrain
    deliberately never writes to disk (by design, see retrain.py), the
    drift report would keep reporting the SAME pre-retrain numbers
    forever, making the retrain button look like it silently did nothing.
    Pass agent_live's live in-memory array (post-retrain) here when one
    exists, so drift genuinely reflects the currently-active model."""
    account_index = pd.read_csv(f"{DATA_DIR}/account_index.csv")
    xgb_proba = live_xgb_proba if live_xgb_proba is not None else np.load(f"{DATA_DIR}/xgb_proba.npy")
    account_index["risk_score"] = xgb_proba

    train_scores = account_index[account_index["split"] == "train"]["risk_score"].values
    test_scores = account_index[account_index["split"] == "test"]["risk_score"].values

    psi = _psi(train_scores, test_scores)
    wass = _wasserstein(train_scores, test_scores)
    kl = _kl_divergence(train_scores, test_scores)
    ks_stat, ks_pvalue = _ks_test(train_scores, test_scores)

    return {
        "metric": "risk_score_distribution",
        "train_n": len(train_scores), "test_n": len(test_scores),
        "train_mean": round(float(np.mean(train_scores)), 4),
        "test_mean": round(float(np.mean(test_scores)), 4),
        "psi": round(psi, 4), "wasserstein_distance": round(wass, 4),
        "kl_divergence": round(kl, 4), "ks_statistic": round(ks_stat, 4),
        "ks_pvalue": round(ks_pvalue, 6),
        "severity": _severity(psi),
    }


def compute_evidence_mix_drift(ring_details: dict, accounts_df: pd.DataFrame):
    """PSI over evidence-TYPE mix — of confirmed rings, what fraction used
    each evidence type, train-period rings vs. test-period rings. This is
    the fraud-specific signal: it directly shows whether ring operators
    have shifted which signals they trigger, not just generic score
    movement."""
    acct_split = accounts_df.set_index("account_id")["split"].to_dict()

    train_evidence_counts, test_evidence_counts = {}, {}
    for ring in ring_details.values():
        seeds = ring.get("internal_seeds", [])
        if not seeds:
            continue
        # classify the ring by its primary seed's split (train vs test period)
        split = acct_split.get(seeds[0])
        if split is None:
            continue
        bucket = train_evidence_counts if split == "train" else test_evidence_counts
        for e in ring.get("evidence_summary", []):
            bucket[e["type"]] = bucket.get(e["type"], 0) + e["account_count"]

    psi = _psi_categorical(train_evidence_counts, test_evidence_counts)
    return {
        "metric": "evidence_type_mix",
        "train_distribution": train_evidence_counts,
        "test_distribution": test_evidence_counts,
        "psi": round(psi, 4),
        "severity": _severity(psi),
    }


def compute_drift_report(live_xgb_proba=None):
    """Full drift report — both metrics, overall model-health verdict."""
    score_drift = compute_risk_score_drift(live_xgb_proba)

    bundle_path = f"{DATA_DIR}/frontend_bundle.json"
    accounts_df = pd.read_csv(f"{DATA_DIR}/account_index.csv")
    with open(bundle_path) as f:
        bundle = json.load(f)
    evidence_drift = compute_evidence_mix_drift(bundle["ring_details"], accounts_df)

    worst_severity = "severe" if "severe" in (score_drift["severity"], evidence_drift["severity"]) else \
                      "moderate" if "moderate" in (score_drift["severity"], evidence_drift["severity"]) else "none"

    return {
        "risk_score_drift": score_drift,
        "evidence_mix_drift": evidence_drift,
        "overall_severity": worst_severity,
        "retrain_recommended": worst_severity == "severe",
        "thresholds": {"psi_moderate": PSI_MODERATE_THRESHOLD, "psi_severe": PSI_SEVERE_THRESHOLD},
    }
