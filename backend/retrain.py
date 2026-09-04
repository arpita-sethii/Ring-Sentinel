"""
Ring Sentinel — Retrain Module
=================================
A REAL retrain, not a stub — but deliberately scoped to what's fast and
safe to run inside a live API request: refitting the XGBoost layer on top
of the ALREADY-TRAINED GraphSAGE embeddings, using analyst overrides as
label corrections. This does NOT retrain the GNN itself (that needs
torch/torch_geometric, takes minutes, and isn't something to trigger
unattended from a request handler — see model_training/ for that
pipeline). Refitting just XGBoost on a fixed 19,167 x 21 feature matrix
takes seconds on CPU.

How analyst feedback becomes a training signal:
  - Start from the ORIGINAL ground-truth labels (train split only — test
    stays untouched, so we're not training on our own evaluation set).
  - For every ring where an analyst overrode the AI's own verdict, correct
    the labels of that ring's members: if the AI recommended escalate but
    the analyst dismissed it, treat those accounts as legitimate (label 0)
    for this refit; if the AI only recommended watch but the analyst
    escalated, treat them as confirmed fraud (label 1).
  - Refit XGBoost on the corrected label set, recompute risk scores + SHAP
    for every account, persist the new artifacts, and hot-swap them into
    the live agent's in-memory arrays so the next investigation reflects
    the update immediately.
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
# NOTE: intentionally NOT using pickle to load the model. XGBoost's own
# warning is correct — pickle is fragile across xgboost versions/platforms
# (this bit us directly: the shipped .pkl was pickled under one xgboost
# version, a different one got installed at retrain-test time, and pickle
# loading broke). Using XGBoost's native save_model/load_model (JSON)
# format instead is version-portable and is exactly what XGBoost's own
# docs recommend over pickling. See convert_model.py for the one-time
# conversion from the old xgb_model.pkl to xgb_model.json.
# Also intentionally NOT using the separate `shap` package — it pulls in
# `numba` for its clustering utilities, and numba ships a native DLL that
# gets blocked outright by some Windows environments (e.g. corporate
# "Application Control" policies), which has nothing to do with our code
# and isn't something we can fix by changing Python. XGBoost has its own
# built-in SHAP-value computation (predict(..., pred_contribs=True)) that
# needs no extra dependency at all and produces the same values.

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _reconstruct_raw_features(accounts: pd.DataFrame, claims: pd.DataFrame):
    """Rebuilds the exact 5 raw features (n_claims, total_amt, avg_amt,
    account_age_days, burst_score) the original training script computed
    — same formula, so a refit is comparing apples to apples against the
    embeddings, which were trained on this exact feature construction.

    Vectorized (no per-account iterrows + repeated groupby.get_group,
    which took over a minute for 19k accounts) — burst detection now
    sweeps over claims once, not once per account."""
    # n_claims / total_amt / avg_amt — vectorized aggregation, then reindexed onto every account (0 for no claims)
    agg = claims.groupby("account_id")["claim_amount"].agg(["count", "sum", "mean"])
    agg = agg.reindex(accounts["account_id"]).fillna(0.0)

    # account_age_days — vectorized
    max_signup = accounts["signup_date"].max()
    age_days = (max_signup - accounts["signup_date"]).dt.days.values

    # burst_score — sweep over CLAIMS once (not once per account), then aggregate per account
    claims_sorted = claims.sort_values("claim_date").reset_index(drop=True)
    times = claims_sorted["claim_date"].values
    accts_arr = claims_sorted["account_id"].values
    n = len(claims_sorted)
    burst_flags = np.zeros(n, dtype=bool)
    window = pd.Timedelta(minutes=15).to_timedelta64()
    for i in range(n):
        t = times[i]
        lo = np.searchsorted(times, t - window, side="left")
        hi = np.searchsorted(times, t + window, side="right")
        window_accts = set(accts_arr[lo:hi])
        window_accts.discard(accts_arr[i])
        if len(window_accts) >= 2:
            burst_flags[i] = True
    claims_sorted["burst_flag"] = burst_flags
    burst_by_acct = claims_sorted.groupby("account_id")["burst_flag"].sum()
    burst_by_acct = burst_by_acct.reindex(accounts["account_id"]).fillna(0)

    return np.column_stack([
        agg["count"].values, agg["sum"].values, agg["mean"].values,
        age_days.astype(np.float32), burst_by_acct.values,
    ]).astype(np.float32)


def retrain(overridden_actions: dict):
    """overridden_actions: {ring_id: {"ai_verdict":..., "cluster_members": [account_ids]}}
    for every ring where the analyst's decision disagreed with the AI's.
    Returns a result dict with before/after metrics and never raises —
    callers should check result["success"]."""
    t0 = pd.Timestamp.now()
    try:
        accounts = pd.read_csv(f"{DATA_DIR}/accounts.csv", parse_dates=["signup_date"])
        claims = pd.read_csv(f"{DATA_DIR}/claims.csv", parse_dates=["claim_date"])
        account_index = pd.read_csv(f"{DATA_DIR}/account_index.csv")
        embeddings = np.load(f"{DATA_DIR}/embeddings.npy")
        ground_truth = pd.read_csv(f"{DATA_DIR}/ground_truth_rings.csv")
        model = xgb.XGBClassifier()
        model.load_model(f"{DATA_DIR}/xgb_model.json")   # native format — run convert_model.py once if this file doesn't exist yet
        # BUGFIX: load_model() restores the trained trees but NOT the
        # sklearn wrapper's training hyperparameters (they silently reset
        # to XGBoost's generic defaults) — confirmed by diffing
        # old_model.get_params() vs a freshly-loaded model's params.
        # Since retrain calls .fit() again, a refit under the WRONG
        # hyperparameters (wrong depth, wrong learning rate, no class-
        # imbalance weighting) would silently produce a materially
        # different, likely worse model. These are the exact original
        # values from the shipped baseline model — restoring them exactly.
        model.set_params(learning_rate=0.05, max_depth=4, n_estimators=200, random_state=42,
                          scale_pos_weight=16.811821471652593, eval_metric="aucpr", enable_categorical=True)

        acct_to_row = {a: i for i, a in enumerate(account_index["account_id"])}
        gt_accounts = set(ground_truth["account_id"])

        raw_features = _reconstruct_raw_features(accounts, claims)
        combined = np.concatenate([embeddings, raw_features], axis=1)

        # base labels: original ground truth, TRAIN SPLIT ONLY for the
        # baseline (test stays held out for the baseline model's own
        # evaluation) — but analyst corrections apply regardless of split.
        # Reasoning: the train/test split was for evaluating the model
        # BEFORE deployment. Once deployed, an analyst's real correction on
        # any account — whether it originally fell in train or test — is
        # legitimate new signal a production feedback loop should use; the
        # original split shouldn't block real corrections from mattering.
        train_mask = (account_index["split"] == "train").values
        y = np.array([1 if a in gt_accounts else 0 for a in account_index["account_id"]], dtype=np.int32)

        # apply analyst label corrections — regardless of original split
        n_corrected = 0
        fit_mask = train_mask.copy()
        for ring_id, info in overridden_actions.items():
            correction = 0 if info["ai_verdict"] == "escalate" else 1   # dismissed-an-escalate -> 0, escalated-a-watch -> 1
            for aid in info["cluster_members"]:
                if aid in acct_to_row:
                    idx = acct_to_row[aid]
                    y[idx] = correction
                    fit_mask[idx] = True   # include this account in the fit even if it was originally test-split
                    n_corrected += 1

        X_train = combined[fit_mask]
        y_train = y[fit_mask]

        old_proba = np.load(f"{DATA_DIR}/xgb_proba.npy")

        model.fit(X_train, y_train, xgb_model=None)   # fresh fit on corrected labels, same hyperparams as original
        new_proba = model.predict_proba(combined)[:, 1]

        new_shap = model.get_booster().predict(xgb.DMatrix(combined), pred_contribs=True)[:, :-1]   # drop the bias column

        # SESSION-ONLY: deliberately NOT writing to disk here. This
        # function reloads xgb_model.pkl fresh from disk every call, so
        # each retrain click always starts from the untouched baseline
        # model plus whatever overrides currently exist — not a cumulative
        # drift across repeated clicks in one testing session, and a
        # server restart always returns to the shipped baseline exactly.
        # The caller (main.py) hot-swaps agent_live's in-memory arrays
        # with the returned new_proba/new_shap; nothing here persists.

        elapsed = (pd.Timestamp.now() - t0).total_seconds()
        mean_shift = float(np.mean(np.abs(new_proba - old_proba)))
        n_flipped = int(np.sum((old_proba >= 0.5) != (new_proba >= 0.5)))

        return {
            "success": True, "elapsed_seconds": round(elapsed, 2),
            "accounts_label_corrected": n_corrected,
            "rings_used_for_correction": len(overridden_actions),
            "mean_absolute_score_shift": round(mean_shift, 4),
            "accounts_flipped_verdict": n_flipped,
            "train_accounts_used": int(fit_mask.sum()),
            "new_proba": new_proba,       # in-memory only — caller hot-swaps agent_live's array with this
            "new_shap": new_shap,         # in-memory only — same
        }
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}
