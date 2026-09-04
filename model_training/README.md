# Model Training — GraphSAGE + XGBoost

These are the actual scripts that produced the trained-model artifacts the
backend serves (`backend/data/xgb_proba.npy`, `shap_values.npy`,
`account_index.csv`, `feature_names.json`). Included so the ML pipeline
itself — not just its output — is part of this project.

## What's here

```
model_training/
├── generate_dataset_v7.py          Synthetic benchmark generator: 6 heterogeneous
│                                    ring types + 5 legit-lookalike community types,
│                                    grounded in real IEEE-CIS (Kaggle) transaction
│                                    amount/timing distributions
├── stage1_2_gnn_xgboost_v8.py      GraphSAGE (relation-attention, PyTorch Geometric)
│                                    → embeddings → XGBoost classifier on top
├── kaggle_derived/
│   ├── real_legit_amounts.npy       Real transaction amounts sampled from IEEE-CIS
│   └── real_time_of_day_hours.npy   Real time-of-day patterns sampled from IEEE-CIS
└── requirements-training.txt
```

**Honest note on the Kaggle data:** the two small `.npy` files in
`kaggle_derived/` are real values sampled from Kaggle's IEEE-CIS Fraud
Detection dataset (569,877 legit transactions) — included directly since
they're tiny (160KB each) and sufficient to reproduce the exact same
synthetic benchmark. The raw ~650MB Kaggle CSV itself is NOT included. If
you want to regenerate `kaggle_derived/` from scratch instead of using the
included files: download `train_transaction.csv` from Kaggle's
[IEEE-CIS Fraud Detection competition](https://www.kaggle.com/c/ieee-fraud-detection),
filter to `isFraud == 0`, and sample `TransactionAmt` and
`(TransactionDT % 86400) / 3600` (time-of-day) into two `.npy` files with
those names.

## Run it

```bash
pip install -r requirements-training.txt
# PyTorch Geometric install can be platform/CUDA-specific — see
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
# if the plain pip install above doesn't resolve cleanly on your machine.

python3 generate_dataset_v7.py
# → writes accounts.csv, claims.csv, payouts.csv, ground_truth_rings.csv,
#   legit_communities.csv, account_index.csv into ./output/

STAGE_DATA_DIR=./output STAGE_OUT_DIR=./output_model python3 stage1_2_gnn_xgboost_v8.py
# → trains the GNN + XGBoost, writes xgb_proba.npy, shap_values.npy,
#   feature_names.json, account_index.csv into ./output_model/
```

**Verified working from this exact location** — `generate_dataset_v7.py`
was re-run from this folder as part of packaging this project and
reproduced the same 142-ring benchmark (26/24/24/24/24/20 rings across the
6 types) that `backend/data/` was built from.

## Wiring the output into the backend

To have the live backend use freshly-trained model output instead of the
included snapshot: copy `accounts.csv`, `claims.csv`, `payouts.csv` (from
`./output/`) and `xgb_proba.npy`, `shap_values.npy`, `feature_names.json`,
`account_index.csv` (from `./output_model/`) into `backend/data/`,
replacing the existing files, then restart the backend. You'll also want
to rebuild `frontend_bundle.json` (the pre-built ring/account views) —
that step isn't included here since it depends on the full agent
evaluation pipeline, not just the model training step.

## Architecture summary

- **GraphSAGE** (2-layer, relation-specific aggregation with learned
  per-relation attention — referral / payout-convergence / tight-proximity
  / loose-proximity kept as separate channels rather than one merged
  graph) produces a dense embedding per account, trained strictly
  inductively (train-period nodes/edges only ever seen during training;
  test-period accounts are scored only at inference, after weights are
  frozen — no leakage).
- **XGBoost** trained on `[GNN embedding ⊕ raw evidence features]` produces
  the final fraud-risk probability and SHAP feature importances — this is
  the `risk_score` and `shap_top_features` the backend serves for every
  account.
