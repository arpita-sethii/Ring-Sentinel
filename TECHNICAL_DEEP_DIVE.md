# Ring Sentinel — Technical Deep Dive (Self-Reference)

This is the "explain it to myself" doc — the actual mechanics, numbers, and
reasoning behind every modeling decision, not the pitch version. Written to
be read again in six months when I've forgotten why a threshold is exactly
0.6766.

---

## 1. Dataset

### 1.1 What's already included (the real part)

The base isn't fully synthetic. Two arrays were extracted directly from
Kaggle's **IEEE-CIS Fraud Detection** competition dataset (569,877 real
legitimate transactions):

- `real_legit_amounts.npy` — real transaction amounts from actual legit
  (`isFraud == 0`) transactions
- `real_time_of_day_hours.npy` — real time-of-day distribution, computed as
  `(TransactionDT % 86400) / 3600` from the same legit population

These aren't decoration — every synthetic account's claim amounts and
timing are **sampled from these real distributions** (amounts scaled
0.35× to fit a promo-claim scale rather than raw e-commerce transaction
scale). This matters because it means the "normal" behavior baseline the
model has to distinguish fraud from is grounded in real consumer
transaction patterns, not a hand-picked Gaussian.

### 1.2 What we added — the synthetic layer

`generate_dataset_v7.py` builds the full account/claim/payout graph on top
of that real baseline:

**19,167 total accounts**, split three ways:
- 16,000 plain legit accounts (no structural relationship to anything)
- 2,144 accounts inside **73 legit communities** — real-world-shaped
  clusters (families, coworkers, classmates) that are *not* fraud but
  *do* share signup timing, city, or even payout habits. These exist
  specifically as **hard negatives** — without them, a model could hit
  high recall by just flagging "any cluster of related accounts,"
  which would be useless in production where families and offices
  legitimately share networks constantly.
- 1,023 accounts inside **142 actual fraud rings**

**Community types (5), 73 total:**
| Type | Count |
|---|---|
| Family | 35 |
| Office | 14 |
| College | 6 |
| Cashback campaign | 10 |
| Gig cohort | 8 |

**Ring types (6), 142 total — deliberately heterogeneous, not one attack pattern:**
| Type | Count | Signature |
|---|---|---|
| A — Referral | 26 | Chain of referral links, one recruiter pattern |
| B — Convergence | 24 | Multiple accounts routing payouts to the same mule UPI |
| C — Proximity | 24 | Tight signup/claim timing clusters, no shared payout |
| D — Mixed | 24 | Combines weaker versions of multiple signals — deliberately diluted |
| E — Adversarial | 24 | Built to evade single-signal detection (camouflaged claims, spread-out convergence) |
| F — Sleeper | 20 | Dormant accounts that activate late — evades velocity-based detection specifically |

**23,772 total claims**, ₹1,105,651.91 total claim volume across the dataset.

**Train/test split is by date** (2026-10-01 cutoff), not random shuffling —
14,766 accounts train, 4,401 test. This matters: a random split would let
ring-mates from the same ring leak across train/test, inflating apparent
performance. Date-based splitting forces genuine generalization.

### 1.3 The variance that actually matters

The reason ring types A–F exist as separate categories rather than one
generic "fraud ring" label: **each stresses a different detection signal**,
and this is what lets the ablation results (section 4.3) mean anything.
Ring B exists specifically to test whether payout convergence detection
works. Ring F exists specifically to test whether the model can catch
something with *no* velocity signal at all. If all 142 rings used the same
attack pattern, ring-level recall would just be one number with no
diagnostic value — I wouldn't know *which* capability is weak.

**Deliberate hardening added on top of the base generation** (to prevent
the model from learning shortcuts):
- Wider claim-burst windows (harder to catch via a naive fixed-window rule)
- Camouflage claims injected into ring members (some claims look
  individually legitimate)
- Diluted convergence (not every ring member routes to the mule UPI —
  some route elsewhere to blend in)
- **7% of *legit* accounts are heavy-claimers** — this one's important.
  Without it, "claims a lot" alone would be a strong enough signal to
  reach high recall without the model ever learning real relational
  structure. This forces the model to actually use the graph, not just
  claim velocity.

---

## 2. GraphSAGE — what it is, what it does, how it's customized

### 2.1 What GraphSAGE actually is

GraphSAGE ("SAmple and aggreGatE") is a graph neural network that learns
a dense vector representation (an *embedding*) for each node by
repeatedly aggregating information from its neighbors. Unlike a plain
feature-based classifier that only sees an account's own claim count,
GraphSAGE lets an account's embedding be shaped by *who it's connected to*
— so an account with completely unremarkable claims but which is tightly
connected to 6 other flagged accounts still ends up with an embedding
that reflects that risky neighborhood.

### 2.2 How it calculates the risk-relevant signal

It doesn't calculate risk scores directly — **GraphSAGE never sees a fraud
label during its forward pass at inference time.** What it produces is a
**16-dimensional embedding per account**, a compressed representation of
"what does this account's position in the network look like." That
embedding then feeds into XGBoost (section 3), which is the component
that actually outputs a risk probability. This separation matters: GNN =
representation learning, XGBoost = classification. Neither does the
other's job.

Mechanically, each of the 2 layers does:
1. For every account, gather the embeddings of its directly connected
   neighbors (separately, per relation type — see 2.3)
2. Aggregate those neighbor embeddings into a single vector
3. Combine the aggregated neighbor signal with the account's own current
   embedding to produce its updated embedding for this layer
4. Stack 2 of these layers, so information can propagate 2 hops out —
   an account's embedding is shaped not just by direct neighbors but by
   neighbors-of-neighbors

### 2.3 How we customized it

**The single biggest customization: 4 separate relation-specific
attention channels instead of one merged graph.**

A naive GNN setup merges every kind of edge (referral, shared payout,
proximity) into one homogeneous graph and treats them identically. We
didn't — the model has **4 distinct relation types**, each with its own
**learned attention weight**:

- Referral (who invited whom)
- Shared payout (same cash-out UPI)
- Tight proximity (signup within 60 minutes, same city + ASN)
- Loose proximity (signup 60 minutes to 5 days apart, same city + ASN)

The model learns, during training, *how much to trust each relation type*
— rather than assuming a referral link and a loose-proximity link carry
equal signal (they don't; see the ablation results, where proximity turns
out to be largely redundant while convergence is load-bearing).

**Other customizations:**
- **Scaled residual connections** across the 2 layers — stabilizes
  training, prevents the account's own original signal from being
  washed out after 2 rounds of neighbor aggregation.
- **Deliberately minimal, non-tautological node features (5, not more):**
  `n_claims`, `total_amt`, `avg_amt`, `account_age_days`, `burst_score`.
  We explicitly **excluded** `velocity` and `conv_strength` as raw input
  features, even though they'd likely boost raw accuracy — including
  them would let the model essentially memorize a reworded version of
  the label rather than learn genuine relational/behavioral patterns.
  This was a deliberate accuracy-for-validity trade (see section 5).
- **Strictly inductive training** — this is the part most likely to be
  gotten wrong by accident, so worth stating precisely: gradients are
  only ever computed using train-split-induced nodes and edges. The
  validation split is used purely for early stopping (patience = 4
  epochs). Test-period accounts are **never seen during training at
  all** — they're scored only at inference time, after weights are
  completely frozen. This is what makes the held-out metrics in
  section 4 actually mean something rather than being inflated by
  leakage.

---

## 3. XGBoost — what it does, what parameters, how risk scores get computed

### 3.1 What it does

XGBoost is the actual classifier. It takes the **21-dimensional combined
feature vector** — 16-dim GraphSAGE embedding concatenated with the same
5 raw features used as GNN input — and outputs:
- `risk_score`: a probability in [0, 1] that the account is part of a
  fraud ring
- SHAP-style per-feature contribution values, computed via XGBoost's own
  native `predict(..., pred_contribs=True)` (not the separate `shap`
  package — that dependency caused real environment problems, see the
  build log; native contribs give the identical values without the
  fragile dependency chain)

### 3.2 Hyperparameters (the exact, restored values)

```
learning_rate     = 0.05
max_depth         = 4
n_estimators      = 200
random_state      = 42
scale_pos_weight  = 16.811821471652593
eval_metric       = "aucpr"
enable_categorical = True
```

Two of these deserve explanation:

- **`scale_pos_weight ≈ 16.8`** — the dataset is severely imbalanced
  (1,023 fraud accounts out of 19,167 total ≈ 5.3%). Without reweighting,
  XGBoost would happily predict "not fraud" for almost everything and
  still post a deceptively high raw accuracy. This weight roughly
  matches the legit:fraud ratio, forcing the loss function to actually
  care about catching the minority class.
- **`eval_metric = "aucpr"`** (precision-recall AUC, not plain ROC-AUC)
  — on an imbalanced problem, ROC-AUC can look artificially good because
  the true-negative count dominates. PR-AUC is the honest metric for
  this class balance.

(These exact values became load-bearing during a later engineering
episode: retraining the model required *exactly* reproducing them, since
XGBoost's native `save_model`/`load_model` restores the trained trees but
**not** these sklearn-wrapper hyperparameters — silently falling back to
generic defaults on load, which would have produced a materially
different, worse refit if not caught and explicitly restored.)

---

## 4. Final results (v11)

### 4.1 Single-seed result (seed = 7, activity-based split)

| Metric | Value |
|---|---|
| Account precision | 99.4% |
| Account recall | 82.7% |
| Ring-level recall | 53/55 = 96.4% |
| False positives | 2 |
| FPs from legit communities (hard negatives) | **0** |

Per-ring-type recall:
| Type | Recall |
|---|---|
| A — Referral | 100% |
| B — Convergence | 100% |
| C — Proximity | 100% |
| D — Mixed | **71.4%** |
| E — Adversarial | 100% |
| F — Sleeper | 100% |

The zero false positives from legit communities specifically is the
number I actually trust most here — it means the hard-negative design
(section 1.2) worked: the model isn't just pattern-matching "cluster of
related accounts," it's distinguishing fraud clusters from legitimate
ones that look structurally similar.

### 4.2 Multi-seed result (3 seeds, more honest than the single run above)

| Metric | Mean | Std |
|---|---|---|
| Precision | 96.4% | ±1.6% |
| Recall | 82.0% | ±2.6% |
| Ring-level recall | 94.0% | ±1.9% |
| Sleeper (F-type) recall | 73.8% | ±5.1% |

Sleeper recall's much larger variance (±5.1% vs. ~2% for everything else)
is the honest signal that F-type detection is the least *reliable*
capability, even though the single-seed run above happened to show 100%.

### 4.3 Agent iteration history (v9 → v10 → v11) — the tiering fix, quantified

| | v9 (expand-first) | v10 (binary think-first) | v11 (tiered, current) |
|---|---|---|---|
| Precision | 74.2% | 99.3% | 99.4% |
| Recall | 86.2% | 77.8% | 82.7% |
| Ring-level recall | 98.2% | 92.7% | 96.4% |
| False positives | 111 | 2 | 2 |
| Community FPs | many | 0 | 0 |
| Ring C recall | 100% | 60% | 100% |

This table is the actual record of *why* the tiered evidence system
exists. v9 (expand on any lead, no verification) had great recall but
111 false positives — unusable. v10 (verify before expanding, binary)
fixed precision but broke Ring C specifically down to 60%, because its
signature (proximity + burst, no convergence) got treated too strictly
by a single global think-first gate. v11's 3-tier system (strong/medium/
weak, with medium requiring independent verification) recovered Ring C
back to 100% *without* giving back the precision gains.

### 4.4 Ablation study (2 seeds) — which signals actually matter

- **Removing payout convergence is catastrophic**: overall recall
  0.84 → 0.52, Ring B (convergence-defined) crashes 1.0 → 0.3–0.43.
  This is the load-bearing signal.
- **Removing burst_score** narrowly hurts Ring C specifically (1.0 → 0.8).
- **Referral and both proximity types are surprisingly redundant** —
  removing them individually barely moves the numbers. They function as
  corroborating signals more than primary detection signals.

---

## 5. Trade-offs (stated plainly, not glossed over)

1. **Recall ceiling is ~82-83%, deliberately.** The tiered "verify
   before expanding" design trades recall for precision on purpose —
   v9 proved that chasing every lead gets recall to 86% but at 111 false
   positives, which is not a usable system. The ~4-point recall cost
   between v9 and v11 is the actual price of getting FPs down to 2.

2. **Ring D (mixed-signal) is the unsolved case.** 71.4% recall, and
   unlike Ring C's regression, this one was never specifically targeted
   or fixed — it's an open gap, not a solved-then-broken capability.

3. **Sleeper ring detection is the least stable capability**, not the
   weakest on average. ±5.1% standard deviation across seeds means the
   model hasn't converged on a fully robust signature for dormant/
   delayed-activation fraud — it sometimes catches it well, sometimes
   doesn't, depending on initialization.

4. **Excluding velocity/conv_strength as raw features was a deliberate
   accuracy-for-validity trade.** Including them would likely raise
   the headline numbers, but for the wrong reason — they're close enough
   to being a reworded label that the model would be memorizing rather
   than learning transferable relational patterns. This is a real cost,
   paid on purpose.

5. **Retrain only touches XGBoost — GraphSAGE embeddings are permanently
   frozen after initial training.** Analyst feedback can recalibrate how
   existing embeddings map to a risk score, but the system cannot learn
   *new* graph relationship patterns from feedback. If ring operators
   invent a genuinely new relational pattern (a 5th relation type,
   effectively), retraining XGBoost alone won't catch it — that would
   require rerunning the full GNN training pipeline.

6. **The live agent's confidence formula is a simplified reimplementation**
   of this offline pipeline's more nuanced confidence calculation — a
   weighted noisy-OR combination, not the exact same code path. They can
   diverge slightly, which is why a live-computed confidence for a given
   seed account won't always exactly match the pre-computed value shown
   for that ring in the dashboard.

7. **Everything above is measured against a synthetic benchmark.**
   `generate_dataset_v7.py`'s ground truth is a carefully constructed
   approximation of real fraud-ring behavior, not real labeled
   production data. Real adversarial rings will differ from this
   distribution in ways this benchmark can't anticipate — these numbers
   describe performance against a specific, disclosed synthetic
   construction, not a guarantee of real-world performance.
