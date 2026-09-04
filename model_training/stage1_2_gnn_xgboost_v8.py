"""
Ring Sentinel v3 — Stage 1 (GraphSAGE + relation attention) + Stage 2 (XGBoost)
==================================================================================
Follows the confirmed architecture exactly:

  DATA GENERATOR -> GRAPH CONSTRUCTION (referral/payout/proximity) ->
  GraphSAGE + RELATION ATTENTION (learned embedding) -> XGBoost (fraud probability)

Inductive-split discipline (per the Elliptic-leakage finding in the plan doc):
  - TRAINING forward+backward pass runs ONLY on the subgraph induced by
    train-split accounts (nodes AND edges both endpoints train-split).
    No gradient ever touches a test-split node or edge.
  - After training, weights are frozen and ONE inference forward pass runs
    on the FULL graph (train+test) to produce embeddings for every node,
    test included. This is standard inductive GraphSAGE usage, not leakage:
    test nodes never influenced any weight update.
"""
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import xgboost as xgb
import shap
import json
import pickle

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = os.environ.get("STAGE_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
OUT_DIR = os.environ.get("STAGE_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_model"))
ABLATE = os.environ.get("ABLATE_SIGNAL", "")
import os
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================================
# LOAD DATA (reuse v2's generated dataset — same accounts/claims/payouts)
# =========================================================================
accounts = pd.read_csv(f"{DATA_DIR}/accounts.csv", parse_dates=["signup_date"])
claims = pd.read_csv(f"{DATA_DIR}/claims.csv", parse_dates=["claim_date"])
payouts = pd.read_csv(f"{DATA_DIR}/payouts.csv")
gt = pd.read_csv(f"{DATA_DIR}/ground_truth_rings.csv")
communities = pd.read_csv(f"{DATA_DIR}/legit_communities.csv")

accounts = accounts.reset_index(drop=True)
acct_to_idx = {a: i for i, a in enumerate(accounts["account_id"])}
n_nodes = len(accounts)

gt_accounts = set(gt["account_id"])
y_all = np.array([1 if a in gt_accounts else 0 for a in accounts["account_id"]])

# =========================================================================
# NODE FEATURES — reuse the same evidence signals as v2's tools, but as
# continuous features instead of booleans (GNN can use the raw signal,
# doesn't need us to hand-pick a threshold)
# =========================================================================
claims_by_acct = claims.groupby("account_id")
payout_linked = payouts.set_index("upi_id")["linked_accounts"].apply(
    lambda s: set(s.split(";")) if isinstance(s, str) else set()
).to_dict()
accounts_idx = accounts.set_index("account_id")

print("Building node features...")
claims_by_time = claims.sort_values("claim_date")
claim_times_by_acct = claims.groupby("account_id")["claim_date"].apply(list).to_dict()
all_claim_times = claims_by_time["claim_date"].values
all_claim_accts = claims_by_time["account_id"].values

def compute_burst_score(aid, own_times):
    """Count of 15-minute windows (around any of this account's own claims)
    that also contain >=2 OTHER distinct accounts claiming. This is Ring
    C's actual tell (tight synchronized rounds, no referral, no
    convergence) — it previously existed ONLY inside the deterministic
    agent tools; the GNN itself never saw it. Uses a sorted-array
    searchsorted sweep, not per-account nested loops, to stay tractable
    at 19k+ accounts / tens of thousands of claims."""
    if not own_times:
        return 0
    count = 0
    for t in own_times:
        lo = np.searchsorted(all_claim_times, t - pd.Timedelta(minutes=15), side="left")
        hi = np.searchsorted(all_claim_times, t + pd.Timedelta(minutes=15), side="right")
        window_accts = set(all_claim_accts[lo:hi]) - {aid}
        if len(window_accts) >= 2:
            count += 1
    return count

node_features = []
for _, row in accounts.iterrows():
    aid = row["account_id"]
    c = claims_by_acct.get_group(aid) if aid in claims_by_acct.groups else pd.DataFrame(columns=claims.columns)
    n_claims = len(c)
    total_amt = c["claim_amount"].sum() if n_claims else 0.0
    avg_amt = c["claim_amount"].mean() if n_claims else 0.0
    account_age_days = (accounts["signup_date"].max() - row["signup_date"]).days
    burst_score = compute_burst_score(aid, claim_times_by_acct.get(aid, []))   # NEW (v8)

    # NOTE: velocity (claims_per_day) and conv_strength (co-linked-account
    # count on shared payout) remain DELIBERATELY EXCLUDED — both are
    # near-tautological with the ring-generation rules. burst_score is
    # different in kind: it requires cross-referencing many OTHER
    # accounts' timestamps, not a threshold on this account's own row —
    # the same class of signal the deterministic tools always used, now
    # also exposed to the GNN, specifically to give Ring C (proximity +
    # burst only, no referral, no convergence) something to be caught by.
    node_features.append([n_claims, total_amt, avg_amt, account_age_days, burst_score])

node_features = np.array(node_features, dtype=np.float32)
# standardize
_feat_mean = node_features.mean(0)
_feat_std = node_features.std(0) + 1e-6
node_features = (node_features - _feat_mean) / _feat_std
print(f"Node feature matrix: {node_features.shape} (velocity + conv_strength REMOVED, burst_score ADDED)")

if ABLATE == "burst_score":
    node_features[:, 4] = 0.0   # zero the standardized burst_score column
    print("*** ABLATION ACTIVE: burst_score feature zeroed ***")
elif ABLATE == "raw_claim_features":
    node_features[:, 0:4] = 0.0   # zero n_claims, total_amt, avg_amt, account_age_days
    print("*** ABLATION ACTIVE: raw claim/amount/age features zeroed ***")

# =========================================================================
# GRAPH CONSTRUCTION — 3 relation types, exactly as specified
# =========================================================================
print("Building edges by relation type...")

def build_edge_index(pairs):
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    arr = np.array(pairs, dtype=np.int64).T
    return torch.tensor(arr, dtype=torch.long)

referral_pairs, convergence_pairs = [], []
tight_proximity_pairs, loose_proximity_pairs = [], []   # NEW (v8): split by timing tightness

# referral edges
for _, row in accounts.iterrows():
    if pd.notna(row["referred_by"]) and row["referred_by"] in acct_to_idx:
        referral_pairs.append((acct_to_idx[row["referred_by"]], acct_to_idx[row["account_id"]]))
        referral_pairs.append((acct_to_idx[row["account_id"]], acct_to_idx[row["referred_by"]]))

# convergence edges: accounts sharing a payout UPI with >1 linked account
for upi, linked in payout_linked.items():
    linked = [a for a in linked if a in acct_to_idx]
    for i in range(len(linked)):
        for j in range(i+1, len(linked)):
            convergence_pairs.append((acct_to_idx[linked[i]], acct_to_idx[linked[j]]))
            convergence_pairs.append((acct_to_idx[linked[j]], acct_to_idx[linked[i]]))

# proximity edges: same city+ASN. SPLIT into two relation types by signup
# gap tightness, instead of one undifferentiated "proximity" relation:
#   TIGHT  (<=60 min apart) — what Ring C's coordinated signups look like
#   LOOSE  (60min - 5 days) — what office/college/gig_cohort's organic,
#                              staggered signups look like
# Previously both were the same relation, so the GNN had no way to tell a
# ring's synchronized cluster apart from an office's normal staggered
# hiring — this was the direct, structural cause of Ring C being the
# weakest-performing type across every seed.
accounts_sorted = accounts.sort_values("signup_date")
from collections import defaultdict
city_asn_groups = defaultdict(list)
for _, row in accounts_sorted.iterrows():
    city_asn_groups[(row["city"], row["asn"])].append((row["account_id"], row["signup_date"]))

for key, members in city_asn_groups.items():
    for i in range(len(members)):
        cnt = 0
        for j in range(i+1, len(members)):
            gap = members[j][1] - members[i][1]
            if gap.days > 5:
                break
            pair = (acct_to_idx[members[i][0]], acct_to_idx[members[j][0]])
            pair_rev = (acct_to_idx[members[j][0]], acct_to_idx[members[i][0]])
            if gap <= pd.Timedelta(minutes=60):
                tight_proximity_pairs.append(pair); tight_proximity_pairs.append(pair_rev)
            else:
                loose_proximity_pairs.append(pair); loose_proximity_pairs.append(pair_rev)
            cnt += 1
            if cnt >= 8:   # cap fan-out per node
                break

edge_index_referral = build_edge_index(referral_pairs)
edge_index_convergence = build_edge_index(convergence_pairs)
edge_index_tight_proximity = build_edge_index(tight_proximity_pairs)
edge_index_loose_proximity = build_edge_index(loose_proximity_pairs)

# device edges: accounts sharing the SAME device_fingerprint (NEW v8 signal
# — see generate_dataset_v7.py's assign_shared_devices). Independent of
# every existing signal: two accounts can share a device having never
# referred each other, never converged on a payout, and never signed up
# anywhere near each other in time.
device_groups = defaultdict(list)
for _, row in accounts.iterrows():
    device_groups[row["device_fingerprint"]].append(row["account_id"])
device_pairs = []
for device, members in device_groups.items():
    if len(members) < 2:
        continue
    for i in range(len(members)):
        for j in range(i+1, len(members)):
            device_pairs.append((acct_to_idx[members[i]], acct_to_idx[members[j]]))
            device_pairs.append((acct_to_idx[members[j]], acct_to_idx[members[i]]))
edge_index_device = build_edge_index(device_pairs)

# ABLATION SWITCH: zero out one signal at a time, data untouched otherwise.
if ABLATE == "referral":
    edge_index_referral = torch.zeros((2, 0), dtype=torch.long)
elif ABLATE == "convergence":
    edge_index_convergence = torch.zeros((2, 0), dtype=torch.long)
elif ABLATE == "tight_proximity":
    edge_index_tight_proximity = torch.zeros((2, 0), dtype=torch.long)
elif ABLATE == "loose_proximity":
    edge_index_loose_proximity = torch.zeros((2, 0), dtype=torch.long)
elif ABLATE == "device":
    edge_index_device = torch.zeros((2, 0), dtype=torch.long)
if ABLATE:
    print(f"*** ABLATION ACTIVE: {ABLATE} disabled ***")
print(f"Edges: referral={edge_index_referral.shape[1]}, convergence={edge_index_convergence.shape[1]}, "
      f"tight_proximity={edge_index_tight_proximity.shape[1]}, loose_proximity={edge_index_loose_proximity.shape[1]}, "
      f"device={edge_index_device.shape[1]}")

x = torch.tensor(node_features, dtype=torch.float)
y = torch.tensor(y_all, dtype=torch.float)

train_mask = torch.tensor((accounts["split"] == "train").values)
test_mask = torch.tensor((accounts["split"] == "test").values)
train_idx_set = set(np.where(train_mask.numpy())[0].tolist())

# FIX (v6): carve a validation slice OUT OF TRAIN (stratified by label, since
# positives are a small minority) for early stopping. Loss/gradients only
# ever come from the FIT slice; val is used purely to decide when to stop,
# so this still never touches test.
from sklearn.model_selection import train_test_split
train_idx_arr = np.array(sorted(train_idx_set))
train_labels_arr = y_all[train_idx_arr]
fit_idx, val_idx = train_test_split(train_idx_arr, test_size=0.15, stratify=train_labels_arr, random_state=42)
fit_mask = torch.zeros(n_nodes, dtype=torch.bool); fit_mask[fit_idx] = True
val_mask = torch.zeros(n_nodes, dtype=torch.bool); val_mask[val_idx] = True
print(f"Train split further divided: fit={fit_mask.sum().item()} nodes ({int(train_labels_arr[np.isin(train_idx_arr, fit_idx)].sum())} positive), "
      f"val={val_mask.sum().item()} nodes ({int(y_all[val_idx].sum())} positive)")

def induce_train_subgraph_edges(edge_index):
    """Keep only edges where BOTH endpoints are train-split — strict inductive
    training discipline (no test-node ever contributes to a gradient)."""
    if edge_index.shape[1] == 0:
        return edge_index
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    mask = np.array([s in train_idx_set and d in train_idx_set for s, d in zip(src, dst)])
    return edge_index[:, mask]

train_edge_referral = induce_train_subgraph_edges(edge_index_referral)
train_edge_convergence = induce_train_subgraph_edges(edge_index_convergence)
train_edge_tight_proximity = induce_train_subgraph_edges(edge_index_tight_proximity)
train_edge_loose_proximity = induce_train_subgraph_edges(edge_index_loose_proximity)
train_edge_device = induce_train_subgraph_edges(edge_index_device)
print(f"Train-only edges: referral={train_edge_referral.shape[1]}, "
      f"convergence={train_edge_convergence.shape[1]}, tight_proximity={train_edge_tight_proximity.shape[1]}, "
      f"loose_proximity={train_edge_loose_proximity.shape[1]}, device={train_edge_device.shape[1]}")

# =========================================================================
# STAGE 1 — GraphSAGE + RELATION ATTENTION
# =========================================================================

class RelationAttentionSAGE(nn.Module):
    """Per-relation SAGEConv, combined via a LEARNED attention weight per
    relation (simplified stand-in for CARE-GNN's RL-based neighbor
    selector — the model learns which relation types are informative,
    rather than us hand-assigning weights).

    FIX #3 (residual self-path): a node with ZERO edges across all three
    relations previously collapsed toward a near-constant embedding,
    because every conv layer's neighbor-aggregation term is zero for an
    isolated node and the only surviving signal was each conv's own root
    (self) weight — small, shared across all isolated nodes regardless of
    their raw features, so they became nearly indistinguishable. This
    residual adds an EXPLICIT, feature-only linear path straight from the
    raw input features to the final embedding, added on top of whatever
    the graph convolutions produce. An isolated node's embedding is now
    guaranteed to vary with its own features (claim count, amounts,
    account age) instead of defaulting to one degenerate value shared by
    every disconnected node in the graph."""
    def __init__(self, in_dim, hidden_dim, out_dim, n_relations=3):
        super().__init__()
        self.conv1 = nn.ModuleList([SAGEConv(in_dim, hidden_dim) for _ in range(n_relations)])
        self.conv2 = nn.ModuleList([SAGEConv(hidden_dim, out_dim) for _ in range(n_relations)])
        self.attn1 = nn.Parameter(torch.ones(n_relations))
        self.attn2 = nn.Parameter(torch.ones(n_relations))
        self.residual = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        # FIX (v6): the residual in v5 had no scale control and dominated the
        # embedding, which let the model overfit train via the feature-only
        # shortcut instead of graph structure — this is the most likely
        # cause of v5's bimodal test-score collapse (better AUC, worse
        # recall at threshold). A learnable, SIGMOID-BOUNDED scale (init
        # ~0.3, so residual starts as a minority contributor) keeps the
        # "isolated nodes aren't degenerate" benefit while preventing the
        # residual from swamping the graph-derived signal.
        self.residual_scale_raw = nn.Parameter(torch.tensor(-0.85))  # sigmoid(-0.85) ≈ 0.30
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, x, edge_indices):
        attn1 = F.softmax(self.attn1, dim=0)
        h = 0
        for i, ei in enumerate(edge_indices):
            if ei.shape[1] == 0:
                continue
            h = h + attn1[i] * self.conv1[i](x, ei)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)

        attn2 = F.softmax(self.attn2, dim=0)
        h2 = 0
        for i, ei in enumerate(edge_indices):
            if ei.shape[1] == 0:
                continue
            h2 = h2 + attn2[i] * self.conv2[i](h, ei)

        residual_term = self.residual(x)
        embedding = h2 + torch.sigmoid(self.residual_scale_raw) * residual_term   # FIX (v6): bounded scale
        logit = self.classifier(embedding).squeeze(-1)
        return embedding, logit, attn1.detach(), attn2.detach()


model = RelationAttentionSAGE(in_dim=x.shape[1], hidden_dim=40, out_dim=20, n_relations=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_edges = [train_edge_referral, train_edge_convergence, train_edge_tight_proximity, train_edge_loose_proximity, train_edge_device]
full_edges = [edge_index_referral, edge_index_convergence, edge_index_tight_proximity, edge_index_loose_proximity, edge_index_device]

# class-weighted BCE — rings are a small minority

# class-weighted BCE — rings are a small minority
pos_weight = torch.tensor([(y[fit_mask] == 0).sum().item() / max((y[fit_mask] == 1).sum().item(), 1)])

print("\nTraining GraphSAGE (fit-split only for gradients, val-split for early stopping, "
      "test never touched)...")
model.train()
from sklearn.metrics import roc_auc_score as _roc_auc_score
best_val_auc = -1
best_state = None
patience = 3 if os.environ.get("FAST_ABLATION") else 4
no_improve_checks = 0
CHECK_EVERY = 15 if os.environ.get("FAST_ABLATION") else 10
MAX_EPOCHS = 150 if os.environ.get("FAST_ABLATION") else 300

for epoch in range(MAX_EPOCHS):
    model.train()
    optimizer.zero_grad()
    embedding, logit, _, _ = model(x, train_edges)
    loss = F.binary_cross_entropy_with_logits(logit[fit_mask], y[fit_mask], pos_weight=pos_weight)
    loss.backward()
    optimizer.step()

    if epoch % CHECK_EVERY == 0:
        model.eval()
        with torch.no_grad():
            _, val_logit, _, _ = model(x, train_edges)   # still train-induced edges only — val nodes are IN train split
            val_proba = torch.sigmoid(val_logit[val_mask]).numpy()
            val_y = y[val_mask].numpy()
            val_auc = _roc_auc_score(val_y, val_proba) if len(set(val_y.tolist())) > 1 else 0.5
            pred = (torch.sigmoid(logit[fit_mask]) > 0.5).float()
            acc = (pred == y[fit_mask]).float().mean().item()
        print(f"  epoch {epoch:3d}  loss {loss.item():.4f}  fit_acc {acc:.3f}  val_auc {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_checks = 0
        else:
            no_improve_checks += 1
            if no_improve_checks >= patience:
                print(f"  Early stopping at epoch {epoch} (best val_auc={best_val_auc:.4f}, "
                      f"{patience} checks without improvement)")
                break

model.load_state_dict(best_state)
print(f"Restored best checkpoint (val_auc={best_val_auc:.4f})")

print("\nInference pass on FULL graph (frozen weights, test nodes included — standard "
      "inductive use, not leakage: no gradient ever touched them)...")
model.eval()
with torch.no_grad():
    full_embedding, full_logit, attn1, attn2 = model(x, full_edges)

print(f"\nLearned relation attention (layer 1): referral={attn1[0]:.3f}, "
      f"convergence={attn1[1]:.3f}, tight_proximity={attn1[2]:.3f}, loose_proximity={attn1[3]:.3f}, device={attn1[4]:.3f}")
print(f"Learned relation attention (layer 2): referral={attn2[0]:.3f}, "
      f"convergence={attn2[1]:.3f}, tight_proximity={attn2[2]:.3f}, loose_proximity={attn2[3]:.3f}, device={attn2[4]:.3f}")

embeddings_np = full_embedding.numpy()

# =========================================================================
# FIX #1 — PERIODIC RESCORING SIMULATION
# =========================================================================
# Build an EARLY snapshot: node features + convergence edges restricted to
# only claims within EARLY_WINDOW_DAYS of each account's OWN signup date —
# i.e. what a live system would actually see if it scored the account
# almost immediately. Referral and proximity edges are available
# immediately at signup regardless (referral code entered at signup;
# city/ASN/signup-date known instantly), so those are unchanged. Weights
# are the SAME trained model (no retraining) — this simulates deploying a
# model once and re-scoring accounts at two points in time, not two models.
print("\nBuilding EARLY snapshot (claims within 3 days of each account's own signup)...")
EARLY_WINDOW_DAYS = 3

claims_indexed = claims.set_index("account_id")
signup_by_acct = accounts.set_index("account_id")["signup_date"]

early_node_features = []
early_payout_linked = defaultdict(set)
early_claims_rows = []
for _, row in accounts.iterrows():
    aid = row["account_id"]
    cutoff = row["signup_date"] + pd.Timedelta(days=EARLY_WINDOW_DAYS)
    c = claims_by_acct.get_group(aid) if aid in claims_by_acct.groups else pd.DataFrame(columns=claims.columns)
    c_early = c[c["claim_date"] <= cutoff]
    n_claims = len(c_early)
    total_amt = c_early["claim_amount"].sum() if n_claims else 0.0
    avg_amt = c_early["claim_amount"].mean() if n_claims else 0.0
    account_age_days = 0  # unknown/irrelevant at near-immediate scoring time, kept for shape parity
    early_burst_score = 0.0  # burst detection needs cross-account data beyond a 3-day window to be meaningful; omitted here
    early_node_features.append([n_claims, total_amt, avg_amt, account_age_days, early_burst_score])
    for _, cr in c_early.iterrows():
        early_payout_linked[cr["payout_upi_id"]].add(aid)

early_node_features = np.array(early_node_features, dtype=np.float32)
early_node_features = (early_node_features - _feat_mean) / _feat_std   # SAME scaler as full-snapshot training
early_x = torch.tensor(early_node_features, dtype=torch.float)

early_convergence_pairs = []
for upi, linked in early_payout_linked.items():
    linked = [a for a in linked if a in acct_to_idx]
    for i in range(len(linked)):
        for j in range(i+1, len(linked)):
            early_convergence_pairs.append((acct_to_idx[linked[i]], acct_to_idx[linked[j]]))
            early_convergence_pairs.append((acct_to_idx[linked[j]], acct_to_idx[linked[i]]))
early_edge_convergence = build_edge_index(early_convergence_pairs)
# referral + proximity are available immediately — reuse the full-snapshot versions
early_edges = [edge_index_referral, early_edge_convergence, edge_index_tight_proximity, edge_index_loose_proximity]

print(f"Early-snapshot convergence edges: {early_edge_convergence.shape[1]} "
      f"(full snapshot has {edge_index_convergence.shape[1]})")

model.eval()
with torch.no_grad():
    early_embedding, early_logit, _, _ = model(early_x, early_edges)

early_risk = torch.sigmoid(early_logit).numpy()
full_risk_gnn = torch.sigmoid(full_logit).numpy()

np.save(f"{OUT_DIR}/early_snapshot_gnn_risk.npy", early_risk)
np.save(f"{OUT_DIR}/full_snapshot_gnn_risk.npy", full_risk_gnn)

# report specifically on the 3 previously-isolated accounts from v4
PREVIOUSLY_MISSED = ["ACC003871", "ACC003911", "ACC003959"]
print("\n=== Periodic rescoring check on the 3 previously-isolated v4 misses ===")
for aid in PREVIOUSLY_MISSED:
    if aid in acct_to_idx:
        idx = acct_to_idx[aid]
        print(f"  {aid}: early_snapshot_risk={early_risk[idx]:.4f}  ->  full_snapshot_risk={full_risk_gnn[idx]:.4f}")

# =========================================================================
# STAGE 2 — XGBoost on [GNN embedding ⊕ raw evidence features]
# =========================================================================
print("\nTraining XGBoost on GNN embeddings + raw features (train split only)...")

combined_features = np.concatenate([embeddings_np, node_features], axis=1)
feature_names = [f"gnn_emb_{i}" for i in range(embeddings_np.shape[1])] + \
                 ["n_claims", "total_amt", "avg_amt", "account_age_days", "burst_score"]

X_train = combined_features[train_mask.numpy()]
y_train = y_all[train_mask.numpy()]
X_test = combined_features[test_mask.numpy()]
y_test = y_all[test_mask.numpy()]

scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
    random_state=42,
)
xgb_model.fit(X_train, y_train)

proba_all = xgb_model.predict_proba(combined_features)[:, 1]
proba_test = proba_all[test_mask.numpy()]

from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
auc = roc_auc_score(y_test, proba_test)
ap = average_precision_score(y_test, proba_test)
print(f"Test AUC-ROC: {auc:.4f}  |  Test Average Precision: {ap:.4f}")

# SHAP explainer (for the agent's evidence stage)
print("Computing SHAP values...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(combined_features)

# =========================================================================
# SAVE everything the agent stage needs
# =========================================================================
np.save(f"{OUT_DIR}/embeddings.npy", embeddings_np)
np.save(f"{OUT_DIR}/xgb_proba.npy", proba_all)
np.save(f"{OUT_DIR}/shap_values.npy", shap_values)
torch.save(model.state_dict(), f"{OUT_DIR}/gnn_model_state.pt")
np.save(f"{OUT_DIR}/feat_mean.npy", _feat_mean)
np.save(f"{OUT_DIR}/feat_std.npy", _feat_std)
with open(f"{OUT_DIR}/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
with open(f"{OUT_DIR}/feature_names.json", "w") as f:
    json.dump(feature_names, f)
accounts[["account_id", "split"]].to_csv(f"{OUT_DIR}/account_index.csv", index=False)

stage12_summary = {
    "n_nodes": n_nodes,
    "n_edges": {"referral": int(edge_index_referral.shape[1]), "convergence": int(edge_index_convergence.shape[1]),
                "tight_proximity": int(edge_index_tight_proximity.shape[1]),
                "loose_proximity": int(edge_index_loose_proximity.shape[1])},
    "n_train_edges": {"referral": int(train_edge_referral.shape[1]), "convergence": int(train_edge_convergence.shape[1]),
                       "tight_proximity": int(train_edge_tight_proximity.shape[1]),
                       "loose_proximity": int(train_edge_loose_proximity.shape[1])},
    "learned_attention_layer1": {"referral": float(attn1[0]), "convergence": float(attn1[1]),
                                  "tight_proximity": float(attn1[2]), "loose_proximity": float(attn1[3])},
    "learned_attention_layer2": {"referral": float(attn2[0]), "convergence": float(attn2[1]),
                                  "tight_proximity": float(attn2[2]), "loose_proximity": float(attn2[3])},
    "xgb_test_auc_roc": round(float(auc), 4),
    "xgb_test_average_precision": round(float(ap), 4),
    "final_train_loss": round(float(loss.item()), 4),
}
with open(f"{OUT_DIR}/stage12_summary.json", "w") as f:
    json.dump(stage12_summary, f, indent=2)

print("\n" + json.dumps(stage12_summary, indent=2))
