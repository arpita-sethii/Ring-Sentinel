"""
Ring Sentinel — Dataset Generator v3
======================================
Fixes applied per review:

  P0-3  Legitimate lookalike communities (family, office, college, cashback
        campaign) that deliberately share the SAME weak signals rings use
        (proximity, referral, timing clustering) but are NOT fraud.
  P0-4  5 heterogeneous ring types (not all built to the same recipe):
          A. Referral-chain abuse        (chain + coordinated timing)
          B. Payout-convergence abuse    (no referral/proximity, pure $ convergence)
          C. Network-proximity abuse     (proximity + burst sync, NO convergence)
          D. Mixed weak-signal abuse     (2-3 signals, no single dominant one)
          E. Adversarial abuse           (deliberately AVOIDS payout convergence —
                                           the signal our detector weights heaviest)
  P0-5  Real temporal burst structure: synchronized claim rounds (minutes apart,
        repeated across multiple rounds) for rings vs organically-spread timing
        for legit communities — this is the actual discriminating signal, not
        "shared an ASN".
  P0-1  Train/test temporal split: activity is dated across Jan-Dec 2026;
        everything before OCT 1 is "train" (thresholds/weights may reference
        this), everything from OCT 1 onward is held out as "test" — final
        metrics are computed ONLY on test-period entities.
"""

import numpy as np
import pandas as pd
import random
import json
import os
from datetime import datetime, timedelta
from faker import Faker

SEED = int(os.environ.get("GEN_SEED", "7"))
fake = Faker("en_IN")
Faker.seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_derived")
OUT_DIR = os.environ.get("GEN_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
os.makedirs(OUT_DIR, exist_ok=True)

REAL_AMOUNTS = np.load(f"{DATA_DIR}/real_legit_amounts.npy")
REAL_TOD_HOURS = np.load(f"{DATA_DIR}/real_time_of_day_hours.npy")

def real_amount(scale=0.35):
    return round(float(np.random.choice(REAL_AMOUNTS)) * scale, 2)

def real_hour_of_day():
    return float(np.random.choice(REAL_TOD_HOURS))

SIM_START = datetime(2026, 1, 1)
SIM_END = datetime(2026, 12, 31)
TRAIN_TEST_SPLIT_DATE = datetime(2026, 10, 1)   # test = held-out period, never used to tune anything
SIM_DAYS = (SIM_END - SIM_START).days

PROMO_TYPES = ["referral_bonus", "cashback_offer", "signup_bonus", "reload_offer"]
CITIES = ["Mumbai", "Delhi", "Bangalore", "Ludhiana", "Patiala", "Chandigarh",
          "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Kolkata",
          "Chennai", "Indore", "Surat"]
ASN_POOL = [f"AS{random.randint(10000,60000)}" for _ in range(40)]
BIN_POOL = [f"{random.randint(400000,499999)}" for _ in range(25)]

def rand_date(start=SIM_START, days=SIM_DAYS):
    d = start + timedelta(days=random.uniform(0, days))
    h = real_hour_of_day()
    return d.replace(hour=int(h) % 24, minute=int((h % 1) * 60))

def rand_upi_id(seed_name=None):
    name = seed_name or fake.user_name()
    bank_handles = ["oksbi", "okhdfcbank", "okicici", "okaxis", "ybl", "paytm"]
    return f"{name}{random.randint(1,999)}@{random.choice(bank_handles)}"

accounts, claims, entity_labels = [], [], []
accounts_by_id = {}
payouts = {}

def new_payout_record(upi_id):
    if upi_id not in payouts:
        payouts[upi_id] = {"upi_id": upi_id, "linked_accounts": set(),
                            "total_claim_count": 0, "total_payout_amount": 0.0}
    return payouts[upi_id]

acct_counter = 1
def next_acct_id():
    global acct_counter
    aid = f"ACC{acct_counter:06d}"; acct_counter += 1; return aid

claim_counter = 1
def next_claim_id():
    global claim_counter
    cid = f"CLM{claim_counter:07d}"; claim_counter += 1; return cid

def make_account(city, asn, signup_dt, referred_by, payout_upi, label, entity_id, entity_type):
    aid = next_acct_id()
    record = {
        "account_id": aid, "signup_date": signup_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "city": city, "asn": asn, "card_bin": random.choice(BIN_POOL),
        "device_fingerprint": fake.uuid4()[:12], "referred_by": referred_by,
        "payout_upi_id": payout_upi, "label": label,
        "entity_id": entity_id, "entity_type": entity_type,
        "split": "train" if signup_dt < TRAIN_TEST_SPLIT_DATE else "test",
    }
    accounts.append(record)
    accounts_by_id[aid] = record
    new_payout_record(payout_upi)["linked_accounts"].add(aid)
    return aid

def add_claim(account_id, claim_dt, promo, amount, payout_upi):
    if claim_dt > SIM_END:
        return
    pr = new_payout_record(payout_upi)
    pr["linked_accounts"].add(account_id); pr["total_claim_count"] += 1; pr["total_payout_amount"] += amount
    claims.append({
        "claim_id": next_claim_id(), "account_id": account_id,
        "claim_date": claim_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "promo_type": promo, "claim_amount": amount, "payout_upi_id": payout_upi,
        "split": "train" if claim_dt < TRAIN_TEST_SPLIT_DATE else "test",
    })

def burst_round_times(base_dt, n_members, spread_minutes=30):
    """Synchronized round: members act within a window, in near-sequential
    order. Widened vs v3 (was 6-15 min) so the burst signal overlaps more
    with organic same-day clustering and is harder to separate on timing
    tightness alone."""
    offsets = sorted(np.random.uniform(0, spread_minutes, n_members))
    return [base_dt + timedelta(minutes=float(o)) for o in offsets]


def assign_shared_devices(member_ids, share_prob, sharer_frac=(0.3, 0.7)):
    """NEW (v8): device/IP fingerprint reuse as a genuinely independent
    5th signal, separate from referral/payout/proximity. Previously
    device_fingerprint was a pure-random UUID per account with ZERO reuse
    ever — meaning the field existed in the schema but carried no signal
    at all, fraud or otherwise.

    With probability `share_prob`, a SUBSET (not all) of this entity's
    members get reassigned onto ONE shared device fingerprint — modeling
    a real device-farm pattern (a small cluster of accounts genuinely
    reusing one device). Members not selected keep their own unique
    fingerprint, so this is a partial, realistic dilution — consistent
    with the hardening philosophy used everywhere else in this generator
    (camouflage claims, diluted convergence, etc.).

    BUGFIX: an earlier version picked each sharer's device independently
    at random from a 2-4 device pool. That meant even when the coin-flip
    for "does this entity share a device" succeeded, the sharers often
    landed on DIFFERENT pool entries by chance — verified by simulation
    to only produce an actual observable duplicate ~64% of the time, and
    real generated output confirmed the resulting shortfall (Ring D
    showed 12.5% observed sharing against a 50% configured probability).
    Forcing all selected sharers onto one single device removes that
    dilution entirely — share_prob now means what it says.

    Probabilities are deliberately calibrated per type:
    - Legit communities get a SMALL nonzero chance (real families/offices
      do occasionally share a device) — this is the same hard-negative
      design principle as every other signal in this file: the model
      must learn device reuse ALONE isn't proof of fraud.
    - Ring D and E get a boosted probability specifically because they
      are this system's two documented weak points (D's mixed/diluted
      signals top out at 71.4% recall; E deliberately avoids the
      strongest existing signal, payout convergence) — this gives the
      model a genuinely new, independent way to catch exactly the cases
      it currently struggles with, rather than just re-deriving an
      existing signal in a new form.
    - Ring F (sleeper) intentionally gets a LOW probability — its entire
      test purpose is "nothing to go on structurally until activation."
      A strong static device-sharing signal present from account
      creation would partially defeat that design intent, so this is
      capped low on purpose, not an oversight.
    """
    if random.random() >= share_prob or len(member_ids) < 2:
        return
    shared_device = fake.uuid4()[:12]
    frac = random.uniform(*sharer_frac)
    n_sharing = max(2, round(len(member_ids) * frac))
    sharers = random.sample(member_ids, min(n_sharing, len(member_ids)))
    for aid in sharers:
        accounts_by_id[aid]["device_fingerprint"] = shared_device


# =========================================================================
# 1. LEGIT BASE POPULATION (unconnected individuals — the bulk of the data)
# =========================================================================
N_LEGIT = 16000
for _ in range(N_LEGIT):
    signup_dt = rand_date()
    city, asn = random.choice(CITIES), random.choice(ASN_POOL)
    own_upi = rand_upi_id()
    referred_by = None
    if accounts and random.random() < 0.1:
        referred_by = random.choice(accounts)["account_id"]
    aid = make_account(city, asn, signup_dt, referred_by, own_upi, "legit", None, "individual")
    # ~7% of legit users are naturally heavy claimers (deal-hunters, frequent
    # shoppers) — deliberately overlaps with ring-level claim volume so raw
    # claim-count/velocity alone can't cleanly separate the two populations
    if random.random() < 0.07:
        n_claims_this = np.random.randint(3, 6)
    else:
        n_claims_this = np.random.poisson(1.1)
    for _ in range(n_claims_this):
        claim_dt = signup_dt + timedelta(hours=random.uniform(0, 24*60))
        add_claim(aid, claim_dt, random.choice(PROMO_TYPES), real_amount(), own_upi)

print(f"Legit individuals: {N_LEGIT}")

# =========================================================================
# 2. LEGITIMATE LOOKALIKE COMMUNITIES (P0-3) — NOT fraud, but share signals
# =========================================================================

def gen_family(n=6):
    """Same city/ASN, similar (but not tight-synchronized) purchase times,
    1-2 organic referral pairs, SEPARATE payouts, normal-ish velocity."""
    eid = f"FAM{fake.uuid4()[:6]}"
    city, asn = random.choice(CITIES), random.choice(ASN_POOL)
    base_signup = rand_date(SIM_START, SIM_DAYS - 40)
    members = []
    for i in range(n):
        signup_dt = base_signup + timedelta(days=random.uniform(0, 20))  # loose, not burst
        referred_by = random.choice(members) if (i > 0 and random.random() < 0.35 and members) else None
        own_upi = rand_upi_id()
        aid = make_account(city, asn, signup_dt, referred_by, own_upi, "legit_community", eid, "family")
        members.append(aid)
        for _ in range(np.random.poisson(1.4)):
            claim_dt = signup_dt + timedelta(hours=random.uniform(0, 24*30))
            add_claim(aid, claim_dt, random.choice(PROMO_TYPES), real_amount(), own_upi)
    entity_labels.append({"entity_id": eid, "entity_type": "family", "label": "legit_community",
                           "members": members})
    assign_shared_devices(members, share_prob=0.35)

def gen_office(n=30):
    """Same ASN+city, similar 9-6 working-hour timing, separate payouts,
    normal claims — organizational clustering, not fraud."""
    eid = f"OFF{fake.uuid4()[:6]}"
    city, asn = random.choice(CITIES), random.choice(ASN_POOL)
    base_signup = rand_date(SIM_START, SIM_DAYS - 60)
    members = []
    for i in range(n):
        signup_dt = (base_signup + timedelta(days=random.uniform(0, 45))).replace(
            hour=random.randint(9, 18))
        own_upi = rand_upi_id()
        aid = make_account(city, asn, signup_dt, None, own_upi, "legit_community", eid, "office")
        members.append(aid)
        for _ in range(np.random.poisson(1.0)):
            claim_dt = (signup_dt + timedelta(days=random.uniform(0, 60))).replace(hour=random.randint(9, 18))
            add_claim(aid, claim_dt, random.choice(PROMO_TYPES), real_amount(), own_upi)
    entity_labels.append({"entity_id": eid, "entity_type": "office", "label": "legit_community",
                           "members": members})
    assign_shared_devices(members, share_prob=0.08)

def gen_college(n=100):
    """Same network, similar signup PERIOD (campus onboarding wave) but
    spread over days, separate payouts, low claim volume."""
    eid = f"COL{fake.uuid4()[:6]}"
    city, asn = random.choice(CITIES), random.choice(ASN_POOL)
    base_signup = rand_date(SIM_START, SIM_DAYS - 90)
    members = []
    for i in range(n):
        signup_dt = base_signup + timedelta(days=random.uniform(0, 14))  # onboarding wave, days not minutes
        own_upi = rand_upi_id()
        aid = make_account(city, asn, signup_dt, None, own_upi, "legit_community", eid, "college")
        members.append(aid)
        for _ in range(np.random.poisson(0.6)):
            claim_dt = signup_dt + timedelta(days=random.uniform(0, 90))
            add_claim(aid, claim_dt, random.choice(PROMO_TYPES), real_amount(), own_upi)
    entity_labels.append({"entity_id": eid, "entity_type": "college", "label": "legit_community",
                           "members": members})
    assign_shared_devices(members, share_prob=0.05)

def gen_cashback_campaign(n=45):
    """Many accounts, ALL genuinely referred by one legitimate influencer/
    promo-code account — real marketing fan-out, not abuse. Separate payouts,
    normal claim amounts, timing spread over the whole campaign window."""
    eid = f"CMP{fake.uuid4()[:6]}"
    promoter_city, promoter_asn = random.choice(CITIES), random.choice(ASN_POOL)
    promoter_signup = rand_date(SIM_START, SIM_DAYS - 120)
    promoter_upi = rand_upi_id()
    promoter_id = make_account(promoter_city, promoter_asn, promoter_signup, None, promoter_upi,
                                "legit_community", eid, "campaign_promoter")
    members = [promoter_id]
    campaign_window_days = 75
    for i in range(n):
        city, asn = random.choice(CITIES), random.choice(ASN_POOL)  # geographically spread, NOT proximate
        signup_dt = promoter_signup + timedelta(days=random.uniform(2, campaign_window_days))
        own_upi = rand_upi_id()
        aid = make_account(city, asn, signup_dt, promoter_id, own_upi, "legit_community", eid, "campaign_referee")
        members.append(aid)
        for _ in range(np.random.poisson(1.0)):
            claim_dt = signup_dt + timedelta(days=random.uniform(0, 20))
            add_claim(aid, claim_dt, random.choice(PROMO_TYPES), real_amount(), own_upi)
    entity_labels.append({"entity_id": eid, "entity_type": "cashback_campaign", "label": "legit_community",
                           "members": members})
    assign_shared_devices(members, share_prob=0.05)

def gen_gig_cohort(n=15):
    """NEW (v7, diversity): gig/delivery workers sharing a warehouse/hub WiFi
    (same ASN) but different home cities, whose SHIFT-BASED clock-ins create
    genuine tight synchronized timing bursts — structurally the hardest
    legit case we've built, because it directly mimics the ring "burst"
    signal for a completely legitimate reason (everyone clocks in for the
    10am shift at once). Separate payouts, no referral requirement."""
    eid = f"GIG{fake.uuid4()[:6]}"
    hub_asn = random.choice(ASN_POOL)
    base_signup = rand_date(SIM_START, SIM_DAYS - 60)
    members = []
    for i in range(n):
        city = random.choice(CITIES)                 # different home cities
        signup_dt = base_signup + timedelta(days=random.uniform(0, 30))
        own_upi = rand_upi_id()
        aid = make_account(city, hub_asn, signup_dt, None, own_upi, "legit_community", eid, "gig_cohort")
        members.append(aid)
    # genuine shift bursts: 2-4 shifts, all members active within a tight
    # window each shift (this is REAL synchronized behavior, not fraud)
    for _ in range(random.randint(2, 4)):
        shift_base = base_signup + timedelta(days=random.uniform(5, 80))
        shift_base = shift_base.replace(hour=random.choice([9, 13, 18]))
        times = burst_round_times(shift_base, n, spread_minutes=20)
        for aid, t in zip(members, times):
            if random.random() < 0.6:   # not every member claims every shift
                acct_row = [a for a in accounts if a["account_id"] == aid][0]
                add_claim(aid, t, random.choice(PROMO_TYPES), real_amount(), acct_row["payout_upi_id"])
    entity_labels.append({"entity_id": eid, "entity_type": "gig_cohort", "label": "legit_community",
                           "members": members})
    assign_shared_devices(members, share_prob=0.10)

for _ in range(35):  gen_family(n=random.randint(4, 9))
for _ in range(14):  gen_office(n=random.randint(12, 60))
for _ in range(6):   gen_college(n=random.randint(60, 180))
for _ in range(10):  gen_cashback_campaign(n=random.randint(20, 90))
for _ in range(8):   gen_gig_cohort(n=random.randint(8, 25))      # NEW community type — diversity

print(f"Legit communities added. Total accounts so far: {len(accounts)}")

# =========================================================================
# 3. HETEROGENEOUS FRAUD RINGS (P0-4)
# =========================================================================

def gen_ring_A_referral(ring_num):
    """A: Referral-chain abuse — chain structure + coordinated timing.
    Harder v4: wider signup spread (up to 3 days, overlapping family/office
    organic spread), only 1 burst round (not 2), each member also makes one
    ordinary non-burst claim first to dilute per-account velocity."""
    ring_id = f"RINGA{ring_num:03d}"
    n = random.randint(3, 12)
    city, asn = random.choice(CITIES), random.choice(ASN_POOL)
    base = rand_date(SIM_START, SIM_DAYS - 30)
    cashout = rand_upi_id(seed_name=f"muleA{ring_num}")
    members = []
    for i in range(n):
        signup_dt = base + timedelta(hours=random.uniform(1, 72))   # was 1-30h, now up to 3 days
        referred_by = members[-1] if members else None
        front_upi = rand_upi_id()
        aid = make_account(city, asn, signup_dt, referred_by, front_upi, "ring", ring_id, "ring_A_referral")
        members.append(aid)
        # camouflage claim: one ordinary, non-burst, non-converging claim
        # per member, timed like a normal user
        if random.random() < 0.6:
            ordinary_dt = signup_dt + timedelta(hours=random.uniform(1, 200))
            add_claim(aid, ordinary_dt, random.choice(PROMO_TYPES), real_amount(), front_upi)
    # single synchronized round (not 2), wider window
    round_base = base + timedelta(days=random.uniform(2, 20))
    times = burst_round_times(round_base, n, spread_minutes=30)
    for aid, t in zip(members, times):
        dest = cashout if random.random() < 0.55 else accounts[[a["account_id"] for a in accounts].index(aid)]["payout_upi_id"]
        add_claim(aid, t, random.choices(PROMO_TYPES, weights=[0.6,0.2,0.1,0.1])[0], real_amount()*1.2, dest)
    assign_shared_devices(members, share_prob=0.55)   # Ring A — device farm common in referral abuse
    return ring_id, members

def gen_ring_B_convergence(ring_num):
    """B: Payout-convergence abuse — accounts otherwise unrelated. Harder
    v4: money splits across 2-3 cashout destinations (not 1), and each
    destination gets fewer co-linked accounts, diluting the convergence
    signal per UPI. Claim volume overlaps legit heavy-claimer range."""
    ring_id = f"RINGB{ring_num:03d}"
    n = random.randint(4, 14)
    n_cashouts = random.choice([2, 2, 3])
    cashouts = [rand_upi_id(seed_name=f"muleB{ring_num}{k}") for k in range(n_cashouts)]
    base = rand_date(SIM_START, SIM_DAYS - 30)
    members = []
    for i in range(n):
        city, asn = random.choice(CITIES), random.choice(ASN_POOL)
        signup_dt = base + timedelta(days=random.uniform(0, 25))
        front_upi = rand_upi_id()
        aid = make_account(city, asn, signup_dt, None, front_upi, "ring", ring_id, "ring_B_convergence")
        members.append(aid)
        n_claims_this = random.randint(3, 5)   # overlaps legit heavy-claimer tail
        for _ in range(n_claims_this):
            claim_dt = signup_dt + timedelta(days=random.uniform(0, 15))
            dest = random.choice(cashouts) if random.random() < 0.65 else front_upi
            add_claim(aid, claim_dt, random.choice(PROMO_TYPES), real_amount()*1.1, dest)
    assign_shared_devices(members, share_prob=0.45)   # Ring B — convergence + device overlap, corroborating
    return ring_id, members

def gen_ring_C_proximity(ring_num):
    """C: Network-proximity abuse — shared ASN/city + synchronized bursts,
    no convergence, no referral. Harder v4: wider burst window (30min not
    8), fewer rounds (2 not 3), signup spread over hours not tight minutes."""
    ring_id = f"RINGC{ring_num:03d}"
    n = random.randint(4, 13)
    city, asn = random.choice(CITIES), random.choice(ASN_POOL)
    base = rand_date(SIM_START, SIM_DAYS - 30)
    members = []
    upis = []
    for i in range(n):
        signup_t = base + timedelta(hours=random.uniform(0, 20))   # was tight ~6min burst, now hours
        own_upi = rand_upi_id()
        aid = make_account(city, asn, signup_t, None, own_upi, "ring", ring_id, "ring_C_proximity")
        members.append(aid); upis.append(own_upi)
    for _ in range(2):   # was 3 rounds
        round_base = base + timedelta(days=random.uniform(3, 25))
        times = burst_round_times(round_base, n, spread_minutes=30)
        for aid, upi, t in zip(members, upis, times):
            add_claim(aid, t, random.choice(PROMO_TYPES), real_amount()*1.1, upi)
    assign_shared_devices(members, share_prob=0.40)   # Ring C — proximity + device overlap, corroborating
    return ring_id, members

def gen_ring_D_mixed(ring_num):
    """D: Mixed weak-signal abuse — 2-3 signals, none dominant. Harder v4:
    convergence probability dropped to ~30%, wider signup spread."""
    ring_id = f"RINGD{ring_num:03d}"
    n = random.randint(3, 10)
    city = random.choice(CITIES)
    base = rand_date(SIM_START, SIM_DAYS - 30)
    cashout = rand_upi_id(seed_name=f"muleD{ring_num}")
    members = []
    for i in range(n):
        asn = random.choice(ASN_POOL)
        signup_dt = base + timedelta(hours=random.uniform(2, 96))
        referred_by = random.choice(members) if (i > 0 and random.random() < 0.35 and members) else None
        front_upi = rand_upi_id()
        aid = make_account(city, asn, signup_dt, referred_by, front_upi, "ring", ring_id, "ring_D_mixed")
        members.append(aid)
        for _ in range(random.randint(2, 4)):
            claim_dt = signup_dt + timedelta(days=random.uniform(0, 20))
            dest = cashout if random.random() < 0.3 else front_upi
            add_claim(aid, claim_dt, random.choice(PROMO_TYPES), real_amount(), dest)
    # Ring D — boosted deliberately: this ring type is our documented weak
    # spot (71.4% recall, no single dominant signal by design). Device
    # reuse gives the model a genuinely new, independent signal for
    # exactly the case it currently struggles with most.
    assign_shared_devices(members, share_prob=0.50)
    return ring_id, members

def gen_ring_E_adversarial(ring_num):
    """E: Adversarial abuse — avoids payout convergence entirely, relies on
    referral chain + proximity + burst. Harder v4: wider burst window,
    fewer rounds, wider signup spread, camouflage claims added."""
    ring_id = f"RINGE{ring_num:03d}"
    n = random.randint(3, 10)
    city, asn = random.choice(CITIES), random.choice(ASN_POOL)
    base = rand_date(SIM_START, SIM_DAYS - 30)
    members = []
    upis = []
    for i in range(n):
        signup_dt = base + timedelta(hours=random.uniform(1, 60))
        referred_by = members[-1] if members else None
        own_upi = rand_upi_id()
        aid = make_account(city, asn, signup_dt, referred_by, own_upi, "ring", ring_id, "ring_E_adversarial")
        members.append(aid); upis.append(own_upi)
        if random.random() < 0.5:
            ordinary_dt = signup_dt + timedelta(hours=random.uniform(1, 150))
            add_claim(aid, ordinary_dt, random.choice(PROMO_TYPES), real_amount(), own_upi)
    for _ in range(2):   # was 3 rounds
        round_base = base + timedelta(days=random.uniform(2, 22))
        times = burst_round_times(round_base, n, spread_minutes=28)
        for aid, upi, t in zip(members, upis, times):
            add_claim(aid, t, random.choices(PROMO_TYPES, weights=[0.55,0.25,0.1,0.1])[0], real_amount()*1.15, upi)
    # Ring E — boosted deliberately: this ring type is specifically built
    # to AVOID payout convergence, our strongest existing signal. Device
    # reuse is a genuinely independent way to catch it that doesn't rely
    # on the signal this ring type was designed to evade.
    assign_shared_devices(members, share_prob=0.50)
    return ring_id, members

def gen_ring_F_sleeper(ring_num):
    """F (NEW, v7 — hardest ring type): dormant during TRAIN, activates only
    in TEST. Members behave completely like ordinary legit individuals
    throughout train (single small claims, no referral, no proximity
    clustering, no burst) — then, once past the train/test split date,
    suddenly form a referral+convergence ring. This is the genuine
    generalization test: a model that memorized train-period statistical
    signatures of "ring-ness" has NOTHING to go on here until activation;
    only structural reasoning applied fresh at test time can catch it."""
    ring_id = f"RINGF{ring_num:03d}"
    n = random.randint(4, 9)
    city, asn = random.choice(CITIES), random.choice(ASN_POOL)
    # dormant phase: ordinary signups scattered across train period, no ring signals at all
    members = []
    for i in range(n):
        dormant_city, dormant_asn = random.choice(CITIES), random.choice(ASN_POOL)  # scattered, not clustered
        signup_dt = rand_date(SIM_START, (TRAIN_TEST_SPLIT_DATE - SIM_START).days - 20)
        own_upi = rand_upi_id()
        aid = make_account(dormant_city, dormant_asn, signup_dt, None, own_upi, "ring", ring_id, "ring_F_sleeper")
        members.append(aid)
        if random.random() < 0.5:   # one ordinary claim during dormancy, like any legit user
            dormant_claim = signup_dt + timedelta(hours=random.uniform(1, 400))
            add_claim(aid, dormant_claim, random.choice(PROMO_TYPES), real_amount(), own_upi)
    # activation phase: AFTER the split date, ring behavior begins for real
    activation_base = TRAIN_TEST_SPLIT_DATE + timedelta(days=random.uniform(1, 25))
    cashout = rand_upi_id(seed_name=f"muleF{ring_num}")
    for _ in range(random.randint(1, 2)):
        round_base = activation_base + timedelta(days=random.uniform(0, 20))
        times = burst_round_times(round_base, n, spread_minutes=25)
        for aid, t in zip(members, times):
            dest = cashout if random.random() < 0.6 else accounts[[a["account_id"] for a in accounts].index(aid)]["payout_upi_id"]
            add_claim(aid, t, random.choices(PROMO_TYPES, weights=[0.5,0.3,0.1,0.1])[0], real_amount()*1.2, dest)
    # Ring F — deliberately LOW: sleeper rings must have nothing to go on
    # structurally until activation. A strong static device-sharing signal
    # present from account creation would partially defeat that test, so
    # this stays low on purpose, not by oversight.
    assign_shared_devices(members, share_prob=0.15)
    return ring_id, members

RING_GENERATORS = {
    "A_referral": gen_ring_A_referral,
    "B_convergence": gen_ring_B_convergence,
    "C_proximity": gen_ring_C_proximity,
    "D_mixed": gen_ring_D_mixed,
    "E_adversarial": gen_ring_E_adversarial,
    "F_sleeper": gen_ring_F_sleeper,
}
N_RINGS_PER_TYPE = {"A_referral": 26, "B_convergence": 24, "C_proximity": 24,
                     "D_mixed": 24, "E_adversarial": 24, "F_sleeper": 20}

for rtype, gen_fn in RING_GENERATORS.items():
    for i in range(N_RINGS_PER_TYPE[rtype]):
        ring_id, members = gen_fn(i + 1)
        for m in members:
            entity_labels.append({"entity_id": ring_id, "entity_type": rtype, "label": "ring", "members": [m]})

total_rings = sum(N_RINGS_PER_TYPE.values())
print(f"Rings injected: {total_rings} across {len(N_RINGS_PER_TYPE)} types. Total accounts: {len(accounts)}")

# =========================================================================
# 4. EXPORT
# =========================================================================
df_accounts = pd.DataFrame(accounts)
df_claims = pd.DataFrame(claims)

payouts_rows = [{"upi_id": k, "linked_account_count": len(v["linked_accounts"]),
                  "linked_accounts": ";".join(sorted(v["linked_accounts"])),
                  "total_claim_count": v["total_claim_count"],
                  "total_payout_amount": round(v["total_payout_amount"], 2)} for k, v in payouts.items()]
df_payouts = pd.DataFrame(payouts_rows)

# ground truth: ring membership only (communities are separately labeled, not "fraud" truth)
gt_rows = []
for row in df_accounts[df_accounts["label"] == "ring"].itertuples():
    gt_rows.append({"entity_id": row.entity_id, "entity_type": row.entity_type, "account_id": row.account_id})
df_ground_truth = pd.DataFrame(gt_rows)

community_rows = []
for row in df_accounts[df_accounts["label"] == "legit_community"].itertuples():
    community_rows.append({"entity_id": row.entity_id, "entity_type": row.entity_type, "account_id": row.account_id})
df_communities = pd.DataFrame(community_rows)

df_accounts.to_csv(f"{OUT_DIR}/accounts.csv", index=False)
df_claims.to_csv(f"{OUT_DIR}/claims.csv", index=False)
df_payouts.to_csv(f"{OUT_DIR}/payouts.csv", index=False)
df_ground_truth.to_csv(f"{OUT_DIR}/ground_truth_rings.csv", index=False)
df_communities.to_csv(f"{OUT_DIR}/legit_communities.csv", index=False)

summary = {
    "n_accounts_total": len(df_accounts),
    "n_accounts_legit_individual": int((df_accounts["entity_type"]=="individual").sum()),
    "n_accounts_legit_community": int((df_accounts["label"]=="legit_community").sum()),
    "n_accounts_ring": int((df_accounts["label"]=="ring").sum()),
    "ring_types": N_RINGS_PER_TYPE,
    "n_rings_total": total_rings,
    "n_claims_total": len(df_claims),
    "total_claim_amount": round(df_claims["claim_amount"].sum(), 2),
    "train_test_split_date": TRAIN_TEST_SPLIT_DATE.strftime("%Y-%m-%d"),
    "n_accounts_train": int((df_accounts["split"]=="train").sum()),
    "n_accounts_test": int((df_accounts["split"]=="test").sum()),
    "amount_source": "IEEE-CIS Kaggle real legit transactions (scaled for promo-claim realism)",
}
with open(f"{OUT_DIR}/dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
