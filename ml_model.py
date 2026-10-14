"""
ml_model.py
Loads the trained Voting Classifier ensemble and provides a risk scoring
function for incoming authentication requests.

STATUS: Layer 4 (ML) is intentionally NOT being reworked in this pass.
Priority for this build phase is layers 1-3 (device fingerprint, SIM
binding, OTP) being genuinely real. The features marked TODO below are
still hardcoded constants carried over from the original implementation
and are the reason precision was poor in the defense-stage results
(type_encoded alone was ~30% of feature importance while being constant
at inference time). Fixing this is the next phase, not this one.
"""

import pickle
import numpy as np
import os
import time

MODEL_PATH = os.path.join('posttest_result', 'trained_voting_model.pkl')

_model_data = None


def load_model():
    """Load the trained voting ensemble from disk (once, cached in memory)."""
    global _model_data
    if _model_data is None:
        with open(MODEL_PATH, 'rb') as f:
            _model_data = pickle.load(f)
        print("ML model loaded successfully")
    return _model_data


def calculate_risk_score(features: dict) -> float:
    model_data = load_model()
    scaler       = model_data['scaler']
    feature_cols = model_data['feature_columns']

    feature_vector = np.array([[features.get(col, 0) for col in feature_cols]])
    feature_vector_scaled = scaler.transform(feature_vector)

    rf  = model_data['random_forest']
    xgb = model_data['xgboost']
    lr  = model_data['logistic_regression']

    p_rf  = rf.predict_proba(feature_vector_scaled)[0][1]
    p_xgb = xgb.predict_proba(feature_vector_scaled)[0][1]
    p_lr  = lr.predict_proba(feature_vector_scaled)[0][1]

    # Weighted average: RF=1, XGBoost=1, Logistic Regression=2
    risk_score = (p_rf * 1 + p_xgb * 1 + p_lr * 2) / 4

    return float(risk_score)


def get_decision_threshold() -> float:
    model_data = load_model()
    threshold = model_data.get('decision_threshold')
    if threshold is None or not isinstance(threshold, float):
        return 0.5
    return threshold


def classify_risk(risk_score: float) -> str:
    """
    Classify a risk score into LOW / MEDIUM / HIGH relative to the
    decision threshold tuned during training, not hardcoded constants.

    Bucketing logic:
      - HIGH:   score >= threshold               (same direction as training)
      - MEDIUM: score >= threshold * 0.6         (60% of the way to threshold)
      - LOW:    everything below medium
    This means the thresholds scale correctly if you retrain with a
    different optimal threshold, rather than always being 0.40/0.70.
    """
    threshold = get_decision_threshold()
    medium_threshold = threshold * 0.6

    if risk_score >= threshold:
        return 'HIGH'
    elif risk_score >= medium_threshold:
        return 'MEDIUM'
    else:
        return 'LOW'


def extract_features(session_data: dict) -> dict:
    """
    Build the model's feature vector from the current session context.

    REAL signals currently flowing in (from layers 1-3 of this build):
      - device_consistency : from auth.check_device_fingerprint(), now a
        genuine 1.0 / 0.2 signal based on real client-side fingerprint
        comparison, not a server-side header hash.

    TODO (phase 2 — feature pipeline rebuild, do not treat as done):
      - type_encoded, sim_age_days, new_account_flag, balance_change_*,
        transaction_velocity, high_risk_type, balance_discrepancy* are
        still constants passed in via app.py's session_context. These
        were trained on transaction data and don't yet reflect real
        per-login signals in an authentication-only flow. This is the
        root cause of the precision problem from the defense results
        and needs a proper redesign (e.g. replacing transaction-shaped
        features with auth-shaped ones: time-since-last-login,
        distinct-device-count-30d, geo-velocity, failed-attempt-rate).
    """
    hour = int(time.strftime('%H'))
    amount = session_data.get('last_transaction_amount', 100.0)

    AMOUNT_MEAN = 1397291.0
    AMOUNT_STD  = 2870788.5

    features = {
        # Layer 1: Device Fingerprinting — REAL signal
        'device_consistency': session_data.get('device_consistency', 0.8),

        # TODO(phase2): transaction-shaped, not auth-shaped — see docstring
        'type_encoded':       session_data.get('type_encoded', 3),

        # Layer 2: SIM Binding
        # TODO(phase2): sim_age_days/new_account_flag still constants
        'sim_age_days':       session_data.get('sim_age_days', 180),
        'new_account_flag':   session_data.get('new_account_flag', 0),

        # Layer 3: Time-based — REAL signal (computed from current time)
        'hour_of_day':          hour,
        'is_night_transaction': 1 if (hour >= 22 or hour <= 5) else 0,
        'is_peak_hours':        1 if (9 <= hour <= 17) else 0,

        # Layer 4: Behavioural — REAL where amount is passed in, else default
        'amount_log':                np.log1p(amount),
        'amount_z_score':            (amount - AMOUNT_MEAN) / AMOUNT_STD,
        'amount_anomaly':            1 if amount > (AMOUNT_MEAN + 2 * AMOUNT_STD) else 0,

        # TODO(phase2): all of the below are still constants from app.py
        'balance_change_initiator':  session_data.get('balance_change_initiator', 100.0),
        'balance_change_recipient':  session_data.get('balance_change_recipient', 100.0),
        'balance_change_ratio':      session_data.get('balance_change_ratio', 0.1),
        'zero_balance_initiator':    session_data.get('zero_balance_initiator', 0),
        'zero_balance_recipient':    session_data.get('zero_balance_recipient', 0),
        'transaction_velocity':      session_data.get('transaction_velocity', 2),
        'high_risk_type':            session_data.get('high_risk_type', 0),
        'balance_discrepancy':       session_data.get('balance_discrepancy', 0.0),
        'balance_discrepancy_ratio': session_data.get('balance_discrepancy_ratio', 0.0),
    }

    return features


def extract_transaction_features(transaction_data: dict, recent_transaction_count: int,
                                   device_consistency: float = 1.0) -> dict:
    """
    Build the model's feature vector at TRANSACTION time.

    HISTORY OF FIXES TO THIS FUNCTION (read before changing again):

    v1: passed raw amount directly as balance_change_initiator (e.g.
    9,000,000). Scaled to +87 std outside training range, saturated LR.

    v2: clipped balance_change_initiator/recipient to a 4-std ceiling
    (~400,676 / ~251,105). This caused a NEW bug: every transfer above
    that ceiling clipped to the SAME constant, so 2,000,000 and 4,500,000
    scored nearly identically (stress_test.py scenario 6 caught this —
    scores were flat at ~0.0245 for amounts 2M-3.5M). Also,
    balance_discrepancy used abs(amount - AMOUNT_MEAN), which is
    NON-MONOTONIC: since AMOUNT_MEAN (~1,397,291) is itself large, a
    SMALL transfer (1,000) is far from the mean and looked MORE
    anomalous than a large one (2,000,000) that happened to be close
    to the mean. This is why the baseline 1,000 transfer scored MEDIUM
    while 2-4.5M transfers scored LOW — backwards.

    v3 (this version): replaced distance-from-mean math with monotonic,
    amount-scaled features that increase consistently with transfer size,
    using percentile-based scaling against the training distribution
    instead of a hard 4-std clip ceiling that flattens everything above it.
    """
    hour = int(time.strftime('%H'))
    amount = float(transaction_data.get('amount', 0.0))

    model_data = load_model()
    amount_stats = model_data.get('amount_stats', {})
    AMOUNT_MEAN = float(amount_stats.get('mean', 1397291.0))
    AMOUNT_STD  = float(amount_stats.get('std',  2870788.5))

    # Training distribution stats for balance features (from Step 3 of
    # the training notebook's dataset analysis).
    BAL_CHANGE_STD = 102720.72
    BAL_RECIP_STD  = 59275.42

    # MONOTONIC scaling: use log-space growth so very large transfers
    # keep pushing the feature higher (never flattening to a constant)
    # while still landing in a range the model has seen. log1p(amount)
    # grows without bound but slowly, so a 1,000 transfer and a
    # 9,000,000 transfer land at clearly different, ORDERED positions
    # rather than both being clipped to the same ceiling.
    balance_change_initiator = float(np.log1p(amount) * (BAL_CHANGE_STD / 12.0))
    balance_change_recipient = float(np.log1p(amount) * (BAL_RECIP_STD / 12.0))

    # balance_change_ratio: fraction of typical balance this represents,
    # using log-scaled amount so it grows smoothly toward 1.0 instead of
    # being capped at a fixed transfer size.
    balance_change_ratio = float(np.clip(np.log1p(amount) / 18.0, 0.0, 1.0))

    # zero_balance_initiator: large transfers are more likely to drain
    # an account. Use a percentile-style threshold based on amount, not
    # a fixed cutoff, so it scales sensibly across the amount range.
    zero_balance_initiator = 1 if amount > AMOUNT_MEAN else 0

    # balance_discrepancy: previously non-monotonic (distance from mean).
    # Replaced with a MONOTONIC measure: how far above a "normal" small
    # transfer this amount is, scaled logarithmically so it always
    # increases with amount rather than being smallest near the mean.
    NORMAL_TRANSFER_BASELINE = 5000.0   # a typical small, low-risk transfer
    balance_discrepancy = float(np.log1p(max(amount - NORMAL_TRANSFER_BASELINE, 0)) * 5000.0)
    balance_discrepancy_ratio = float(np.clip(
        np.log1p(amount) / 10.0, 0.0, 10.0
    ))

    amount_z = (amount - AMOUNT_MEAN) / AMOUNT_STD

    features = {
        'device_consistency':        device_consistency,
        'type_encoded':              1,    # TRANSFER = most common fraud type
        'sim_age_days':              180,
        'new_account_flag':          0,

        'hour_of_day':               hour,
        'is_night_transaction':      1 if (hour >= 22 or hour <= 5) else 0,
        'is_peak_hours':             1 if (9 <= hour <= 17) else 0,

        # Amount features — REAL from actual transfer, monotonic
        'amount_log':                float(np.log1p(max(amount, 0))),
        'amount_z_score':            float(np.clip(amount_z, -5, 5)),
        'amount_anomaly':            1 if amount > (AMOUNT_MEAN + 2 * AMOUNT_STD) else 0,

        # Balance features — monotonic log-scaled, never flatten to a constant
        'balance_change_initiator':  balance_change_initiator,
        'balance_change_recipient':  balance_change_recipient,
        'balance_change_ratio':      balance_change_ratio,
        'zero_balance_initiator':    zero_balance_initiator,
        'zero_balance_recipient':    0,

        # Velocity — REAL from TransactionLog count
        'transaction_velocity':      float(np.clip(recent_transaction_count, 1, 100)),

        'high_risk_type':            1 if amount > (AMOUNT_MEAN + AMOUNT_STD) else 0,
        'balance_discrepancy':       balance_discrepancy,
        'balance_discrepancy_ratio': balance_discrepancy_ratio,
    }

    return features