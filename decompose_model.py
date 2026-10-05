"""
decompose_ensemble.py
Breaks the soft-voting ensemble into its three component models (Random
Forest, XGBoost, Logistic Regression) and shows what EACH one predicts,
alongside what calculate_risk_score() in ml_model.py ACTUALLY returns —
the real weighted score your Flask app uses, not a separately
reimplemented unweighted average.

PREVIOUS BUG: this script used to call voting_clf.predict_proba()
directly, which is the OLD unweighted soft-voting logic. That meant it
never reflected the weighted-voting fix applied to ml_model.py, and gave
the impression the fix wasn't working when actually the fix was just
never being exercised by this script. Fixed by importing and calling
calculate_risk_score() directly.

Run with: python decompose_ensemble.py
"""

import pickle
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_model import calculate_risk_score, classify_risk, get_decision_threshold

MODEL_PATH = os.path.join('posttest_result', 'trained_voting_model.pkl')

with open(MODEL_PATH, 'rb') as f:
    data = pickle.load(f)

cols      = data['feature_columns']
scaler    = data['scaler']
rf        = data['random_forest']
xgb       = data['xgboost']
lr        = data['logistic_regression']
threshold = get_decision_threshold()

print("=" * 80)
print(f"ENSEMBLE DECOMPOSITION  (decision_threshold = {threshold:.4f})")
print("=" * 80)
print("\n*** 'LIVE weighted score' below comes from the REAL calculate_risk_score()")
print("*** in ml_model.py — this is what your Flask app actually uses.\n")


def score_all(feature_dict, label):
    vec = np.array([[feature_dict.get(c, 0) for c in cols]])
    scaled = scaler.transform(vec)

    p_rf  = rf.predict_proba(scaled)[0][1]
    p_xgb = xgb.predict_proba(scaled)[0][1]
    p_lr  = lr.predict_proba(scaled)[0][1]
    simple_avg = (p_rf + p_xgb + p_lr) / 3

    live_score = calculate_risk_score(feature_dict)
    live_level = classify_risk(live_score)

    print(f"\n{label}")
    print(f"  Random Forest:          {p_rf:.6f}")
    print(f"  XGBoost:                {p_xgb:.6f}")
    print(f"  Logistic Regression:    {p_lr:.6f}")
    print(f"  Old unweighted avg:     {simple_avg:.6f}  (NOT used by the app anymore)")
    print(f"  -- LIVE weighted score: {live_score:.6f}  {live_level}  "
          f"(this is what app.py actually sees)")
    return p_rf, p_xgb, p_lr, live_score


BASELINE = {
    'device_consistency': 1.0, 'type_encoded': 4, 'sim_age_days': 180,
    'new_account_flag': 0, 'hour_of_day': 14, 'is_night_transaction': 0,
    'is_peak_hours': 1, 'amount_log': 6.2, 'amount_z_score': 0.0,
    'amount_anomaly': 0, 'balance_change_initiator': 500.0,
    'balance_change_recipient': 500.0, 'balance_change_ratio': 0.1,
    'zero_balance_initiator': 0, 'zero_balance_recipient': 0,
    'transaction_velocity': 2, 'high_risk_type': 0,
    'balance_discrepancy': 0.0, 'balance_discrepancy_ratio': 0.0,
}
score_all(BASELINE, "1. Baseline -- legitimate, normal login")

attack_device_only = {**BASELINE, 'device_consistency': 0.0}
score_all(attack_device_only, "2. Only device_consistency = 0 (everything else baseline)")

realistic_attack = {
    'device_consistency': 0.0,
    'type_encoded': 0,
    'sim_age_days': 1,
    'new_account_flag': 1,
    'hour_of_day': 3,
    'is_night_transaction': 1,
    'is_peak_hours': 0,
    'amount_log': float(np.log1p(400000)),
    'amount_z_score': 1.5,
    'amount_anomaly': 0,
    'balance_change_initiator': 380000,
    'balance_change_recipient': 220000,
    'balance_change_ratio': 0.9,
    'zero_balance_initiator': 1,
    'zero_balance_recipient': 0,
    'transaction_velocity': 95,
    'high_risk_type': 1,
    'balance_discrepancy': 80000,
    'balance_discrepancy_ratio': 0.8,
}
score_all(realistic_attack, "3. Realistic attacker -- values WITHIN training distribution range")

extreme_attack = {
    'device_consistency': 0.0, 'type_encoded': 0, 'sim_age_days': 1,
    'new_account_flag': 1, 'hour_of_day': 3, 'is_night_transaction': 1,
    'is_peak_hours': 0, 'amount_log': 16.0, 'amount_z_score': 5.0,
    'amount_anomaly': 1, 'balance_change_initiator': 9000000,
    'balance_change_recipient': 9000000, 'balance_change_ratio': 0.99,
    'zero_balance_initiator': 1, 'zero_balance_recipient': 0,
    'transaction_velocity': 100, 'high_risk_type': 1,
    'balance_discrepancy': 500000, 'balance_discrepancy_ratio': 8.0,
}
score_all(extreme_attack, "4. Extreme out-of-distribution attack (9,000,000 transfer)")

print("\n" + "=" * 80)
print("CHECKLIST")
print("=" * 80)
print("""
[ ] Scenario 1 (baseline) LIVE weighted score should be LOW
[ ] Scenario 3 (realistic attacker) LIVE weighted score should be MEDIUM or HIGH
[ ] LIVE weighted score should DIFFER from "Old unweighted avg" on scenario 3/4
    (if they're identical, ml_model.py's calculate_risk_score() still isn't
    using the weighted logic -- double check you copied the UPDATED
    ml_model.py from the most recent message into your project folder,
    overwriting the old one, and that there's no leftover __pycache__
    using a stale compiled version)
""")