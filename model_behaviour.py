"""
probe_model_behavior.py
Inspects HOW the trained risk model responds to its inputs — a sensitivity
analysis showing which features move the risk score, in which direction,
and by how much, plus where your LOW/MEDIUM/HIGH thresholds actually
land for realistic feature combinations.

IMPORTANT SCOPE NOTE (be precise about this in chapter 4):
This probes the MODEL's behavior in isolation, by feeding it hypothetical
feature vectors directly. It does NOT yet reflect how a real user's
actual login behavior maps to those features end-to-end, because most
of the 18 features are still hardcoded constants in app.py's
session_context (see ml_model.py's TODO markers). Use this to understand
and validate the model itself; the full pipeline rebuild is a separate,
later step.

Run with: python probe_model_behavior.py
"""

from ml_model import load_model, calculate_risk_score, classify_risk

# A realistic "baseline" login — nothing suspicious, mid-range values.
BASELINE = {
    'device_consistency':        1.0,
    'type_encoded':              3,
    'sim_age_days':               180,
    'new_account_flag':          0,
    'hour_of_day':                14,
    'is_night_transaction':       0,
    'is_peak_hours':              1,
    'amount_log':                 6.2,
    'amount_z_score':             0.0,
    'amount_anomaly':             0,
    'balance_change_initiator':   500.0,
    'balance_change_recipient':   500.0,
    'balance_change_ratio':       0.1,
    'zero_balance_initiator':     0,
    'zero_balance_recipient':     0,
    'transaction_velocity':       2,
    'high_risk_type':             0,
    'balance_discrepancy':        0.0,
    'balance_discrepancy_ratio':  0.0,
}

# For each feature, a list of alternative values to swap in one at a time,
# holding everything else at BASELINE. Edit these ranges to match what
# you actually expect a real attacker/legitimate-user scenario to produce.
PROBE_VALUES = {
    'device_consistency':  [1.0, 0.8, 0.5, 0.2, 0.0],
    'sim_age_days':        [1, 7, 30, 180, 365],
    'new_account_flag':    [0, 1],
    'hour_of_day':         [9, 14, 22, 2, 4],
    'is_night_transaction':[0, 1],
    'transaction_velocity':[1, 2, 5, 10, 20],
    'high_risk_type':      [0, 1],
    'amount_z_score':      [0.0, 1.0, 2.0, 3.0, 5.0],
    'zero_balance_recipient': [0, 1],
    'balance_discrepancy_ratio': [0.0, 0.1, 0.5, 1.0],
}


def run_baseline():
    score = calculate_risk_score(BASELINE)
    level = classify_risk(score)
    print(f"BASELINE risk score: {score:.4f}  ->  {level}\n")
    return score


def run_sensitivity_sweep():
    print("=" * 70)
    print("SENSITIVITY SWEEP — one feature varied at a time, rest at baseline")
    print("=" * 70)

    for feature, values in PROBE_VALUES.items():
        if feature not in BASELINE:
            print(f"  (skipping '{feature}' — not in feature set)")
            continue

        print(f"\n--- {feature} ---")
        for v in values:
            test_vec = dict(BASELINE)
            test_vec[feature] = v
            score = calculate_risk_score(test_vec)
            level = classify_risk(score)
            marker = " <-- baseline" if v == BASELINE[feature] else ""
            print(f"  {feature} = {v:<8}  risk = {score:.4f}  ({level}){marker}")


def find_threshold_crossings():
    """
    For continuous-ish features, do a finer sweep to find roughly where
    the score crosses 0.40 (LOW->MEDIUM) and 0.70 (MEDIUM->HIGH).
    Useful for explaining in chapter 4 exactly what triggers escalation.
    """
    print("\n" + "=" * 70)
    print("THRESHOLD CROSSING — device_consistency (finer resolution)")
    print("=" * 70)
    prev_level = None
    for i in range(0, 21):
        v = round(1.0 - (i * 0.05), 2)  # 1.00 down to 0.00
        test_vec = dict(BASELINE)
        test_vec['device_consistency'] = v
        score = calculate_risk_score(test_vec)
        level = classify_risk(score)
        flag = ""
        if prev_level is not None and level != prev_level:
            flag = f"  <-- crosses into {level} here"
        print(f"  device_consistency = {v:.2f}  risk = {score:.4f}  ({level}){flag}")
        prev_level = level


def two_factor_combinations():
    """
    Real attacks rarely move one feature in isolation. Check a few
    combined scenarios that approximate real threat patterns.
    """
    print("\n" + "=" * 70)
    print("COMBINED SCENARIOS")
    print("=" * 70)

    scenarios = {
        "Legitimate user, normal daytime login":
            dict(BASELINE),
        "Legitimate user, new phone, but everything else normal":
            {**BASELINE, 'device_consistency': 0.2},
        "Attacker: unknown device + odd hour + high velocity":
            {**BASELINE, 'device_consistency': 0.0, 'hour_of_day': 3,
             'is_night_transaction': 1, 'transaction_velocity': 15},
        "Attacker: unknown device + new account + high-risk type":
            {**BASELINE, 'device_consistency': 0.0, 'new_account_flag': 1,
             'high_risk_type': 1},
        "Edge case: known device but everything else maxed out suspicious":
            {**BASELINE, 'transaction_velocity': 20, 'high_risk_type': 1,
             'amount_z_score': 5.0, 'zero_balance_recipient': 1},
    }

    for name, vec in scenarios.items():
        score = calculate_risk_score(vec)
        level = classify_risk(score)
        print(f"\n  {name}")
        print(f"    -> risk = {score:.4f}  ({level})")


if __name__ == '__main__':
    load_model()
    print()
    run_baseline()
    run_sensitivity_sweep()
    find_threshold_crossings()
    two_factor_combinations()
    print("\nDone. Compare these against what you'd intuitively expect —")
    print("any feature that should clearly raise risk but doesn't move the")
    print("score (or moves it the wrong way) is worth flagging for the")
    print("phase-2 model rebuild.")