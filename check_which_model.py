"""
check_which_model.py
Run this BEFORE running model_behaviour.py / probe_model_behavior.py,
in the SAME directory and SAME terminal session, to confirm exactly
which trained_voting_model.pkl file is being loaded and whether it's
the one your retrain just produced.

Run with: python check_which_model.py
"""

import os
import pickle
import datetime

MODEL_PATH = os.path.join('posttest_result', 'trained_voting_model.pkl')

print("=" * 70)
print("MODEL FILE DIAGNOSTIC")
print("=" * 70)

print(f"\nCurrent working directory:")
print(f"  {os.getcwd()}")

print(f"\nResolved MODEL_PATH (relative to above):")
print(f"  {os.path.abspath(MODEL_PATH)}")

if not os.path.exists(MODEL_PATH):
    print(f"\n  *** FILE NOT FOUND AT THIS PATH ***")
    print(f"  This means model_behaviour.py would either crash, or — if")
    print(f"  you're running from a different directory — it may be silently")
    print(f"  finding a DIFFERENT trained_voting_model.pkl elsewhere on your")
    print(f"  system (e.g. via a cached sys.path import). Run this script")
    print(f"  from the exact folder where you run model_behaviour.py.")
else:
    mtime = os.path.getmtime(MODEL_PATH)
    mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    size_kb = os.path.getsize(MODEL_PATH) / 1024

    print(f"\n  File found.")
    print(f"  Last modified: {mtime_str}")
    print(f"  File size:     {size_kb:,.1f} KB")

    now = datetime.datetime.now()
    age_minutes = (now - datetime.datetime.fromtimestamp(mtime)).total_seconds() / 60
    print(f"  Age:           {age_minutes:.1f} minutes ago")

    if age_minutes > 60:
        print(f"\n  *** WARNING: this file is over an hour old. ***")
        print(f"  If you retrained recently, this is probably NOT your new model.")

    # Load it and check the threshold + a fingerprint of the scaler stats
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)

    threshold = data.get('decision_threshold')
    print(f"\n  decision_threshold: {threshold}  (type: {type(threshold).__name__})")

    if threshold is None:
        print(f"\n  *** This is the OLD model (or the notebook fix wasn't applied). ***")
        print(f"  A correctly retrained model should have a float threshold,")
        print(f"  e.g. 0.2119, not None.")
    elif isinstance(threshold, float) and 0 < threshold < 1:
        print(f"\n  This LOOKS like a correctly retrained model (valid float threshold).")
    else:
        print(f"\n  *** Unexpected threshold value/type — investigate. ***")

    # Check for amount_stats — only present in the new train_model.py output
    has_amount_stats = 'amount_stats' in data
    print(f"\n  Has 'amount_stats' key (new training script only): {has_amount_stats}")
    if not has_amount_stats:
        print(f"  *** This pkl was NOT produced by the updated train_model.py. ***")
        print(f"  It's either the original notebook's output, or an even older file.")

    scaler = data['scaler']
    dc_idx = data['feature_columns'].index('device_consistency')
    print(f"\n  device_consistency training mean: {scaler.mean_[dc_idx]:.6f}")
    print(f"  (Compare this to previous runs — if it's EXACTLY 0.1585 again,")
    print(f"   you're loading the same file as before.)")

print("\n" + "=" * 70)