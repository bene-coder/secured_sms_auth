"""
test_layers.py
Direct functional tests for each authentication layer's actual decision
logic — not just "does the page render", but "does the layer correctly
GRANT in the legitimate case and DENY in the attack/edge case".

Run with: python test_layers.py

Uses an in-memory SQLite database so this never touches your real
auth_system.db. Each test prints PASS/FAIL and a one-line reason, plus a
summary count at the end. Treat any FAIL as a real bug to fix before
your demo — these are the same kinds of checks an examiner might
informally try by hand, just automated and repeatable.
"""

import sys
from datetime import datetime, timedelta, timezone

from flask import Flask
from models import db, User, DeviceFingerprint, OTPRequest
from auth import (verify_credentials, check_device_fingerprint,
                   add_trusted_device, check_sim_binding, simulate_sim_swap,
                   clear_sim_swap_flag, send_otp, verify_otp_code)
from otp import generate_secret, get_otp_remaining_seconds
import bcrypt

# ── TEST APP / DB SETUP ────────────────────────────────────────────────
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'test-only'
db.init_app(app)

results = []  # (name, passed: bool, detail: str)


def check(name, condition, detail=""):
    results.append((name, bool(condition), detail))
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))


def make_user(username="testuser", password="CorrectPass1!", phone="+2348011111111", sim="SIM_TEST_001"):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user = User(
        username=username, email=f"{username}@example.com",
        password_hash=hashed, phone_number=phone,
        totp_secret=generate_secret(),
        registered_sim_id=sim, sim_registered_at=datetime.utcnow(),
    )
    db.session.add(user)
    db.session.commit()
    return user


# ════════════════════════════════════════════════════════════════════════
print("\n=== LAYER 0: Credential verification ===")
with app.app_context():
    db.create_all()
    user = make_user()

    print("Test: correct password is accepted")
    u, err = verify_credentials("testuser", "CorrectPass1!")
    check("Correct password accepted", u is not None and err is None)

    print("Test: wrong password is rejected")
    u, err = verify_credentials("testuser", "WrongPassword!")
    check("Wrong password rejected", u is None and err is not None)

    print("Test: nonexistent username is rejected (no user enumeration leak)")
    u, err = verify_credentials("nobody_here", "anything")
    check("Unknown username rejected", u is None and err is not None)

    print("Test: account locks after MAX_FAILED_ATTEMPTS wrong passwords")
    fresh = make_user(username="locktest", password="RightPass1!")
    for i in range(3):
        verify_credentials("locktest", "WrongOne!")
    locked_user = User.query.filter_by(username="locktest").first()
    check("Account locks after repeated failures", locked_user.is_locked is True,
          f"is_locked={locked_user.is_locked} after 3 bad attempts")

    print("Test: locked account rejects even the CORRECT password")
    u, err = verify_credentials("locktest", "RightPass1!")
    check("Locked account rejects correct password too", u is None,
          "this matters: lockout must not be bypassable just by knowing the password")


# ════════════════════════════════════════════════════════════════════════
print("\n=== LAYER 1: Device fingerprinting ===")
with app.app_context():
    user = User.query.filter_by(username="testuser").first()
    fp_a = "a" * 64  # pretend client hash, device A
    fp_b = "b" * 64  # a different device

    print("Test: first-ever login auto-registers the device as trusted")
    passed, msg, consistency = check_device_fingerprint(user, fp_a)
    check("First login auto-trusts device", passed is True and consistency == 1.0, msg)

    print("Test: same device on second login is recognized (no false step-up)")
    passed, msg, consistency = check_device_fingerprint(user, fp_a)
    check("Returning known device recognized", passed is True, msg)

    print("Test: a genuinely different device fingerprint triggers step-up")
    passed, msg, consistency = check_device_fingerprint(user, fp_b)
    check("Unknown device triggers step-up (passed=False)", passed is False, msg)
    check("Unknown device gets a low consistency score", consistency < 0.5,
          f"consistency={consistency}")

    print("Test: after add_trusted_device, that second device is now recognized too")
    add_trusted_device(user, fp_b, label="second device")
    passed, msg, consistency = check_device_fingerprint(user, fp_b)
    check("Promoted device now passes", passed is True, msg)

    print("Test: a third, still-unseen device still triggers step-up")
    fp_c = "c" * 64
    passed, msg, consistency = check_device_fingerprint(user, fp_c)
    check("Third unseen device still triggers step-up", passed is False, msg)


# ════════════════════════════════════════════════════════════════════════
print("\n=== LAYER 2: SIM binding (no client input accepted) ===")
with app.app_context():
    user = make_user(username="simtest", sim="SIM_ORIGINAL")

    print("Test: normal login with no SIM swap passes")
    passed, msg = check_sim_binding(user)
    check("No swap flagged -> SIM check passes", passed is True, msg)

    print("Test: simulate_sim_swap() flags the account")
    simulate_sim_swap(user)
    check("sim_swap_flagged is True after simulation", user.sim_swap_flagged is True)

    print("Test: SIM check now DENIES, with no way for client input to override it")
    passed, msg = check_sim_binding(user)
    check("Flagged swap -> SIM check denies", passed is False, msg)

    print("Test: clearing the flag restores normal access")
    clear_sim_swap_flag(user)
    passed, msg = check_sim_binding(user)
    check("Cleared swap -> SIM check passes again", passed is True, msg)

    print("Test: check_sim_binding signature takes NO client-controlled argument")
    import inspect
    sig = inspect.signature(check_sim_binding)
    param_names = list(sig.parameters.keys())
    check("check_sim_binding only accepts 'user', nothing client-supplied",
          param_names == ['user'], f"params={param_names}")


# ════════════════════════════════════════════════════════════════════════
print("\n=== LAYER 3: OTP generation, validation, expiry ===")
with app.app_context():
    user = make_user(username="otptest")

    print("Test: send_otp creates a logged OTPRequest row")
    code = send_otp(user)
    req = OTPRequest.query.filter_by(user_id=user.id).order_by(OTPRequest.requested_at.desc()).first()
    check("OTPRequest row created on send", req is not None)
    check("Generated code is 6 digits", len(code) == 6 and code.isdigit(), f"code={code}")

    print("Test: correct OTP code is accepted")
    passed, msg = verify_otp_code(user, code)
    check("Correct OTP accepted", passed is True, msg)

    print("Test: a wrong 6-digit code is rejected")
    fresh = make_user(username="otptest2")
    code2 = send_otp(fresh)
    wrong_code = "000000" if code2 != "000000" else "111111"
    passed, msg = verify_otp_code(fresh, wrong_code)
    check("Wrong OTP rejected", passed is False, msg)

    print("Test: malformed OTP (wrong length) is rejected before even checking TOTP")
    passed, msg = verify_otp_code(fresh, "123")
    check("Short/malformed OTP rejected", passed is False, msg)

    print("Test: countdown timer math is correct for a freshly-requested OTP")
    remaining = get_otp_remaining_seconds(datetime.utcnow(), window_seconds=90)
    check("Fresh OTP shows ~90s remaining (not ~3690 from the old TZ bug)",
          85 <= remaining <= 90, f"remaining={remaining}")

    print("Test: countdown correctly hits 0 for an OTP requested 91s ago")
    old_time = datetime.utcnow() - timedelta(seconds=91)
    remaining = get_otp_remaining_seconds(old_time, window_seconds=90)
    check("Old OTP shows 0 remaining, not negative or huge", remaining == 0,
          f"remaining={remaining}")


# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
total = len(results)
passed_count = sum(1 for _, p, _ in results if p)
failed = [(n, d) for n, p, d in results if not p]

print(f"RESULTS: {passed_count}/{total} passed")
if failed:
    print("\nFAILED CHECKS:")
    for name, detail in failed:
        print(f"  - {name}" + (f" ({detail})" if detail else ""))
    sys.exit(1)
else:
    print("All layer logic checks passed.")
    sys.exit(0)