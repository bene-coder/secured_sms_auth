"""
stress_test.py
═══════════════════════════════════════════════════════════════════════════════
FRONT-END STRESS TEST — Secured SMS Authentication System

Drives the full HTTP flow through Flask's test client, covering six scenarios
that between them exercise every layer, every denial path, and the ML risk
model under realistic conditions. Each scenario is independent (fresh session,
fresh in-memory database) so a failure in one doesn't poison the others.

These are the scenarios a defense panel is most likely to ask you to
demonstrate live. Run this after every significant code change to confirm
nothing regressed.

Run with:   python stress_test.py
Expected:   All scenarios PASS

Scenarios:
  1. Legitimate user — full happy path (all layers pass, transfer approved)
  2. Wrong password — blocked at Layer 0
  3. Account lockout — 3 wrong passwords lock the account
  4. SIM swap attack — attacker blocked at Layer 2 after swap simulation
  5. Unknown device — step-up required, then promoted after OTP success
  6. Rapid-fire transfers — ML risk score escalates with transaction velocity
"""

import os
import sys
import pyotp

os.environ.setdefault('FLASK_SECRET_KEY', 'stress-test-key')
os.environ.setdefault('DEMO_USER_PASSWORD', 'StressTest123!')

# ── Bootstrap ────────────────────────────────────────────────────────────────
from app import app, init_db, load_model
from models import db, User, AuthenticationLog, TransactionLog
from auth import simulate_sim_swap, add_trusted_device

# Use a single shared in-memory DB for the whole run, but properly DROP
# and RECREATE every table between scenarios. Previously each scenario
# reassigned SQLALCHEMY_DATABASE_URI to a new ':memory:' string, but
# Flask-SQLAlchemy caches the underlying engine/connection on first use,
# so all "fresh" databases were actually silently sharing the SAME
# in-memory database and accumulating state across scenarios — which is
# why scenario 2 found a GRANT log left over from scenario 1.
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'

RESULTS = []


def reset_database():
    """Drop and recreate every table — true isolation between scenarios."""
    with app.app_context():
        db.drop_all()
        db.create_all()
        init_db()


def scenario(name):
    """Decorator — runs each scenario against a freshly wiped database."""
    def decorator(fn):
        def wrapper():
            print(f"\n{'─'*70}")
            print(f"SCENARIO: {name}")
            print('─'*70)
            checks = []

            reset_database()
            with app.app_context():
                client = app.test_client()
                fn(client, checks)

            passed = all(ok for _, ok, _ in checks)
            RESULTS.append((name, passed, checks))
            for label, ok, detail in checks:
                status = 'PASS' if ok else 'FAIL'
                print(f"  [{status}] {label}" +
                      (f"\n         ↳ {detail}" if detail and not ok else ""))
            print(f"  → {'ALL PASS' if passed else 'SCENARIO FAILED'}")
        return wrapper
    return decorator


def check(checks, label, condition, detail=""):
    checks.append((label, bool(condition), detail))


def get_fresh_otp():
    """Pull the current TOTP code for the demo user directly from the DB secret.
    Must be called from WITHIN an active app context (the scenario wrapper
    already provides one) — previously this took an app_ctx argument and
    opened a SECOND nested context, which on some Flask-SQLAlchemy versions
    can desync session state from the outer context's connection."""
    user = User.query.filter_by(username='demo_user').first()
    return pyotp.TOTP(user.totp_secret).now(), user.id


def login_step(client, username='demo_user', password='StressTest123!',
               device_fp='a'*64):
    return client.post('/login', data={
        'username': username,
        'password': password,
        'device_fp_hash': device_fp,
    }, follow_redirects=True)


def otp_step(client, otp_code):
    return client.post('/verify-otp', data={'otp': otp_code},
                       follow_redirects=True)


def transfer_step(client, amount, recipient='Test Recipient'):
    return client.post('/transfer', data={
        'amount': str(amount),
        'recipient_label': recipient,
    }, follow_redirects=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1: Legitimate user — full happy path
# Expected: All layers pass, reaches dashboard, transfer approved
# ═══════════════════════════════════════════════════════════════════════════════
@scenario("Legitimate user — full happy path")
def scenario_happy_path(client, checks):
    # Layer 0 + 1 + 2: credential check + known device + SIM OK
    r = login_step(client, device_fp='a'*64)
    check(checks, "Login POST doesn't 500", r.status_code == 200)
    check(checks, "Redirected to OTP page (not denied)", b'verify' in r.request.url.encode()
          or b'Enter your code' in r.data or b'countdown' in r.data,
          f"URL was: {r.request.url}")

    # Layer 3: OTP
    otp, uid = get_fresh_otp()
    r = otp_step(client, otp)
    check(checks, "OTP accepted, reaches dashboard",
          b'Welcome' in r.data or b'dashboard' in r.request.url.encode(),
          f"URL: {r.request.url}")

    # Layer 4: small legitimate transfer
    r = transfer_step(client, 5000, 'Legitimate recipient')
    check(checks, "Small legitimate transfer not blocked",
          b'Access denied' not in r.data,
          "Transfer was blocked unexpectedly")

    # Confirm AuthenticationLog has a GRANT entry
    log = AuthenticationLog.query.filter_by(decision='GRANT').first()
    check(checks, "GRANT recorded in AuthenticationLog",
          log is not None)
    txn = TransactionLog.query.filter_by(decision='GRANT').first()
    check(checks, "Transfer logged with GRANT decision",
          txn is not None)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2: Wrong password — blocked at Layer 0
# Expected: Login denied, user stays on login page, no session created
# ═══════════════════════════════════════════════════════════════════════════════
@scenario("Wrong password — blocked at Layer 0")
def scenario_wrong_password(client, checks):
    r = login_step(client, password='CompletelyWrong!')
    check(checks, "Wrong password: stays on login page (not dashboard)",
          b'Welcome' not in r.data and b'dashboard' not in r.request.url.encode())
    check(checks, "Wrong password: no OTP page shown",
          b'countdown' not in r.data and b'Enter your code' not in r.data)
    check(checks, "Wrong password: error message shown",
          b'Invalid' in r.data or b'danger' in r.data,
          "No error message found in response")

    # Confirm no GRANT in logs
    grant = AuthenticationLog.query.filter_by(decision='GRANT').first()
    check(checks, "No GRANT logged for wrong-password attempt",
          grant is None)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 3: Account lockout — 3 wrong passwords, then correct password blocked
# Expected: Account locks after 3 failures; correct password still denied
# ═══════════════════════════════════════════════════════════════════════════════
@scenario("Account lockout — 3 failures then locked")
def scenario_lockout(client, checks):
    for i in range(3):
        login_step(client, password='WrongOne!')

    user = User.query.filter_by(username='demo_user').first()
    check(checks, "Account is_locked=True after 3 failures",
          user.is_locked is True,
          f"is_locked={user.is_locked}, failed_attempts={user.failed_attempts}")

    # Now try the CORRECT password — must still be denied
    r = login_step(client, password='StressTest123!')
    check(checks, "Correct password denied when account is locked",
          b'Welcome' not in r.data,
          "Account was unlocked by correct password — lockout bypass!")
    check(checks, "Lockout message shown",
          b'locked' in r.data.lower() or b'danger' in r.data)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 4: SIM swap attack
# Expected: Attacker blocked at Layer 2 after swap is flagged
# ═══════════════════════════════════════════════════════════════════════════════
@scenario("SIM swap attack — blocked at Layer 2")
def scenario_sim_swap(client, checks):
    # First simulate a carrier-reported SIM swap
    user = User.query.filter_by(username='demo_user').first()
    simulate_sim_swap(user)
    check(checks, "sim_swap_flagged set to True",
          user.sim_swap_flagged is True)

    # Now try to login — should be denied at Layer 2
    r = login_step(client)
    check(checks, "Login denied after SIM swap (Layer 2)",
          b'Access denied' in r.data or b'SIM' in r.data,
          f"Expected denial page, got: {r.request.url}")
    check(checks, "No OTP page reached (denied before Layer 3)",
          b'countdown' not in r.data and b'Enter your code' not in r.data)
    check(checks, "Denial shows Layer 2 context",
          b'Layer 2' in r.data or b'SIM' in r.data)

    # Confirm DENY logged with layer_failed = LAYER_2
    log = AuthenticationLog.query.filter_by(
        decision='DENY', layer_failed='LAYER_2'
    ).first()
    check(checks, "DENY/LAYER_2 recorded in AuthenticationLog",
          log is not None)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 5: Unknown device — step-up, then promoted
# Expected: Unknown FP triggers step-up; OTP success promotes device;
#           second login from same FP goes straight through without step-up
# ═══════════════════════════════════════════════════════════════════════════════
@scenario("Unknown device — step-up then promoted")
def scenario_unknown_device(client, checks):
    UNKNOWN_FP = 'f' * 64   # never-seen-before fingerprint

    # Login with unknown device — should still reach OTP (step-up)
    r = login_step(client, device_fp=UNKNOWN_FP)
    check(checks, "Unknown device reaches OTP page (step-up, not hard denial)",
          b'Enter your code' in r.data or b'countdown' in r.data,
          f"URL: {r.request.url}")

    # OTP success — device should be promoted
    otp, uid = get_fresh_otp()
    r = otp_step(client, otp)
    check(checks, "Step-up OTP accepted, reaches dashboard",
          b'Welcome' in r.data,
          f"URL: {r.request.url}")

    # Confirm device was added to DeviceFingerprint table
    from models import DeviceFingerprint
    fp_row = DeviceFingerprint.query.filter_by(
        fingerprint=UNKNOWN_FP
    ).first()
    check(checks, "Unknown device promoted to trusted after step-up OTP",
          fp_row is not None and fp_row.trusted is True,
          "DeviceFingerprint row not found after step-up")

    # Logout, then login again with the same FP — should NOT trigger step-up now
    client.get('/logout')
    r = login_step(client, device_fp=UNKNOWN_FP)
    check(checks, "Previously-unknown device now goes straight to OTP (no step-up flash)",
          b'Security Alert' not in r.data,
          "Device still triggering step-up after being promoted")


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 6: Rapid-fire transfers — ML risk score escalates with velocity
# Expected: Risk score increases with each successive transfer;
#           score for a large transfer at high velocity > small single transfer
# ═══════════════════════════════════════════════════════════════════════════════
@scenario("Rapid-fire transfers — ML risk escalates with velocity")
def scenario_rapid_transfers(client, checks):
    # Full login first
    login_step(client)
    otp, _ = get_fresh_otp()
    otp_step(client, otp)

    # Single small transfer — baseline
    transfer_step(client, 1000, 'Baseline transfer')

    t1 = TransactionLog.query.order_by(
        TransactionLog.timestamp.asc()
    ).first()
    baseline_score = t1.risk_score if t1 else 0.0
    check(checks, "Baseline transfer logged with a risk score",
          t1 is not None and t1.risk_score is not None)

    # Now fire 6 rapid large transfers
    for i in range(6):
        transfer_step(client, 2000000 + i * 500000,
                      f'Rapid transfer {i+1}')

    txns = TransactionLog.query.order_by(
        TransactionLog.timestamp.asc()
    ).all()
    scores = [t.risk_score for t in txns if t.risk_score is not None]
    check(checks, "All transfers logged with risk scores",
          len(scores) == 7,
          f"Expected 7 logged transfers, got {len(scores)}")

    # The last high-velocity, high-amount transfer should score
    # higher than the first small baseline transfer
    if len(scores) >= 2:
        check(checks,
              "High-velocity large transfer scores higher than baseline",
              scores[-1] > scores[0],
              f"Last={scores[-1]:.6f}  Baseline={scores[0]:.6f}")

    # Print the score progression so you can see the trend
    print("\n  Transfer risk score progression:")
    for i, t in enumerate(txns):
        level = t.risk_level or 'N/A'
        print(f"    #{i+1}  amount=₦{t.amount:>12,.0f}  "
              f"risk={t.risk_score:.6f}  {level}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 70)
    print("STRESS TEST — Secured SMS Authentication System")
    print("=" * 70)

    load_model()

    scenario_happy_path()
    scenario_wrong_password()
    scenario_lockout()
    scenario_sim_swap()
    scenario_unknown_device()
    scenario_rapid_transfers()

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total   = len(RESULTS)
    passed  = sum(1 for _, ok, _ in RESULTS if ok)
    failed  = [(name, checks) for name, ok, checks in RESULTS if not ok]

    for name, ok, _ in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    print(f"\n  {passed}/{total} scenarios passed")

    if failed:
        print("\nFAILED CHECKS:")
        for name, checks in failed:
            print(f"\n  Scenario: {name}")
            for label, ok, detail in checks:
                if not ok:
                    print(f"    ✗ {label}")
                    if detail:
                        print(f"      {detail}")
        sys.exit(1)
    else:
        print("\n  All scenarios passed.")
        print("  The system is behaving correctly across all tested attack")
        print("  and legitimate-user paths. Safe to demo.")
        sys.exit(0)