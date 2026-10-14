"""
auth.py
Four-layer authentication engine for the secured SMS system.
Layer 1: Device Fingerprinting (real client-collected signals)
Layer 2: SIM Binding Check (server-side state only, never client input)
Layer 3: OTP Validation (TOTP algorithm, delivered via simulated SMS)
Layer 4: ML Anomaly Detection

Key fixes vs the original version:
- Device fingerprint now comes from the client (JS-collected hash) and is
  checked against a *set* of known devices per user, not a single slot
  derived from volatile request headers/IP.
- SIM binding never reads from the login form. The registered_sim_id is
  set once at registration and can only change via simulate_sim_swap(),
  which stands in for a carrier-side event in this academic deployment.
- The "step-up" path (unknown device) no longer skips the SIM-binding
  check — it now runs SIM binding too, then escalates to OTP either way.
"""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from flask import request
from models import db, User, DeviceFingerprint, AuthenticationLog, OTPRequest, EmailChallenge
from ml_model import calculate_risk_score, classify_risk, extract_features, extract_transaction_features
from otp import match_otp_step, generate_otp, simulate_sms, simulate_email, OTP_VALIDITY_SECONDS
import bcrypt


# ── CONSTANTS ────────────────────────────────────────────────────────────────
MAX_FAILED_ATTEMPTS = 3
LOCKOUT_MINUTES     = 30
EMAIL_CODE_VALIDITY_SECONDS = 300   # unknown-device email step-up code


# ── LAYER 0: CREDENTIAL VERIFICATION ─────────────────────────────────────────
def verify_credentials(username: str, password: str):
    """
    Verify username and password against the database.
    Returns (User object, error message) tuple.
    """
    user = db.session.query(User).filter_by(username=username).first()

    if not user:
        # Run the hash check anyway against a dummy hash to keep timing
        # roughly constant whether or not the username exists. This is a
        # small mitigation against username-enumeration via timing.
        bcrypt.checkpw(password.encode('utf-8'),
                        bcrypt.hashpw(b'dummy', bcrypt.gensalt()))
        return None, "Invalid username or password"

    if user.is_locked:
        return None, "Account is temporarily locked due to multiple failed attempts"

    password_valid = bcrypt.checkpw(
        password.encode('utf-8'),
        user.password_hash.encode('utf-8')
    )

    if not password_valid:
        user.failed_attempts += 1
        if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
            user.is_locked = True
        db.session.commit()
        return None, "Invalid username or password"

    return user, None


# ── LAYER 1: DEVICE FINGERPRINTING ───────────────────────────────────────────
# NOTE: the fingerprint itself is now computed client-side (see
# static/js/fingerprint.js) and submitted as a form field `device_fp`.
# This function only validates that something was actually submitted.
def validate_fingerprint_submission(submitted_fp: str) -> bool:
    """Basic sanity check that a client-side fingerprint hash was sent."""
    return bool(submitted_fp) and len(submitted_fp) == 64  # sha256 hex length


def check_device_fingerprint(user: User, current_fp: str) -> tuple:
    """
    Compare the submitted device fingerprint against the user's known
    device set (not a single stored value), so a legitimate second device
    doesn't get treated identically to an attacker's device.

    Returns (passed: bool, message: str, consistency_score: float)
    """
    known_devices = db.session.query(DeviceFingerprint).filter_by(user_id=user.id).all()

    if not known_devices:
        # First login ever — register this as the trusted baseline device.
        new_device = DeviceFingerprint(
            user_id=user.id,
            fingerprint=current_fp,
            label='Initial registered device',
        )
        db.session.add(new_device)
        db.session.commit()
        return True, "Device registered", 1.0

    for device in known_devices:
        if hmac.compare_digest(device.fingerprint, current_fp):
            device.last_seen = datetime.utcnow()
            db.session.commit()
            return True, "Device verified", 1.0

    # Unknown device — do NOT auto-trust it. Caller decides whether to
    # step up to OTP. consistency_score is a continuous signal fed to the
    # ML layer rather than the previous fixed 0.25.
    return False, "Unrecognized device detected", 0.2


def add_trusted_device(user: User, fingerprint: str, label: str = None):
    """Call this once a step-up flow has fully succeeded (email code AND SMS
    OTP AND the ML risk check) - promotes the new fingerprint to trusted."""
    existing = db.session.query(DeviceFingerprint).filter_by(
        user_id=user.id, fingerprint=fingerprint).first()
    if existing:
        existing.last_seen = datetime.utcnow()
        existing.trusted = True
    else:
        db.session.add(DeviceFingerprint(
            user_id=user.id, fingerprint=fingerprint,
            label=label or 'Verified via step-up'
        ))
    db.session.commit()


# ── LAYER 2: SIM BINDING CHECK ────────────────────────────────────────────────
def check_sim_binding(user: User) -> tuple:
    """
    Verify SIM binding status using ONLY server-side state. The SIM ID is
    never accepted from the login request — it is set at registration and
    can only change via simulate_sim_swap() (standing in for a carrier
    webhook in production). This closes the bypass where a submitted
    sim_id field could be used to defeat the check.

    Returns (passed: bool, message: str)
    """
    if user.sim_swap_flagged:
        return False, "SIM swap detected — access denied"

    if not user.registered_sim_id:
        # Should not normally happen post-registration, but handle gracefully.
        return False, "No SIM bound to this account — contact support"

    return True, "SIM binding verified"


def simulate_sim_swap(user: User):
    """
    Stands in for a carrier-side SIM-swap notification, since real HLR /
    carrier API access isn't available in this academic deployment. Call
    this from an admin/testing route to demonstrate the SIM-swap-detection
    path during defense without needing live telecom integration.
    """
    user.sim_swap_flagged = True
    user.sim_swap_flagged_at = datetime.utcnow()
    db.session.commit()


def clear_sim_swap_flag(user: User, new_sim_id: str = None):
    """Administrative recovery path: clears the swap flag and optionally
    re-binds a new (verified) SIM id to the account."""
    user.sim_swap_flagged = False
    user.sim_swap_flagged_at = None
    if new_sim_id:
        user.registered_sim_id = new_sim_id
        user.sim_registered_at = datetime.utcnow()
    db.session.commit()


# ── LAYER 3: OTP SENDING AND VALIDATION ───────────────────────────────────────
def send_otp(user: User) -> str:
    """
    Generate and send OTP to the user's registered phone number.
    Returns the generated OTP code (useful for the simulated-SMS console
    output; never return this to the client in a real deployment).
    """
    otp_code = generate_otp(user.totp_secret)
    simulate_sms(user.phone_number, otp_code)

    new_request = OTPRequest(
        user_id=user.id,
        requested_at=datetime.utcnow(),
        used=False
    )
    db.session.add(new_request)
    db.session.commit()

    return otp_code


def _otp_failure(user: User, message: str) -> tuple:
    """Record a failed OTP attempt (and lock the account at the limit)."""
    user.failed_attempts += 1
    if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
        user.is_locked = True
    db.session.commit()
    return False, message


def verify_otp_code(user: User, submitted_otp: str) -> tuple:
    """
    Validate the OTP submitted by the user.

    A code is accepted only if ALL of the following hold:
      1. it is a 6-digit string;
      2. there is an unused OTP request for this user issued within the last
         OTP_VALIDITY_SECONDS (so a code cannot be used without having been
         requested, and cannot outlive its request);
      3. it matches the user's TOTP for the current step +/- 1; and
      4. its time step is newer than the last step already accepted
         (single use: a replayed or re-submitted code is rejected).
    On success the request(s) are consumed and the step is recorded.

    Returns (passed: bool, message: str)
    """
    if not submitted_otp or len(submitted_otp) != 6 or not submitted_otp.isdigit():
        return False, "Invalid OTP format"

    active = (OTPRequest.query
              .filter_by(user_id=user.id, used=False)
              .order_by(OTPRequest.requested_at.desc())
              .first())
    if active is None:
        return _otp_failure(user, "Invalid or expired OTP")
    age = (datetime.utcnow() - active.requested_at).total_seconds()
    if age > OTP_VALIDITY_SECONDS:
        return _otp_failure(user, "Invalid or expired OTP")

    step = match_otp_step(user.totp_secret, submitted_otp)
    if step is None:
        return _otp_failure(user, "Invalid or expired OTP")
    if user.last_otp_step is not None and step <= user.last_otp_step:
        return _otp_failure(user, "OTP already used")

    # Success: consume this code and invalidate any other pending requests.
    OTPRequest.query.filter_by(user_id=user.id, used=False).update({'used': True})
    user.last_otp_step = step
    user.failed_attempts = 0
    db.session.commit()
    return True, "OTP verified"


# ── LAYER 1 STEP-UP: EMAIL CHALLENGE (non-SMS channel) ────────────────────────
def _email_code_hash(user_id: int, code: str) -> str:
    return hashlib.sha256(f"{user_id}:{code}".encode('utf-8')).hexdigest()


def send_email_challenge(user: User) -> str:
    """
    Issue a 6-digit code to the user's registered email address. Any earlier
    unused challenge is invalidated. Returns the code only so the simulated
    email (console) and the tests can use it; never return it to the client.
    """
    EmailChallenge.query.filter_by(user_id=user.id, used=False).update({'used': True})
    code = f"{secrets.randbelow(10 ** 6):06d}"
    db.session.add(EmailChallenge(
        user_id=user.id,
        code_hash=_email_code_hash(user.id, code),
        created_at=datetime.utcnow(),
        used=False,
    ))
    db.session.commit()
    simulate_email(user.email, code)
    return code


def verify_email_code(user: User, submitted_code: str) -> tuple:
    """
    Validate the email step-up code. Accepted only if it is a 6-digit string,
    there is an unused challenge issued within EMAIL_CODE_VALIDITY_SECONDS,
    and the hash matches. The challenge is consumed on success (single use).
    Failures count towards the same lockout counter as OTP failures.
    Returns (passed: bool, message: str)
    """
    if not submitted_code or len(submitted_code) != 6 or not submitted_code.isdigit():
        return False, "Invalid code format"

    challenge = (EmailChallenge.query
                 .filter_by(user_id=user.id, used=False)
                 .order_by(EmailChallenge.created_at.desc())
                 .first())
    if challenge is None:
        return _otp_failure(user, "Invalid or expired code")
    age = (datetime.utcnow() - challenge.created_at).total_seconds()
    if age > EMAIL_CODE_VALIDITY_SECONDS:
        return _otp_failure(user, "Invalid or expired code")
    if not hmac.compare_digest(challenge.code_hash,
                               _email_code_hash(user.id, submitted_code)):
        return _otp_failure(user, "Invalid or expired code")

    challenge.used = True
    db.session.commit()
    return True, "Email verified"


# ── LAYER 4: ML ANOMALY DETECTION ────────────────────────────────────────────
def run_anomaly_detection(user: User, session_context: dict) -> tuple:
    """
    Run the trained Voting Classifier ensemble to calculate risk score
    AT LOGIN TIME. Returns (passed: bool, risk_score: float, risk_level: str, message: str)
    """
    features   = extract_features(session_context)
    risk_score = calculate_risk_score(features)
    risk_level = classify_risk(risk_score)

    if risk_level == 'HIGH':
        return False, risk_score, risk_level, "Anomalous behaviour detected"
    elif risk_level == 'MEDIUM':
        return True, risk_score, risk_level, "Medium risk — step-up verification recommended"
    else:
        return True, risk_score, risk_level, "Authentication successful"


def run_transaction_risk_check(transaction_data: dict, recent_transaction_count: int,
                                 device_consistency: float = 1.0) -> tuple:
    """
    Run the same trained ensemble at TRANSACTION time, using real values
    derived from an actual logged transfer attempt rather than login-time
    constants. This is the second checkpoint in the two-stage risk
    architecture (login risk, then transaction/takeover risk) — the
    point where rapid, unusual, or large transactions immediately after
    a takeover are meant to be caught, matching what the training
    dataset (Azamuke's Synthetic Mobile Money Transaction Dataset) was
    actually built to represent.

    Returns (passed: bool, risk_score: float, risk_level: str, message: str)
    """
    features   = extract_transaction_features(transaction_data, recent_transaction_count,
                                                 device_consistency)
    risk_score = calculate_risk_score(features)
    risk_level = classify_risk(risk_score)

    if risk_level == 'HIGH':
        return False, risk_score, risk_level, "Transaction blocked — pattern consistent with account takeover"
    elif risk_level == 'MEDIUM':
        return True, risk_score, risk_level, "Transaction flagged as medium risk — monitor"
    else:
        return True, risk_score, risk_level, "Transaction approved"


# ── LOGGING ───────────────────────────────────────────────────────────────────
def log_attempt(user, decision: str, reason: str = None,
                layer: str = None, risk_score: float = None,
                risk_level: str = None, device_fp: str = None):
    """Log every authentication attempt to the database."""
    log = AuthenticationLog(
        user_id        = user.id if user else None,
        username       = user.username if user else 'unknown',
        ip_address     = request.remote_addr,
        device_fp      = device_fp,
        risk_score     = risk_score,
        risk_level     = risk_level,
        decision       = decision,
        failure_reason = reason,
        layer_failed   = layer
    )
    db.session.add(log)
    db.session.commit()