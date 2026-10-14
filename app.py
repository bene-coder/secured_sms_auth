"""
app.py
Main Flask application for the Secured SMS Authentication System.

Key fixes vs the original version (see chat for full rationale):
1. SIM id is NEVER read from the login form anymore. check_sim_binding()
   takes no client input at all — it only reads server-side state. This
   closes the bypass where submitting no sim_id (or the demo default)
   satisfied the check for any account.
2. Device fingerprint now comes from the client (JS-collected hash,
   submitted as device_fp_hash / device_fp_raw hidden fields), not a
   hash of request headers + IP, which was unstable across networks and
   spoofable via User-Agent.
3. The step-up (unknown device) path now ALSO runs the SIM-binding
   check before going to OTP, instead of skipping it. Previously the
   "insecure path" (unknown device) was MORE permissive than the normal
   path, which inverted the intended security model.
4. Secret key and demo password now come from environment variables.
"""

import os
from flask import (Flask, render_template, request, redirect,
                    url_for, session, flash, jsonify)
from datetime import datetime, timedelta
import bcrypt

from models import db, User, DeviceFingerprint, AuthenticationLog, OTPRequest, TransactionLog
from auth import (verify_credentials, check_device_fingerprint,
                   check_sim_binding, simulate_sim_swap, add_trusted_device,
                   send_otp, verify_otp_code, send_email_challenge, verify_email_code,
                   run_anomaly_detection,
                   run_transaction_risk_check, log_attempt)
from otp import generate_secret, get_otp_remaining_seconds
from ml_model import load_model, calculate_risk_score, classify_risk

# ── APP CONFIGURATION ────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-only-change-me')

app.config['SQLALCHEMY_DATABASE_URI']    = os.environ.get('AUTH_DB_URI', 'sqlite:///auth_system.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

db.init_app(app)


# ── DATABASE INITIALISATION ──────────────────────────────────────────────
def _ensure_last_otp_step_column():
    """
    db.create_all() does not add columns to an existing table. This adds
    users.last_otp_step to databases created before single-use OTP existed.
    """
    from sqlalchemy import inspect, text
    cols = [c['name'] for c in inspect(db.engine).get_columns('users')]
    if 'last_otp_step' not in cols:
        db.session.execute(text('ALTER TABLE users ADD COLUMN last_otp_step BIGINT'))
        db.session.commit()


def init_db():
    """Create all tables and seed a demo user if none exist."""
    with app.app_context():
        db.create_all()
        _ensure_last_otp_step_column()

        if User.query.count() == 0:
            demo_password = os.environ.get('DEMO_USER_PASSWORD', 'ChangeMe123!')
            hashed_pw = bcrypt.hashpw(
                demo_password.encode('utf-8'), bcrypt.gensalt()
            ).decode('utf-8')

            demo_user = User(
                username           = 'demo_user',
                email              = 'demo@secureauth.com',
                password_hash      = hashed_pw,
                phone_number       = '+2348012345678',
                totp_secret        = generate_secret(),
                registered_sim_id  = 'SIM_ABC123456',
                sim_registered_at  = datetime.utcnow(),
            )
            db.session.add(demo_user)
            db.session.commit()
            print(f"Demo user created: username=demo_user, password={demo_password}")


# ── ROUTES ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('registration.html')

    username     = request.form.get('username', '').strip()
    email        = request.form.get('email', '').strip()
    password     = request.form.get('password', '').strip()
    phone_number = request.form.get('phone_number', '').strip()
    sim_id       = request.form.get('sim_id', '').strip()
    device_fp    = request.form.get('device_fp_hash', '').strip()

    if not all([username, email, password, phone_number, sim_id]):
        flash('All fields are required.', 'danger')
        return render_template('registration.html')

    if User.query.filter((User.username == username) | (User.email == email)).first():
        flash('Registration Error: Username or Email already exists.', 'danger')
        return render_template('registration.html')

    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    new_user = User(
        username           = username,
        email              = email,
        password_hash      = hashed_pw,
        phone_number       = phone_number,
        totp_secret        = generate_secret(),
        registered_sim_id  = sim_id,            # bound ONCE, here, at registration
        sim_registered_at  = datetime.utcnow(),
    )
    db.session.add(new_user)
    db.session.commit()

    # Register the device fingerprint as the trusted baseline, if one was
    # submitted (it should be — the registration form includes the JS).
    if device_fp and len(device_fp) == 64:
        db.session.add(DeviceFingerprint(
            user_id=new_user.id,
            fingerprint=device_fp,
            label='Initial registered device',
        ))
        db.session.commit()

    flash('Registration successful! Your device has been registered.', 'success')
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    username  = request.form.get('username', '').strip()
    password  = request.form.get('password', '').strip()
    device_fp = request.form.get('device_fp_hash', '').strip()

    # ── LAYER 0: CREDENTIAL CHECK ──────────────────────────────────────
    user, error = verify_credentials(username, password)
    if not user:
        flash(error, 'danger')
        log_attempt(None, 'DENY', reason=error, layer='LAYER_0')
        return render_template('login.html')

    if not device_fp or len(device_fp) != 64:
        # Fingerprint JS didn't run (JS disabled, or form submitted
        # without it) — treat as unknown device rather than guessing.
        device_fp = 'unsubmitted'

    # ── LAYER 1: DEVICE FINGERPRINTING ─────────────────────────────────
    fp_passed, fp_message, consistency = check_device_fingerprint(user, device_fp)

    # ── LAYER 2: SIM BINDING CHECK ──────────────────────────────────────
    # Runs regardless of the device-fingerprint outcome now — previously
    # an unknown device skipped this check entirely, which was backwards.
    sim_passed, sim_message = check_sim_binding(user)

    if not sim_passed:
        flash(f'Security Alert: {sim_message}', 'danger')
        log_attempt(user, 'DENY', reason=sim_message, layer='LAYER_2', device_fp=device_fp)
        return render_template('denied.html',
                                reason=sim_message,
                                layer='Layer 2 — SIM Binding')

    # ── STEP-UP DECISION ──────────────────────────────────────────────
    # Known device + SIM binding OK   -> standard SMS OTP.
    # Unknown device + SIM binding OK -> step-up: a code is first sent to the
    #   registered EMAIL (a channel independent of SMS), and only after it is
    #   verified does the SMS OTP step follow. An attacker who holds only the
    #   victim's SIM and password therefore cannot proceed from a new device.
    is_step_up = not fp_passed
    session['pending_user_id']     = user.id
    session['device_fp']           = device_fp
    session['device_consistency']  = consistency
    session['is_step_up_required'] = is_step_up
    session['email_verified']      = False

    if is_step_up:
        flash(f'Security Alert: {fp_message}', 'warning')
        log_attempt(user, 'STEP_UP', reason=fp_message, layer='LAYER_1', device_fp=device_fp)
        send_email_challenge(user)
        flash('We sent a verification code to your registered email address.', 'info')
        return redirect(url_for('verify_email'))

    log_attempt(user, 'PROCEED', layer='LAYER_1', device_fp=device_fp)
    send_otp(user)
    session['otp_sent_at'] = datetime.utcnow().isoformat()
    flash('OTP sent to your registered phone number.', 'info')
    return redirect(url_for('verify_otp'))


def _mask_email(address: str) -> str:
    local, _, domain = address.partition('@')
    return f"{local[:1]}***@{domain}" if domain else '***'


@app.route('/verify-email', methods=['GET', 'POST'])
def verify_email():
    """Layer 1 step-up: confirm the new device via a code sent by email."""
    if 'pending_user_id' not in session or not session.get('is_step_up_required'):
        return redirect(url_for('login'))

    user = db.session.get(User, session['pending_user_id'])
    if not user:
        session.clear()
        return redirect(url_for('login'))

    if session.get('email_verified'):
        return redirect(url_for('verify_otp'))

    if request.method == 'GET':
        return render_template('verify_email.html', username=user.username,
                                email=_mask_email(user.email))

    submitted = request.form.get('email_code', '').strip()
    passed, message = verify_email_code(user, submitted)

    if not passed:
        flash(message, 'danger')
        log_attempt(user, 'DENY', reason=message, layer='LAYER_1',
                    device_fp=session.get('device_fp'))
        if user.is_locked:
            session.clear()
            return render_template('denied.html',
                                    reason='Account locked — too many failed attempts',
                                    layer='Layer 1 — Step-up verification')
        return render_template('verify_email.html', username=user.username,
                                email=_mask_email(user.email))

    session['email_verified'] = True
    log_attempt(user, 'PROCEED', reason='Email step-up verified', layer='LAYER_1',
                device_fp=session.get('device_fp'))
    send_otp(user)
    session['otp_sent_at'] = datetime.utcnow().isoformat()
    flash('Email verified. A one-time code was sent to your phone by SMS.', 'info')
    return redirect(url_for('verify_otp'))


@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    if 'pending_user_id' not in session:
        return redirect(url_for('login'))

    user = db.session.get(User, session['pending_user_id'])
    if not user:
        session.clear()
        return redirect(url_for('login'))

    # An unrecognised device must complete the email step-up first.
    if session.get('is_step_up_required') and not session.get('email_verified'):
        return redirect(url_for('verify_email'))

    otp_request = (OTPRequest.query
                   .filter_by(user_id=user.id)
                   .order_by(OTPRequest.requested_at.desc())
                   .first())
    remaining = get_otp_remaining_seconds(otp_request.requested_at) if otp_request else 0

    if request.method == 'GET':
        return render_template('verify_otp.html',
                                username=user.username,
                                phone=user.phone_number[-4:],
                                remaining=remaining)

    submitted_otp = request.form.get('otp', '').strip()

    # ── LAYER 3: OTP VALIDATION ─────────────────────────────────────────
    otp_passed, otp_message = verify_otp_code(user, submitted_otp)

    if not otp_passed:
        flash(otp_message, 'danger')
        log_attempt(user, 'DENY', reason=otp_message, layer='LAYER_3',
                    device_fp=session.get('device_fp'))
        if user.is_locked:
            session.clear()
            return render_template('denied.html',
                                    reason='Account locked — too many failed attempts',
                                    layer='Layer 3 — OTP Validation')
        return render_template('verify_otp.html',
                                username=user.username,
                                phone=user.phone_number[-4:],
                                remaining=remaining)

    # ── LAYER 4: ML ANOMALY DETECTION ────────────────────────────────────
    # NOTE: most of these fields are still placeholder constants — see
    # ml_model.py docstring. device_consistency is the one real signal
    # flowing through end-to-end right now.
    session_context = {
        'device_consistency':        session.get('device_consistency', 0.8),
        'type_encoded':              3,
        'sim_age_days':              180,
        'new_account_flag':          0,
        'last_transaction_amount':   500.0,
        'balance_change_initiator':  500.0,
        'balance_change_recipient':  500.0,
        'balance_change_ratio':      0.1,
        'zero_balance_initiator':    0,
        'zero_balance_recipient':    0,
        'transaction_velocity':      2,
        'high_risk_type':            0,
        'balance_discrepancy':       0.0,
        'balance_discrepancy_ratio': 0.0,
    }

    ml_passed, risk_score, risk_level, ml_message = run_anomaly_detection(user, session_context)
    device_fp = session.get('device_fp')

    if not ml_passed:
        log_attempt(user, 'DENY', reason=ml_message, layer='LAYER_4',
                    risk_score=risk_score, risk_level=risk_level, device_fp=device_fp)
        return render_template('denied.html',
                                reason=ml_message,
                                layer='Layer 4 — ML Anomaly Detection',
                                risk_score=round(risk_score * 100, 2))

    log_attempt(user, 'GRANT', risk_score=risk_score, risk_level=risk_level, device_fp=device_fp)

    # Promote a new device to trusted ONLY now: email code, SMS OTP and the
    # ML risk check have all passed. (Previously an SMS OTP alone was enough.)
    if session.get('is_step_up_required') and session.get('email_verified'):
        add_trusted_device(user, device_fp, label='Verified via email + SMS step-up')

    session.pop('pending_user_id', None)
    session.pop('device_fp', None)
    session.pop('device_consistency', None)
    session.pop('otp_sent_at', None)
    session.pop('is_step_up_required', None)
    session.pop('email_verified', None)

    session['authenticated_user'] = user.id
    session['risk_score']         = round(risk_score * 100, 2)
    session['risk_level']         = risk_level
    session.permanent             = True

    return redirect(url_for('dashboard'))


@app.route('/dashboard')
def dashboard():
    if 'authenticated_user' not in session:
        return redirect(url_for('login'))

    user = db.session.get(User, session['authenticated_user'])
    if not user:
        session.clear()
        return redirect(url_for('login'))

    risk_score = session.get('risk_score', 0)
    risk_level = session.get('risk_level', 'LOW')

    recent_logs = (AuthenticationLog.query
                   .filter_by(user_id=user.id)
                   .order_by(AuthenticationLog.timestamp.desc())
                   .limit(5).all())

    recent_transactions = (TransactionLog.query
                            .filter_by(user_id=user.id)
                            .order_by(TransactionLog.timestamp.desc())
                            .limit(5).all())

    return render_template('dashboard.html',
                            user=user,
                            risk_score=risk_score,
                            risk_level=risk_level,
                            recent_logs=recent_logs,
                            recent_transactions=recent_transactions)


@app.route('/transfer', methods=['GET', 'POST'])
def transfer():
    """
    Simulated post-login transfer. NOT a real fintech feature — no real
    balance, no real recipient account, no money moves. This exists to
    produce real transaction-shaped numbers for Layer 4's risk model,
    which is what your training data is actually built around. This is
    the checkpoint that demonstrates the SIM-swap -> account-takeover ->
    fraudulent-transaction chain: if an attacker reaches this point, this
    is where rapid/large/unusual transfers should get caught.
    """
    if 'authenticated_user' not in session:
        return redirect(url_for('login'))

    user = db.session.get(User, session['authenticated_user'])
    if not user:
        session.clear()
        return redirect(url_for('login'))

    if request.method == 'GET':
        return render_template('transfer.html')

    try:
        amount = float(request.form.get('amount', '0').strip())
    except ValueError:
        flash('Enter a valid amount.', 'danger')
        return render_template('transfer.html')

    recipient_label = request.form.get('recipient_label', '').strip() or 'Unspecified recipient'

    if amount <= 0:
        flash('Amount must be greater than zero.', 'danger')
        return render_template('transfer.html')

    # REAL transaction_velocity: count this user's transfers in the last
    # 10 minutes, computed from actual logged rows — not a constant.
    ten_minutes_ago = datetime.utcnow() - timedelta(minutes=10)
    recent_count = (TransactionLog.query
                     .filter(TransactionLog.user_id == user.id,
                             TransactionLog.timestamp >= ten_minutes_ago)
                     .count())

    # Carry over this session's device-consistency signal from login,
    # since a takeover is still happening from the same (untrusted)
    # device even when the user reaches the transfer screen.
    device_consistency = session.get('device_consistency', 1.0)

    passed, risk_score, risk_level, message = run_transaction_risk_check(
        {'amount': amount, 'recipient_label': recipient_label},
        recent_transaction_count=recent_count + 1,  # +1 to include this attempt
        device_consistency=device_consistency,
    )

    decision = 'GRANT' if passed else 'DENY'

    txn = TransactionLog(
        user_id=user.id,
        amount=amount,
        recipient_label=recipient_label,
        risk_score=risk_score,
        risk_level=risk_level,
        decision=decision,
    )
    db.session.add(txn)
    db.session.commit()

    if not passed:
        return render_template('denied.html',
                                reason=message,
                                layer='Layer 4 — Transaction Risk Check',
                                risk_score=round(risk_score * 100, 2))

    flash(f'{message} (risk: {round(risk_score * 100, 1)}%)',
          'warning' if risk_level == 'MEDIUM' else 'success')
    return redirect(url_for('dashboard'))


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))


# ── DEMO / TESTING ENDPOINT ──────────────────────────────────────────────
# Stands in for a carrier webhook reporting a SIM swap, so the denial
# path can be demonstrated live without real telecom integration.
@app.route('/admin/simulate-sim-swap/<username>')
def admin_simulate_sim_swap(username):
    if not app.debug:
        return "Not available outside debug mode", 403

    user = User.query.filter_by(username=username).first()
    if not user:
        return f"No such user: {username}", 404

    simulate_sim_swap(user)
    return f"SIM swap simulated for {username}. Next login will be denied at Layer 2."


@app.route('/api/risk-score', methods=['POST'])
def api_risk_score():
    data       = request.get_json()
    features   = data.get('features', {})
    risk_score = calculate_risk_score(features) if features else 0.5
    risk_level = classify_risk(risk_score)
    return jsonify({
        'risk_score': round(risk_score * 100, 2),
        'risk_level': risk_level,
        'decision':   'DENY' if risk_level == 'HIGH' else 'GRANT'
    })


# ── ENTRY POINT ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    load_model()
    print("\n" + "=" * 50)
    print("  Secured SMS Authentication System")
    print("  Running on http://0.0.0.0:5000")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', debug=True, port=5000)