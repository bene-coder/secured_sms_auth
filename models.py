"""
models.py
Database models for the secured SMS authentication system.
Uses SQLite via flask-sqlalchemy.
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    """Registered user accounts."""
    __tablename__ = 'users'

    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    phone_number  = db.Column(db.String(20),  nullable=False)
    totp_secret   = db.Column(db.String(32),  nullable=False)
    # Highest TOTP time step already accepted for this user. Any code whose
    # step is <= this value is a replay and is rejected (single-use OTP).
    last_otp_step = db.Column(db.BigInteger, nullable=True)

    # ── Layer 2: SIM Binding ──────────────────────────────────────────────
    # registered_sim_id is set ONCE at registration and is NEVER accepted
    # from client input again on login. sim_swap_flagged is a testing/demo
    # toggle that stands in for a carrier-reported SIM swap event, since
    # real HLR/SS7 carrier integration is outside this project's scope.
    registered_sim_id   = db.Column(db.String(20), nullable=True)
    sim_registered_at   = db.Column(db.DateTime, nullable=True)
    sim_swap_flagged    = db.Column(db.Boolean, default=False)
    sim_swap_flagged_at = db.Column(db.DateTime, nullable=True)

    # Account status
    is_locked       = db.Column(db.Boolean, default=False)
    failed_attempts = db.Column(db.Integer, default=0)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.username}>'


class DeviceFingerprint(db.Model):
    """
    Tracks every distinct device fingerprint a user has authenticated
    from. Using a one-to-many table (rather than a single column on User)
    means a legitimate user with two devices (phone + laptop) doesn't get
    treated as 'attacker' every time they switch — only a genuinely never
    -seen fingerprint triggers step-up.
    """
    __tablename__ = 'device_fingerprints'

    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    fingerprint = db.Column(db.String(64), nullable=False)  # sha256 hex
    label       = db.Column(db.String(100), nullable=True)
    trusted     = db.Column(db.Boolean, default=True)
    first_seen  = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen   = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<DeviceFingerprint user_id={self.user_id} fp={self.fingerprint[:8]}>'


class AuthenticationLog(db.Model):
    """Logs every authentication attempt for audit purposes."""
    __tablename__ = 'auth_logs'

    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    username       = db.Column(db.String(80),  nullable=True)
    timestamp      = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address     = db.Column(db.String(45),  nullable=True)
    device_fp      = db.Column(db.String(64),  nullable=True)
    risk_score     = db.Column(db.Float,       nullable=True)
    risk_level     = db.Column(db.String(10),  nullable=True)
    decision       = db.Column(db.String(10),  nullable=False)
    failure_reason = db.Column(db.String(100), nullable=True)
    layer_failed   = db.Column(db.String(20),  nullable=True)

    def __repr__(self):
        return f'<AuthLog {self.username} {self.decision}>'


class OTPRequest(db.Model):
    """Tracks OTP requests to enforce rate limiting and expiry."""
    __tablename__ = 'otp_requests'

    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    requested_at = db.Column(db.DateTime, default=datetime.utcnow)
    used         = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f'<OTPRequest user_id={self.user_id}>'


class EmailChallenge(db.Model):
    """
    Step-up challenge for an unrecognised device. The code is delivered over
    email (a channel independent of SMS), is stored only as a hash, is valid
    for a short time and can be used once.
    """
    __tablename__ = 'email_challenges'

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    code_hash  = db.Column(db.String(64), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    used       = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f'<EmailChallenge user_id={self.user_id}>'


class TransactionLog(db.Model):
    """
    Logs simulated post-login transfer attempts.

    DELIBERATELY MINIMAL — this is NOT a fintech ledger. There is no real
    balance, no real recipient account, no money movement. It exists only
    to produce real numbers (amount, timing, frequency) for the ML risk
    model's transaction-shaped features, instead of the hardcoded
    constants used previously. recipient_label is free text the user
    types — it is never validated against a real account, by design.
    """
    __tablename__ = 'transaction_logs'

    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    amount           = db.Column(db.Float, nullable=False)
    recipient_label  = db.Column(db.String(100), nullable=True)
    timestamp        = db.Column(db.DateTime, default=datetime.utcnow)
    risk_score       = db.Column(db.Float, nullable=True)
    risk_level       = db.Column(db.String(10), nullable=True)
    decision         = db.Column(db.String(10), nullable=True)  # GRANT / DENY

    def __repr__(self):
        return f'<TransactionLog user_id={self.user_id} amount={self.amount}>'