"""
otp.py
Time-based code generation/validation used as the SMS-delivered OTP layer.

NOTE ON TERMINOLOGY (read before chapter 4 writeup):
This module uses the TOTP algorithm (RFC 6238) purely as a way to generate
a time-windowed 6-digit code server-side, which is then delivered over the
SMS channel. There is no authenticator-app enrollment step, so the secret
never leaves the server — this is functionally an SMS-OTP layer, not a
possession-factor TOTP layer (where the user's device independently
computes the same code from a shared secret). Describe it in the
methodology as "time-based OTP delivered via SMS channel", not as TOTP
in the strict two-factor sense, to avoid overclaiming under questioning.
"""

import pyotp
import time
from datetime import datetime, timezone


def generate_secret() -> str:
    """Generate a new random base32 secret key for a user (server-side only)."""
    return pyotp.random_base32()


def generate_otp(secret: str) -> str:
    """
    Generate the current time-windowed 6-digit code from the user's secret.
    Code is valid for one 30-second step by default.
    """
    totp = pyotp.TOTP(secret)
    return totp.now()


def validate_otp(secret: str, user_otp: str) -> bool:
    """
    Validate a submitted code against the current and adjacent time steps.
    valid_window=1 allows +/- 30 seconds of drift/delay, matching the
    90-second display countdown used in the UI.
    """
    totp = pyotp.TOTP(secret)
    return totp.verify(user_otp, valid_window=1)


def get_otp_remaining_seconds(requested_at, window_seconds: int = 90) -> int:
    """
    Seconds remaining before the OTP request should be considered expired
    for UI countdown purposes. Accepts either a datetime or an ISO string.

    IMPORTANT: requested_at is stored via datetime.utcnow(), which produces
    a "naive" datetime (no timezone attached) that nonetheless represents
    UTC. Calling .timestamp() directly on a naive datetime makes Python
    assume it's in the SERVER'S LOCAL TIMEZONE, not UTC. On any machine
    not set to UTC (e.g. Nigeria, UTC+1), this silently introduces a
    1-hour error into every countdown. We fix this by explicitly attaching
    tzinfo=timezone.utc before calling .timestamp(), so the conversion to
    epoch seconds is correct regardless of the server's local timezone.
    """
    if requested_at is None:
        return 0

    if isinstance(requested_at, str):
        try:
            dt = datetime.fromisoformat(requested_at)
        except ValueError:
            dt = datetime.strptime(requested_at, '%Y-%m-%d %H:%M:%S')
    else:
        dt = requested_at

    # Treat naive datetimes as UTC explicitly (this is what they actually
    # are, since they came from datetime.utcnow()). If a tz-aware datetime
    # is ever passed in instead, leave it untouched.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    start_time = dt.timestamp()
    current_time = datetime.now(timezone.utc).timestamp()
    remaining = window_seconds - (current_time - start_time)
    return max(0, int(remaining))


def simulate_sms(phone_number: str, otp_code: str):
    """
    Simulates sending an SMS by printing to the console. For the real
    submission, swap this for an Africa's Talking or Termii API call —
    both have better Nigerian carrier deliverability than Twilio.
    """
    print(f"\n{'='*50}")
    print(f"  SIMULATED SMS SENT")
    print(f"  To:      {phone_number}")
    print(f"  Message: Your SecureAuth OTP is: {otp_code}")
    print(f"  Valid for 90 seconds.")
    print(f"{'='*50}\n")