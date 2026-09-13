"""
HTTP Basic authentication for Pulse.

Pulse triggers real pipeline runs and approves real deliveries, so on a public
URL it needs a lock on the front door.

Deliberately fails closed: if ``PULSE_PASSWORD`` is not set, protected routes
return 503 rather than serving without authentication. An unset password is a
misconfiguration, and on a public deployment silently allowing anyone in is a
far worse outcome than refusing to serve.

Credentials are compared with :func:`secrets.compare_digest` so that a wrong
guess takes the same time regardless of how much of it was correct.

Note that Basic credentials are base64, not encrypted — they are only private
because Railway terminates TLS. Do not expose this over plain HTTP.
"""

from __future__ import annotations

import logging
import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

logger = logging.getLogger(__name__)

DEFAULT_USERNAME = "admin"

# Sent so browsers show their native sign-in dialog.
_CHALLENGE = {"WWW-Authenticate": 'Basic realm="Pulse"'}

_security = HTTPBasic(auto_error=True)


def _configured() -> tuple[str, str]:
    """Read the expected credentials from the environment."""
    return (
        os.getenv("PULSE_USERNAME", DEFAULT_USERNAME),
        os.getenv("PULSE_PASSWORD", ""),
    )


def auth_enabled() -> bool:
    """Whether a password has been configured."""
    return bool(_configured()[1])


def require_auth(
    credentials: HTTPBasicCredentials = Depends(_security),
) -> str:
    """Validate Basic credentials, or reject the request.

    Returns:
        The authenticated username.

    Raises:
        HTTPException: 503 when no password is configured, 401 when the
            supplied credentials do not match.
    """
    expected_user, expected_password = _configured()

    if not expected_password:
        logger.error(
            "PULSE_PASSWORD is not set - refusing to serve. Set it to enable access."
        )
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Pulse is not configured: PULSE_PASSWORD is unset. "
            "Set it in the environment to enable access.",
        )

    # compare_digest on bytes, so non-ASCII input cannot raise.
    user_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"), expected_user.encode("utf-8")
    )
    password_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"), expected_password.encode("utf-8")
    )

    # Both comparisons always run, so the response time does not reveal
    # whether it was the username or the password that was wrong.
    if not (user_ok and password_ok):
        logger.warning("Rejected sign-in for username %r", credentials.username)
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect username or password.",
            headers=_CHALLENGE,
        )

    return credentials.username


# Attach to a route with: @app.get(..., dependencies=PROTECTED)
PROTECTED = [Depends(require_auth)]
