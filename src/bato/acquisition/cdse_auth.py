"""CDSE authentication (access tokens for downloads).

Catalogue search (Step 2A) does NOT need authentication: the CDSE OData
catalogue is public. This module exists so the download milestone can reuse a
tested, secret-safe token request.

Official mechanism (CDSE documentation, "Access token"): POST a form to the
Keycloak token endpoint with ``client_id=cdse-public``, ``grant_type=password``,
``username``, ``password`` and, if 2FA is enabled, ``totp``.

Secrets are read only from environment variables and are never logged,
included in exception messages, or shown in ``repr``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from bato.acquisition.http import Transport, TransportError, UrllibTransport
from bato.config import CDSESettings
from bato.exceptions import AuthenticationError

ENV_USERNAME = "CDSE_USERNAME"
ENV_PASSWORD = "CDSE_PASSWORD"  # noqa: S105 (variable name, not a secret)
ENV_TOTP = "CDSE_TOTP"


@dataclass(frozen=True)
class CDSECredentials:
    username: str
    password: str = field(repr=False)
    totp: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> CDSECredentials:
        env = os.environ if environ is None else environ
        username = (env.get(ENV_USERNAME) or "").strip()
        password = env.get(ENV_PASSWORD) or ""
        missing = [name for name, value in ((ENV_USERNAME, username), (ENV_PASSWORD, password)) if not value]
        if missing:
            raise AuthenticationError(
                f"CDSE credentials missing: set {', '.join(missing)} (see .env.example)")
        return cls(username=username, password=password, totp=(env.get(ENV_TOTP) or None))


@dataclass(frozen=True)
class AccessToken:
    value: str = field(repr=False)
    expires_at: datetime

    def is_expired(self, now: datetime | None = None, margin_s: float = 60.0) -> bool:
        now = now or datetime.now(timezone.utc)
        return now >= self.expires_at - timedelta(seconds=margin_s)


def request_access_token(credentials: CDSECredentials, settings: CDSESettings,
                         transport: Transport | None = None,
                         now: datetime | None = None) -> AccessToken:
    """Request an access token from the CDSE identity service.

    Raises:
        AuthenticationError: credentials rejected, service unreachable, or the
            response is not a usable token. Messages never contain secrets.
    """
    transport = transport or UrllibTransport()
    form = {"client_id": settings.client_id, "grant_type": "password",
            "username": credentials.username, "password": credentials.password}
    if credentials.totp:
        form["totp"] = credentials.totp
    try:
        response = transport.post_form(settings.token_url, form, settings.timeout_s)
    except TransportError as exc:
        raise AuthenticationError(f"CDSE identity service unreachable: {exc}") from exc

    if response.status in (400, 401, 403):
        raise AuthenticationError(f"CDSE rejected the credentials (HTTP {response.status})")
    if response.status != 200:
        raise AuthenticationError(f"CDSE identity service returned HTTP {response.status}")
    try:
        payload = json.loads(response.body)
        token = payload["access_token"]
        expires_in = float(payload["expires_in"])
    except (ValueError, KeyError, TypeError) as exc:
        raise AuthenticationError("CDSE token response is malformed (no access_token/expires_in)") from exc
    if not isinstance(token, str) or not token:
        raise AuthenticationError("CDSE token response contains an empty access_token")
    issued = now or datetime.now(timezone.utc)
    return AccessToken(value=token, expires_at=issued + timedelta(seconds=expires_in))


class TokenProvider:
    """Caches a CDSE access token and refreshes it when expired or rejected.

    Credentials are read from the environment only on the first ``get()``, so
    dry runs and catalogue searches never need them. The token value is never
    logged or returned in error messages.
    """

    def __init__(self, settings: CDSESettings, transport: Transport | None = None,
                 credentials: CDSECredentials | None = None,
                 environ: dict[str, str] | None = None) -> None:
        self._settings = settings
        self._transport = transport
        self._credentials = credentials
        self._environ = environ
        self._token: AccessToken | None = None
        self.refreshes = 0

    def get(self, now: datetime | None = None) -> str:
        if self._token is None or self._token.is_expired(now):
            if self._credentials is None:
                self._credentials = CDSECredentials.from_env(self._environ)
            self._token = request_access_token(self._credentials, self._settings, self._transport, now=now)
            self.refreshes += 1
        return self._token.value

    def invalidate(self) -> None:
        """Forget the current token (e.g. after HTTP 401) so the next get() refreshes it."""
        self._token = None
