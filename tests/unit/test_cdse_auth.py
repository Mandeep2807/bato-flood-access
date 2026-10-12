"""CDSE authentication: credentials from env, token request, failures, no secret leakage.

All HTTP is faked. A LIVE authentication test is PENDING: it needs real CDSE
credentials and network access to identity.dataspace.copernicus.eu.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from bato.acquisition.cdse_auth import AccessToken, CDSECredentials, request_access_token
from bato.acquisition.http import HttpResponse, TransportError
from bato.exceptions import AuthenticationError

SECRET = "s3cret-Pa55word!"
NOW = datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc)


class FakeTransport:
    def __init__(self, response: HttpResponse | Exception) -> None:
        self.response = response
        self.forms: list[dict[str, str]] = []

    def get(self, url, timeout_s):  # pragma: no cover
        raise AssertionError("token requests must POST")

    def post_form(self, url: str, form: dict[str, str], timeout_s: float) -> HttpResponse:
        self.forms.append(form)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


CREDS = CDSECredentials(username="analyst@example.org", password=SECRET)


def test_credentials_from_env():
    creds = CDSECredentials.from_env({"CDSE_USERNAME": " analyst@example.org ", "CDSE_PASSWORD": SECRET})
    assert creds.username == "analyst@example.org" and creds.totp is None
    assert SECRET not in repr(creds)


@pytest.mark.parametrize("env", [{}, {"CDSE_USERNAME": "a"}, {"CDSE_PASSWORD": SECRET}, {"CDSE_USERNAME": " ",
                                                                                        "CDSE_PASSWORD": SECRET}])
def test_missing_credentials(env):
    with pytest.raises(AuthenticationError, match="credentials missing") as info:
        CDSECredentials.from_env(env)
    assert SECRET not in str(info.value)


def test_successful_token_request_uses_official_form(cfg):
    transport = FakeTransport(HttpResponse(200, json.dumps({"access_token": "tok-123", "expires_in": 600}).encode()))
    token = request_access_token(CREDS, cfg.settings.cdse, transport, now=NOW)
    assert token.expires_at == NOW + timedelta(seconds=600)
    assert transport.forms[0] == {"client_id": "cdse-public", "grant_type": "password",
                                  "username": "analyst@example.org", "password": SECRET}
    assert "tok-123" not in repr(token)
    assert not token.is_expired(NOW) and token.is_expired(NOW + timedelta(seconds=590))


def test_totp_is_sent_when_configured(cfg):
    transport = FakeTransport(HttpResponse(200, b'{"access_token": "t", "expires_in": 60}'))
    request_access_token(CDSECredentials("u", SECRET, totp="123456"), cfg.settings.cdse, transport, now=NOW)
    assert transport.forms[0]["totp"] == "123456"


@pytest.mark.parametrize(("response", "message"), [
    (HttpResponse(401, b'{"error":"invalid_grant"}'), "rejected the credentials"),
    (HttpResponse(400, b'{"error":"invalid_request"}'), "rejected the credentials"),
    (HttpResponse(503, b"maintenance"), "HTTP 503"),
    (HttpResponse(200, b"not json"), "malformed"),
    (HttpResponse(200, b'{"access_token": "", "expires_in": 60}'), "empty access_token"),
    (HttpResponse(200, b'{"expires_in": 60}'), "malformed"),
    (TransportError("request to identity.dataspace.copernicus.eu failed: timed out"), "unreachable"),
])
def test_authentication_failures_never_leak_secrets(cfg, response, message):
    with pytest.raises(AuthenticationError, match=message) as info:
        request_access_token(CREDS, cfg.settings.cdse, FakeTransport(response), now=NOW)
    assert SECRET not in str(info.value)


def test_access_token_repr_hides_value():
    assert "abc" not in repr(AccessToken(value="abc", expires_at=NOW))
