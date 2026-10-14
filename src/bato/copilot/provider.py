"""LLM provider for the SITREP copilot, configured ONLY through environment variables (like OHSOME_API_KEY).

    BATO_LLM_PROVIDER   openai | anthropic   ("openai" = any OpenAI-compatible /chat/completions endpoint:
                                               OpenAI, Groq, OpenRouter, Together, a local Ollama / LM Studio, ...)
    BATO_LLM_MODEL      model name, e.g. gpt-4o-mini, claude-sonnet-4-5, llama3.1
    BATO_LLM_API_KEY    API key (not needed for an OpenAI-compatible server on localhost)
    BATO_LLM_BASE_URL   optional; default https://api.openai.com/v1 or https://api.anthropic.com
    BATO_LLM_TIMEOUT_S  optional request timeout in seconds (default 60)

The key is read server-side, sent only to the configured endpoint, and never returned, logged or written. If the
configuration is incomplete the copilot is UNAVAILABLE (``status()`` says why); there is no silent fallback.
Standard library only (urllib); system proxy settings are honoured.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from bato.exceptions import BatoError

ENV_PROVIDER, ENV_MODEL, ENV_KEY, ENV_BASE, ENV_TIMEOUT = (
    "BATO_LLM_PROVIDER", "BATO_LLM_MODEL", "BATO_LLM_API_KEY", "BATO_LLM_BASE_URL", "BATO_LLM_TIMEOUT_S")
DEFAULT_BASE = {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com"}
ANTHROPIC_VERSION = "2023-06-01"
MAX_TOKENS = 2500


class CopilotUnavailable(BatoError):
    """No usable LLM provider is configured (the reason says which variable is missing)."""


class ProviderError(BatoError):
    """The configured provider failed or returned an unusable response (message is safe to show)."""


class ChatModel(Protocol):
    label: str

    def complete(self, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    base_url: str
    api_key: str | None
    timeout_s: float

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def public(self) -> dict[str, Any]:
        """What the browser may see (never the key)."""
        return {"provider": self.provider, "model": self.model, "endpoint_host": urlsplit(self.base_url).hostname}


def _is_local(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1")


def config_from_env(env: Mapping[str, str] | None = None) -> ProviderConfig:
    """Raises CopilotUnavailable with the reason when the configuration is incomplete."""
    e = os.environ if env is None else env
    provider = (e.get(ENV_PROVIDER) or "").strip().lower()
    if not provider:
        raise CopilotUnavailable(f"no LLM provider configured: set {ENV_PROVIDER} (openai or anthropic), "
                                 f"{ENV_MODEL} and {ENV_KEY}")
    if provider not in DEFAULT_BASE:
        raise CopilotUnavailable(f"{ENV_PROVIDER}={provider!r} is not supported (use openai or anthropic)")
    model = (e.get(ENV_MODEL) or "").strip()
    if not model:
        raise CopilotUnavailable(f"{ENV_MODEL} is not set")
    base = (e.get(ENV_BASE) or DEFAULT_BASE[provider]).strip().rstrip("/")
    if urlsplit(base).scheme not in ("http", "https"):
        raise CopilotUnavailable(f"{ENV_BASE} must be an http(s) URL")
    if urlsplit(base).scheme == "http" and not _is_local(base):
        raise CopilotUnavailable(f"{ENV_BASE} must use https unless it points to localhost")
    key = (e.get(ENV_KEY) or "").strip() or None
    if key is None and not (provider == "openai" and _is_local(base)):
        raise CopilotUnavailable(f"{ENV_KEY} is not set")
    try:
        timeout = float(e.get(ENV_TIMEOUT) or 60)
    except ValueError as exc:
        raise CopilotUnavailable(f"{ENV_TIMEOUT} must be a number") from exc
    return ProviderConfig(provider, model, base, key, timeout)


class HttpChatModel:
    """OpenAI-compatible chat completions or the Anthropic Messages API, temperature 0."""

    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        self.label = cfg.label

    def _post(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as r:  # noqa: S310 - configured endpoint
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"the LLM provider returned HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"the LLM provider could not be reached ({type(exc).__name__})") from None
        except ValueError:
            raise ProviderError("the LLM provider returned a non-JSON response") from None

    def complete(self, system: str, user: str) -> str:
        c = self.cfg
        if c.provider == "anthropic":
            data = self._post(f"{c.base_url}/v1/messages",
                              {"model": c.model, "max_tokens": MAX_TOKENS, "temperature": 0, "system": system,
                               "messages": [{"role": "user", "content": user}]},
                              {"x-api-key": c.api_key or "", "anthropic-version": ANTHROPIC_VERSION})
            try:
                return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
            except (KeyError, TypeError):
                raise ProviderError("unexpected response shape from the Anthropic API") from None
        headers = {"Authorization": f"Bearer {c.api_key}"} if c.api_key else {}
        data = self._post(f"{c.base_url}/chat/completions",
                          {"model": c.model, "temperature": 0, "max_tokens": MAX_TOKENS,
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                          headers)
        try:
            return str(data["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError):
            raise ProviderError("unexpected response shape from the OpenAI-compatible API") from None


def model_from_env(env: Mapping[str, str] | None = None) -> HttpChatModel:
    return HttpChatModel(config_from_env(env))
