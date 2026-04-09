"""OpenAI client with TLS verification that works on macOS/Homebrew Python.

A single CA bundle is often incomplete (certifi-only or truststore-only can both
fail with ``unable to get local issuer certificate``). We merge multiple trusted
CA files via :meth:`ssl.SSLContext.load_verify_locations` when no explicit
``SSL_CERT_FILE`` is set.
"""

from __future__ import annotations

import os
import ssl
from typing import Union

import certifi
import httpx
from openai import OpenAI

VerifyTypes = Union[str, bool, ssl.SSLContext]


def _extra_ca_bundle_paths() -> list[str]:
    """Common PEM locations beyond certifi (Homebrew OpenSSL, macOS/Linux)."""
    candidates = (
        "/opt/homebrew/etc/ca-certificates/cert.pem",
        "/opt/homebrew/etc/openssl@3/cert.pem",
        "/usr/local/etc/openssl@3/cert.pem",
        "/usr/local/etc/ca-certificates/cert.pem",
        "/etc/ssl/cert.pem",
        "/etc/pki/tls/certs/ca-bundle.crt",
    )
    return [p for p in candidates if os.path.isfile(p)]


def _merged_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    bundles = [certifi.where(), *_extra_ca_bundle_paths()]
    seen: set[str] = set()
    for path in bundles:
        if not path or path in seen:
            continue
        seen.add(path)
        try:
            ctx.load_verify_locations(cafile=path)
        except OSError:
            continue
    return ctx


def _httpx_verify() -> VerifyTypes:
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        path = os.environ.get(key)
        if path and os.path.isfile(path):
            return path

    return _merged_ssl_context()


def build_openai_client(api_key: str) -> OpenAI:
    http = httpx.Client(verify=_httpx_verify(), timeout=120.0)
    return OpenAI(api_key=api_key, http_client=http)
