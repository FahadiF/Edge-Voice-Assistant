"""The single controlled egress point (H2).

EVA's central product claim is that it runs fully offline. Before this module
that claim was enforced by convention: four modules used `urllib.request` and
only one of them talked to the internet, which meant a naive "no outbound
sockets" test would have broken the desktop shell, the service supervisor, and
`eva status` — all three of which speak HTTP to **localhost**.

This module makes the distinction structural rather than a matter of reading
the code:

- **Egress** — a connection to a host outside this machine. Only
  `eva.models.manager` performs it (downloading model weights), and only
  through `open_url()` here. Enforced by an import-direction test
  (`tests/test_offline_invariant.py`) asserting nothing else in `eva` imports
  `urllib.request`, plus the three loopback clients named below.
- **Loopback IPC** — a connection to `127.0.0.1`/`::1`/`localhost`. This is how
  EVA's own processes talk to each other; it never leaves the machine and is
  not egress. `eva.desktop.client`, `eva.service`, and `eva.cli`'s status probe
  do exactly this and are deliberately left calling `urllib.request` directly:
  routing local IPC through an egress boundary would misfile it as the very
  thing this module exists to contain.

Deliberately not an HTTP abstraction layer. `open_url()` is a thin, honest
seam over `urllib.request` — one place to audit, one place a future policy
(proxy, allowlist, offline-mode refusal) would attach. It does not wrap
responses, normalise errors, or add retries: `eva.models.manager` already owns
resume/verification semantics for downloads, and duplicating that here would
create two half-policies instead of one.

`urllib.error.URLError` subclasses `OSError`, so callers catch `OSError` and
cover both transport and protocol failures without importing `urllib`.
"""

from __future__ import annotations

import ipaddress
import urllib.request
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

#: Hostnames that mean "this machine" without being IP literals.
_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain"})

#: The wildcard bind address. Not connectable itself (see
#: `eva.service.display_host`), but it denotes a local listener, never a remote
#: host, so an address derived from it is local by intent.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "[::]"})


def is_loopback_host(host: str | None) -> bool:
    """True when `host` addresses this machine.

    The one definition of the loopback/egress boundary, kept here so the
    socket-blocking test fixture and any future policy agree by construction
    rather than by two similar-looking checks drifting apart.

    Accepts IP literals (`127.0.0.1`, `::1`, and the whole `127.0.0.0/8`
    range), bracketed IPv6 (`[::1]`), and the loopback hostnames. Anything
    else — including an empty or unparseable host — is treated as egress,
    because the safe default for "I cannot tell" is to deny.
    """
    if not host:
        return False
    cleaned = host.strip().strip("[]").lower()
    if cleaned in _LOOPBACK_NAMES:
        return True
    if cleaned in _WILDCARD_HOSTS:
        return True
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        return False


def is_loopback_url(url: str) -> bool:
    """True when `url` points at this machine. See `is_loopback_host`."""
    try:
        return is_loopback_host(urlsplit(url).hostname)
    except ValueError:
        return False


def open_url(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> Any:
    """Open `url` and return the response, which is also a context manager.

    The only sanctioned outbound HTTP call in the codebase. Behaviourally
    identical to calling `urllib.request.urlopen()` directly, which is the
    point: this is a boundary, not a rewrite.

    `timeout=None` omits the argument entirely rather than forwarding `None`.
    `urlopen`'s default is a private sentinel (`socket._GLOBAL_DEFAULT_TIMEOUT`),
    not `None`, so forwarding `None` would silently convert "use the default"
    into "block forever" — and model downloads are exactly the long-running
    call where that difference matters.
    """
    request = urllib.request.Request(url, headers=dict(headers or {}))
    if timeout is None:
        return urllib.request.urlopen(request)
    return urllib.request.urlopen(request, timeout=timeout)
