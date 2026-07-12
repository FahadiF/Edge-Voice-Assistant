"""Browser-origin policy, shared by CORS (HTTP) and the WebSocket endpoint.

The API is localhost-only and unauthenticated by design (ADR-017 Part 9);
the browser origin check is what keeps that safe: an arbitrary website the
user happens to visit must not be able to read the live event stream (it
carries transcripts and reply text) or drive the assistant. CORS covers
HTTP, but Starlette's CORSMiddleware does not apply to WebSocket handshakes
— the `/ws` endpoint enforces the same policy itself (M5.6).

Requests without an Origin header are non-browser clients (the CLI, the
desktop shell, curl, tests) and are allowed: Origin is a browser security
mechanism, and only browsers attach it.
"""

from __future__ import annotations

import re

LOCALHOST_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

_LOCALHOST_ORIGIN = re.compile(LOCALHOST_ORIGIN_REGEX)


def origin_allowed(origin: str | None) -> bool:
    """True when `origin` is absent (non-browser client) or localhost."""
    return origin is None or _LOCALHOST_ORIGIN.match(origin) is not None
