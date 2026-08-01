"""Dump the platform API's OpenAPI schema to stdout (Batch 3, ADR-023).

Feeds `openapi-typescript` (see `web/package.json`'s `generate:types` script)
so the TypeScript mirror of REST-facing schemas is generated instead of
hand-transcribed. Constructing the app is enough to get its schema — no
server starts, no audio device opens, no model loads.

Usage: python scripts/dump_openapi.py > openapi.json
"""

from __future__ import annotations

import json

from eva.server.app import create_app


def main() -> None:
    app = create_app()
    print(json.dumps(app.openapi()))


if __name__ == "__main__":
    main()
