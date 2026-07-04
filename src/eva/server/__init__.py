"""Platform API (ADR-017): the FastAPI backend every client — CLI, desktop,
web, plugins, and future mobile — consumes through the same versioned surface.
"""

from eva.server.app import create_app

__all__ = ["create_app"]
