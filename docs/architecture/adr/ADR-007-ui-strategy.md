# ADR-007: One engine, web UI + desktop shell sharing the same frontend

Status: Accepted · Date: 2026-07-03

## Context
Requirements: a modern desktop app AND a localhost web interface, both offline,
without maintaining two UI codebases.

## Decision
- The engine is a **FastAPI server** (localhost only by default) exposing WebSocket
  events + REST control (see ARCHITECTURE §8). Audio devices are owned by the engine
  process, never the browser.
- **Web UI: React + Vite + TypeScript**, built to static assets served by FastAPI.
- **Desktop app: pywebview shell** (native window + tray + launcher) hosting the same
  built UI and supervising the engine process. Single UI codebase for both surfaces.
- CLI remains for development/headless use.

## Rationale
- ChatGPT-Voice-class UX (streaming transcripts, waveforms, live state) is fastest to
  build and iterate in web tech; PySide6 would duplicate that effort for one surface.
- pywebview keeps the whole product in the Python packaging story (one PyInstaller
  bundle). Tauri (Rust toolchain + sidecar process management) adds a second build
  ecosystem — rejected for now, revisitable if the shell needs to grow.
- Server-owned audio means the web page is a pure renderer — no getUserMedia
  permissions, no browser audio latency, works in any local browser.

## Consequences
- The WebSocket protocol is a stable, versioned contract — effectively a third
  "interface" that future integrations (home automation, etc.) can also use.
- Desktop-only affordances (global hotkey for push-to-talk, tray) live in the shell.
