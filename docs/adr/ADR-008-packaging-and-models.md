# ADR-008: Packaging (PyInstaller + installers) and model distribution

Status: Accepted · Date: 2026-07-03

## Context
End users must not need Python. Models are gigabytes and cannot ship inside
installers. Targets: Windows and Linux, macOS-ready architecture.

## Decision
- **PyInstaller** one-dir bundles of engine + desktop shell + built web assets.
- **Windows**: Inno Setup installer (start-menu entry, optional autostart).
- **Linux**: AppImage primary; tarball secondary.
- **Models are NOT bundled.** First-run setup wizard: hardware detection → recommended
  profile → download models (Hugging Face URLs, SHA-256 verified, resumable) into a
  user data dir (`platformdirs`). After that the app never touches the network;
  a fully-offline path (point the model manager at local files) is supported.
- CPU llama.cpp wheels by default + CUDA acceleration detected/enabled when present
  (keeps one installer per OS instead of a CUDA/CPU matrix).

## Rationale
- PyInstaller is the proven path for torch/onnx/llama.cpp stacks; keeps everything in
  one toolchain (vs. Nuitka/briefcase risk).
- Separating app (~hundreds of MB) from models (GBs) keeps installers reasonable and
  lets users switch profiles without reinstalling.

## Consequences
- The model registry (name → files, hashes, licenses, VRAM needs) is data, reviewed
  in-repo; the downloader is the only network code in the product.
- CI builds unsigned artifacts; code-signing is a later, optional step.
