# ADR-013: LLM runtime installed via `eva setup`, not as a base dependency

Status: Accepted · Date: 2026-07-04

## Context
M2 shipped with `llama-cpp-python` installed manually in the development
environment but not declared in `pyproject.toml`, so a clean checkout failed at
runtime with `No module named 'llama_cpp'`. Fixing the declaration surfaced the
real constraint: **`llama-cpp-python` publishes no wheels on PyPI — only an
sdist.** A plain `pip install llama-cpp-python` therefore compiles from source,
requiring CMake and a C++ toolchain (MSVC on Windows), which is unacceptable for
a product installed by non-developers. Prebuilt CPU and CUDA wheels exist, but
only on the project's own wheel index (`abetlen.github.io/llama-cpp-python`),
and a wheel index cannot be expressed in PyPI dependency metadata.

By contrast, the ASR and TTS runtimes (`faster-whisper`, `kokoro-onnx`) publish
universal wheels on PyPI and install cleanly with no compiler.

## Decision
1. **Base dependencies** include everything that has PyPI wheels:
   `faster-whisper` and `kokoro-onnx` join the existing audio/VAD stack. A plain
   `pip install -e "."` yields a runnable ASR + TTS + audio application with no
   compiler on Windows or Linux.
2. **`llama-cpp-python` is not a base dependency.** It is offered as `[cpu]` and
   `[cuda]` extras (the `[cuda]` extra also pulls the NVIDIA cudart/cuBLAS
   redistributable wheels the CUDA build links against).
3. **`eva setup`** is the blessed installer: it detects the hardware, picks the
   `cpu`/`cuda` variant, and runs `pip install` against the correct wheel index.
   The install command is built as data (`eva.runtime.build_llama_install_command`)
   so it is unit-tested and available via `--dry-run`.
4. **`eva doctor`** probes every runtime and model and prints a per-item remedy;
   `eva run`/`eva bench` run the same preflight and refuse to start with clear
   guidance instead of a traceback.

## Alternatives rejected
- **Plain `llama-cpp-python` in base deps** — compiles from sdist; breaks clean
  installs on machines without a C++ toolchain. The opposite of the goal.
- **Hard-coded per-platform wheel URLs in an extra** (PEP 508 direct references)
  — works with plain pip but pins version + Python tag + platform in the
  metadata; rots on every Python/model bump and misses untested platforms. Fails
  the "good design in five years" test. The wheel *index* auto-resolves the right
  wheel per platform/Python version and is far more maintainable.
- **Bundling llama.cpp binaries in the repo** — large, platform-specific, and a
  security/update burden; deferred to the M8 packaged installers, which will ship
  a bundled runtime for end users who never touch pip.

## Consequences
- Installation is a documented three-stage flow (base install → `eva setup` →
  `eva models download`), captured in `INSTALLATION.md`.
- Every milestone from here must pass a clean-environment smoke test: create a
  fresh venv, `pip install -e ".[dev]"`, and confirm the commands either work or
  fail with actionable guidance — no `ModuleNotFoundError` may escape.
- M8's packaged installers will bundle the runtime so end users skip `eva setup`.
