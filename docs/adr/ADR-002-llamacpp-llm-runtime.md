# ADR-002: llama.cpp (GGUF) as the LLM runtime, Qwen3-4B-Instruct default

Status: Accepted · Date: 2026-07-03

## Context
The thesis used `transformers` + Qwen2.5-1.5B fp16 (~3 GB VRAM, blocking
`generate()`, no cancellation). Target hardware is 6 GB VRAM / 16 GB RAM, and the
pipeline needs token streaming and instant abort for barge-in.

## Decision
Use **llama.cpp** (via `llama-cpp-python`) with GGUF models. Default model:
**Qwen3-4B-Instruct Q4_K_M** (~2.8 GB VRAM incl. headroom for KV cache). Profiles
map hardware → model (CPU-only: Qwen3-1.7B/Phi-4-mini class; 12 GB+: 7–9B class).

## Rationale
- **Quality per GB**: a 4B model at Q4 beats a 1.5B at fp16 on the same VRAM budget.
- **Native token streaming** and per-token abort callbacks → clean barge-in
  cancellation, which `transformers.generate()` does not offer cleanly.
- Removes the heavyweight torch/transformers dependency from the LLM path
  (smaller install, faster startup, simpler packaging).
- GGUF is the de-facto local-model format → hot-swap = load a different file;
  huge model catalogue for the model manager.
- Cross-platform (CUDA, CPU, Vulkan, Metal for future macOS).

## Alternatives rejected
- **transformers**: no clean abort, high VRAM, heavy packaging.
- **Ollama**: excellent UX but an external service dependency — wrong fit for a
  self-contained installable product (still possible later as an adapter).
- **vLLM/TensorRT-LLM**: server-class, Linux-centric, overkill for single user.

## Consequences
- LLM port is defined around async token streams + abort; any backend implementing
  that contract (Ollama adapter, transformers adapter) can be added later.
- Exact default model is re-validated in the M2 benchmark before being locked in.
