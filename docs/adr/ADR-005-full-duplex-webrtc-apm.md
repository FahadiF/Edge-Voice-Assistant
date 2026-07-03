# ADR-005: Full-duplex audio with WebRTC APM echo cancellation

Status: Accepted · Date: 2026-07-03

## Context
Barge-in is the top product priority. The thesis either muted the mic during playback
(speakers) or used a headset-only heuristic that discarded the interrupting speech and
could not cancel in-flight generation. True barge-in on speakers requires removing the
assistant's own voice from the mic signal (AEC).

## Decision
- One **duplex PortAudio stream** (single device clock) processing **10 ms frames**.
- **WebRTC Audio Processing Module** (via livekit's `rtc.apm` or
  `webrtc-audio-processing` bindings) provides AEC + noise suppression + AGC.
  Playback frames are fed to the APM as the far-end reference in the same callback,
  giving the tight mic/reference alignment WebRTC AEC needs.
- VAD consumes the echo-cancelled stream; barge-in triggers on ~200 ms confirmed
  speech during playback and bumps the turn epoch (see ARCHITECTURE §3).
- Config fallback ladder: full-duplex AEC → half-duplex mute → push-to-talk.

## Rationale
- WebRTC APM is the most battle-tested open AEC in existence (every browser call).
- The known Python-side failure mode — misaligned reference signal — is avoided by
  doing playback and capture in one duplex stream on one clock, not two independent
  streams as in the thesis.
- Keeping the raw pre-trigger frames in a ring buffer means the interrupting utterance
  ("No, stop") is captured, not discarded.

## Consequences
- Audio subsystem is the most platform-sensitive component → built and validated
  first (M1) with a dedicated echo-loop test rig before any model work depends on it.
- If a specific device's AEC quality is poor, per-device fallback to half-duplex is
  automatic (measured by residual-echo VAD false-trigger rate in diagnostics).
