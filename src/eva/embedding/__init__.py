"""Text-to-vector embeddings (ADR-020). A sibling subsystem to vad/asr/llm/tts
— `eva.memory` depends on this subsystem's port (ADR-010 amendment) to turn
text into vectors for semantic search; nothing about this capability is
memory-specific.
"""

from __future__ import annotations
