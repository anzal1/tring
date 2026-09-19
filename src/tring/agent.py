"""AgentSpec — the single agent contract.

One `AgentSpec` runs unchanged on any runtime. It is plain data (pydantic v2,
YAML-loadable): no callables, no vendor SDK types. Tool handlers are bound at
session start by name, keeping specs serializable and diffable.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class RuntimeMode(str, enum.Enum):
    CASCADE = "cascade"  # STT -> LLM -> TTS
    S2S = "s2s"  # speech-to-speech model end to end
    HYBRID = "hybrid"  # S2S conversation, cascade for tools/fallback


class ProviderSelection(BaseModel):
    """Which providers serve one pipeline slot set.

    Values are registry names, e.g. ``"faster_whisper"``, ``"ollama"``,
    ``"kokoro"``. For S2S mode only ``s2s`` is required.
    """

    stt: str | None = None
    llm: str | None = None
    tts: str | None = None
    s2s: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class RuntimeConfig(BaseModel):
    mode: RuntimeMode = RuntimeMode.CASCADE
    # Per-language provider routing. Key "default" is required; language codes
    # (BCP-47, e.g. "mr", "hi-IN") override it. Real callers code-switch, and
    # the best engine for one language is often the wrong one for another.
    routing: dict[str, ProviderSelection] = Field(
        default_factory=lambda: {"default": ProviderSelection()}
    )

    @field_validator("routing")
    @classmethod
    def _must_have_default(
        cls, v: dict[str, ProviderSelection]
    ) -> dict[str, ProviderSelection]:
        if "default" not in v:
            raise ValueError('routing must contain a "default" entry')
        return v

    def select(self, language: str | None) -> ProviderSelection:
        if language and language in self.routing:
            return self.routing[language]
        if language and "-" in language:
            base = language.split("-", 1)[0]
            if base in self.routing:
                return self.routing[base]
        return self.routing["default"]


class LanguagePolicy(BaseModel):
    primary: str = "en"
    allowed: list[str] = Field(default_factory=list)
    # When True, the language directive is re-asserted every turn via the
    # cache-safe language_lock primitive (never by mutating message history).
    lock: bool = True


class ToolDef(BaseModel):
    """A tool the agent may call.

    ``handler`` is a dotted-path or registry name resolved at session start.
    Choreography (waiting_message / spoken_mode / post_tool_response) is
    injected into the tool's *output schema* by primitives.choreography —
    it is a structural requirement, not per-tool configuration.
    """

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)  # JSON Schema
    handler: str | None = None
    timeout_seconds: float = 10.0
    choreographed: bool = True



class KnowledgeConfig(BaseModel):
    """Optional knowledge-base binding for an agent.

    ``provider`` names a registered knowledge provider; ``options`` go to its
    factory (chroma path, qdrant url, ...). ``speculative`` opts into
    retrieval kicked off from partial transcripts, so results are ready
    before the caller finishes the sentence.
    """

    provider: str
    options: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 4
    speculative: bool = True


class Limits(BaseModel):
    max_duration_seconds: float | None = 600.0
    max_cost: float | None = None
    cost_currency: str = "USD"
    max_tool_calls: int | None = 32


class AgentSpec(BaseModel):
    name: str
    version: str = "0.1.0"
    persona: str  # the system prompt, minus anything primitives inject
    greeting: str | None = None
    language: LanguagePolicy = Field(default_factory=LanguagePolicy)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    tools: list[ToolDef] = Field(default_factory=list)
    knowledge: KnowledgeConfig | None = None
    limits: Limits = Field(default_factory=Limits)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentSpec:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
