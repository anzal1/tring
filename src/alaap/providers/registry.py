"""Provider registry.

Providers register under a short name; specs reference them by that name.
Registration is import-cheap: entries hold a loader callable so heavy vendor
SDKs are only imported when a provider is actually selected.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any, Literal

Kind = Literal["stt", "llm", "tts", "s2s"]

_REGISTRY: dict[tuple[Kind, str], Callable[..., Any]] = {}


class UnknownProviderError(LookupError):
    pass


def register(kind: Kind, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a provider factory: ``@register("stt", "faster_whisper")``.

    The decorated callable receives the ProviderSelection ``options`` dict and
    returns a provider instance.
    """

    def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[(kind, name)] = factory
        return factory

    return deco


def create(kind: Kind, name: str, **options: Any) -> Any:
    try:
        factory = _REGISTRY[(kind, name)]
    except KeyError:
        available = sorted(n for k, n in _REGISTRY if k == kind)
        raise UnknownProviderError(
            f"no {kind} provider named {name!r}; available: {available}"
        ) from None
    return factory(**options)


def available(kind: Kind | None = None) -> list[tuple[str, str]]:
    return sorted((k, n) for (k, n) in _REGISTRY if kind is None or k == kind)


def _load_builtin() -> None:
    """Import built-in provider modules so their @register calls run.

    Modules guard their own heavy imports; importing the module only
    registers factories.
    """
    from importlib import import_module

    for mod in ("alaap.providers.local", "alaap.providers.cloud"):
        with contextlib.suppress(ImportError):
            import_module(mod)
