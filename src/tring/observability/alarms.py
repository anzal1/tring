"""Cache-regression detection: catching a silent prompt-cache collapse.

Prompt caching is invisible when it works and invisible when it breaks. A
provider-side change (a routing update, a context window that grew past the
cache boundary, a stray byte in a "cacheable" prefix) can drop
``cached_units`` to near zero while every other signal, latency, error rate,
even the total bill on a slow day, stays inside normal bounds. The honesty
rule in ``cost/meter.py`` makes the regression *visible* (``cached_units`` is
never netted away); this module makes it *actionable* by watching that number
move and calling out the moment it falls off a cliff.

The detector only ever looks at one ratio per ``(provider, component)``:
``cached_units / units`` from each ``CostRecorded`` line that reports caching
at all. Everything else, in particular anything that never reports
``cached_units``, never enters a window and can never trip the alarm. That is
what keeps a provider that has simply never used prompt caching from ever
being flagged: it has no baseline to regress from.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median

from tring.events import CostComponent, CostRecorded, SessionError, SessionEvent
from tring.session import CallSession

#: A (provider, component) key. Kept as a type alias because it appears on
#: every public surface of this module: the sample windows, the breach
#: tracker, and the alarm payload all key off exactly this pair.
CacheKey = tuple[str, CostComponent]


@dataclass(frozen=True)
class CacheRegressionAlarm:
    """One breach: the cache ratio for ``(provider, component)`` fell off its
    own recent baseline.

    ``baseline_ratio`` is the median of the window *before* the sample that
    tripped the alarm, so it reflects "what this provider's cache hit rate
    normally looks like" rather than being dragged down by the regression it
    is reporting on.
    """

    provider: str
    component: CostComponent
    at: float
    current_ratio: float
    baseline_ratio: float
    drop_fraction: float


class CacheRegressionDetector:
    """Feed it every ``CostRecorded`` event; it calls ``on_alarm`` on a breach.

    Args:
        on_alarm: invoked once per breach *episode*: the ratio dropping below
            threshold and staying there fires exactly once, not once per
            subsequent sample. It fires again only after the ratio recovers
            above threshold and then drops again, i.e. a genuinely new
            episode.
        window_size: how many recent per-key samples feed the rolling median.
            Bounded with a ``deque`` so long-running sessions do not grow this
            detector's memory without limit.
        min_samples: samples required before a key has a baseline at all.
            Below this, incoming ratios are recorded but never compared,
            which is the whole fix for the cold-start false positive: the
            first few cache-ratio samples of a call are exactly as likely to
            look "low" by chance as they are to reflect a real regression,
            and there is no history yet to say which.
        drop_fraction: how far below the baseline median counts as a breach,
            as a fraction of the baseline (0.5 means "ratio at or below half
            of baseline"). Configurable because how sharp a drop is
            noteworthy is a judgment call that differs by provider: a vendor
            whose cache ratio is naturally noisy needs a looser threshold
            than one that normally sits rock-steady.
        session: optional. When given, a breach also emits a ``SessionError``
            (``recoverable=True``) onto the session, so the regression shows
            up in the same event stream every other consumer (transports,
            dashboards, the eval harness) already watches, not only to
            whichever code happened to wire up ``on_alarm``.
    """

    def __init__(
        self,
        on_alarm: Callable[[CacheRegressionAlarm], None],
        window_size: int = 20,
        min_samples: int = 5,
        drop_fraction: float = 0.5,
        session: CallSession | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        if not 0.0 < drop_fraction < 1.0:
            raise ValueError("drop_fraction must be between 0 and 1, exclusive")
        self.on_alarm = on_alarm
        self.window_size = window_size
        self.min_samples = min_samples
        self.drop_fraction = drop_fraction
        self.session = session

        self._windows: dict[CacheKey, deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        # True while a key is currently in breach, so a sustained regression
        # raises exactly one alarm instead of one per sample.
        self._breached: dict[CacheKey, bool] = defaultdict(bool)

    def handle_event(self, event: SessionEvent) -> None:
        """Feed one session event in. Non-``CostRecorded`` events are ignored."""
        if not isinstance(event, CostRecorded):
            return
        if event.cached_units is None or event.units <= 0:
            # No caching signal on this line at all: nothing to compare, and
            # letting it into the window would silently blend "this provider
            # doesn't report caching for this unit" into "this provider's
            # cache ratio is zero", which are not the same fact.
            return

        ratio = event.cached_units / event.units
        key: CacheKey = (event.provider, event.component)
        window = self._windows[key]

        if len(window) >= self.min_samples:
            baseline = median(window)
            threshold = baseline * (1 - self.drop_fraction)
            if baseline > 0 and ratio <= threshold:
                if not self._breached[key]:
                    self._breached[key] = True
                    self._raise_alarm(event, key, ratio, baseline)
            else:
                self._breached[key] = False

        window.append(ratio)

    def _raise_alarm(
        self, event: CostRecorded, key: CacheKey, ratio: float, baseline: float
    ) -> None:
        provider, component = key
        alarm = CacheRegressionAlarm(
            provider=provider,
            component=component,
            at=event.at,
            current_ratio=ratio,
            baseline_ratio=baseline,
            drop_fraction=1 - (ratio / baseline),
        )
        self.on_alarm(alarm)
        if self.session is not None:
            self.session.emit(
                SessionError(
                    session_id=self.session.session_id,
                    at=event.at,
                    message=(
                        f"cache regression on {provider}/{component.value}: "
                        f"ratio dropped to {ratio:.3f} from a baseline of "
                        f"{baseline:.3f}"
                    ),
                    recoverable=True,
                )
            )


__all__ = ["CacheKey", "CacheRegressionAlarm", "CacheRegressionDetector"]
