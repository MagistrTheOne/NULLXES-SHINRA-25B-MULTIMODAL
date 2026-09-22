"""Deterministic capability sampling and release gates; no dataset fetching."""

from dataclasses import dataclass
import random
import math


@dataclass(frozen=True)
class CapabilityCycle:
    name: str
    focused_datasets: tuple[str, str]
    max_updates: int = 650
    replay_fraction: float = 0.25
    evaluation_interval: int = 50

    def __post_init__(self):
        if len(self.focused_datasets) != 2 or len(set(self.focused_datasets)) != 2:
            raise ValueError("A capability cycle needs two distinct focused datasets")
        if not 1 <= self.max_updates <= 850:
            raise ValueError("Capability cycles are capped at 850 updates")
        if not 0 < self.replay_fraction < 1 or self.evaluation_interval < 1:
            raise ValueError("Invalid replay/evaluation contract")


class ReplaySampler:
    def __init__(self, focused, replay, replay_fraction=0.25, seed=0):
        if len(focused) != 2 or any(not pool for pool in focused) or not replay:
            raise ValueError("Two nonempty focus pools and a replay pool are required")
        if not 0 < replay_fraction < 1:
            raise ValueError("Invalid replay fraction")
        self.focused, self.replay = focused, replay
        self.fraction = replay_fraction
        self.rng = random.Random(seed)
        self.draws = 0

    def draw(self):
        self.draws += 1
        pool = self.replay if self.rng.random() < self.fraction else self.focused[self.rng.randrange(2)]
        return pool[self.rng.randrange(len(pool))]

    def state_dict(self):
        return {
            "draws": self.draws,
            "rng": self.rng.getstate(),
            "fraction": self.fraction,
            "sizes": [len(self.focused[0]), len(self.focused[1]), len(self.replay)],
        }

    def load_state_dict(self, state):
        if (
            state["sizes"] != [len(self.focused[0]), len(self.focused[1]), len(self.replay)]
            or state["fraction"] != self.fraction
        ):
            raise ValueError("Replay contract changed during resume")

        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, (list, tuple)) else value

        self.rng.setstate(tuples(state["rng"]))
        self.draws = state["draws"]


def regression_gate(baseline, current, max_drop):
    missing = set(baseline) - set(current)
    if missing:
        raise ValueError(f"Missing regression metrics: {sorted(missing)}")
    if any(not math.isfinite(value) for values in (baseline, current, max_drop) for value in values.values()):
        raise ValueError("Nonfinite regression metric")
    failures = {
        name: baseline[name] - current[name]
        for name in baseline
        if baseline[name] - current[name] > max_drop.get(name, 0.0)
    }
    return {"accepted": not failures, "regressions": failures}
