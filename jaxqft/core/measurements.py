"""Inline measurement framework for production MCMC drivers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass
class MeasurementContext:
    """Mutable shared state passed through inline measurements.

    `cache` is intended for data hand-off between measurements in one step,
    while `state` can hold cross-step persistent helper data.
    """

    cache: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InlineMeasurement(Protocol):
    name: str
    every: int

    def run(self, q, theory, context: MeasurementContext) -> Mapping[str, float]:
        """Run measurement and return a flat mapping of scalar values."""


@dataclass
class PlaquetteMeasurement:
    """Compute the average plaquette at the current gauge field."""

    every: int = 1
    name: str = "plaquette"

    def run(self, q, theory, context: MeasurementContext) -> Mapping[str, float]:
        _ = context
        if not hasattr(theory, "average_plaquette"):
            raise AttributeError("Theory does not provide average_plaquette(q)")
        p = np.asarray(theory.average_plaquette(q), dtype=np.float64)
        return {"value": float(np.mean(p))}


def build_inline_measurements(specs: List[Mapping[str, Any]]) -> List[InlineMeasurement]:
    out: List[InlineMeasurement] = []
    for idx, spec in enumerate(specs):
        mtype = str(spec.get("type", "")).strip().lower()
        name = str(spec.get("name", "")).strip()
        every = int(spec.get("every", 1))
        if every <= 0:
            raise ValueError(f"Measurement[{idx}] has invalid every={every}; expected >=1")
        if mtype == "plaquette":
            out.append(PlaquetteMeasurement(every=every, name=(name or "plaquette")))
            continue
        raise ValueError(f"Unsupported inline measurement type: {mtype!r}")
    return out


def run_inline_measurements(
    measurements: List[InlineMeasurement],
    step: int,
    q,
    theory,
    context: MeasurementContext,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    context.cache = {}
    for m in measurements:
        every = max(1, int(getattr(m, "every", 1)))
        if int(step) % every != 0:
            continue
        vals = dict(m.run(q, theory, context))
        context.cache[str(m.name)] = vals
        records.append(
            {
                "step": int(step),
                "name": str(m.name),
                "values": vals,
            }
        )
    return records

