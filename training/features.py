from typing import Iterable, Mapping, Sequence


def vectorize_features(
    features: Mapping[str, float], feature_order: Sequence[str] | None = None
) -> list[float]:
    """Convert a feature mapping into a stable, ordered list of floats.

    When no feature order is provided, keys are sorted alphabetically to keep the
    input deterministic across requests and training runs.
    """

    names: Iterable[str] = feature_order if feature_order else sorted(features.keys())
    return [float(features.get(name, 0.0)) for name in names]


def ensure_feature_order(feature_order: Sequence[str] | None, observed: Iterable[str]) -> list[str]:
    """Return a concrete feature order, falling back to sorted observed names."""

    if feature_order:
        return list(feature_order)
    return sorted(observed)
