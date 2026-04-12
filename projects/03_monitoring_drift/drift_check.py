from __future__ import annotations

import numpy as np


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    qs = np.linspace(0, 1, bins + 1)
    cuts = np.quantile(expected, qs)
    cuts[0] = -np.inf
    cuts[-1] = np.inf

    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)

    e = e_counts / max(1, expected.size)
    a = a_counts / max(1, actual.size)

    e = np.clip(e, eps, 1)
    a = np.clip(a, eps, 1)

    return float(np.sum((a - e) * np.log(a / e)))


def main():
    rng = np.random.default_rng(0)

    reference = rng.normal(loc=0.0, scale=1.0, size=50_000)
    production_ok = rng.normal(loc=0.05, scale=1.0, size=10_000)
    production_drift = rng.normal(loc=0.6, scale=1.2, size=10_000)

    psi_ok = psi(reference, production_ok)
    psi_bad = psi(reference, production_drift)

    print(f"PSI(reference vs production_ok)={psi_ok:.4f}")
    print(f"PSI(reference vs production_drift)={psi_bad:.4f}")


if __name__ == "__main__":
    main()
