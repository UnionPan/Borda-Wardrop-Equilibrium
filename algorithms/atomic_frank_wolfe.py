"""
Population-level Frank-Wolfe algorithm for atomic environments.

Wraps the standard Frank-Wolfe implementation so it can be used alongside the
`AtomicTrafficEnvironment`, which augments strategic flows with fixed background
traffic before evaluating costs.
"""

from algorithms.frank_wolfe import FrankWolfe


class PopulationFrankWolfe(FrankWolfe):
    """
    Frank-Wolfe projection for homogeneous OD populations.

    The behaviour mirrors the non-atomic algorithm; the distinction is that
    strategic demand represents only the controllable portion of each OD pair's
    flow, while the environment handles any background traffic automatically.
    """


# Backwards-compatible alias
AtomicFrankWolfe = PopulationFrankWolfe

__all__ = ["PopulationFrankWolfe", "AtomicFrankWolfe"]
