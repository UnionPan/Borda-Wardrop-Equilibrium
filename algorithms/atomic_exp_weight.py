"""
Population-level ExpWeight algorithm for atomic environments.

This is a thin wrapper around the non-atomic ExpWeight implementation that
clarifies its use with the `AtomicTrafficEnvironment`, where each OD pair acts
as a single homogeneous player controlling its strategic demand.
"""

from algorithms.exp_weight import ExpWeight


class PopulationExpWeight(ExpWeight):
    """
    ExpWeight dynamics for homogeneous OD populations.

    Identical to the non-atomic variant, but intended to be paired with the
    `AtomicTrafficEnvironment`, which adds fixed background flow to the
    strategic flow updated by this algorithm.
    """


# Backwards-compatible alias
AtomicExpWeight = PopulationExpWeight

__all__ = ["PopulationExpWeight", "AtomicExpWeight"]
