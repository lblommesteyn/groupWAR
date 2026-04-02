"""Optional ML components extracted from the legacy notebooks."""

from .graph import GraphLineupModel
from .stackelberg import ConstraintProjector, stackelberg_optimize

__all__ = [
    "ConstraintProjector",
    "GraphLineupModel",
    "stackelberg_optimize",
]
