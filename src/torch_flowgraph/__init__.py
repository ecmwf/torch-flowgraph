# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0


"""torch-flowgraph: PyTorch computation graph tracing and visualization.

Public API
----------
:class:`GraphBase`
    Base class for all computation graph representations.  JSON-deserialised
    graphs are returned as :class:`GraphBase` instances.

:class:`Clustering`
    Abstract base class for reclustering strategies.

:class:`ClanClustering`
    Re-cluster using the McCreary clan-based graph-parsing algorithm.

:class:`LevelClustering`
    Keep only cluster boxes at a given depth; all nodes remain visible.

:class:`NameClustering`
    Keep only cluster boxes with specific names; all nodes remain visible.

:func:`apply_clan_clustering`
    Convenience wrapper around ``graph.recluster(ClanClustering())``.
"""

from . import patches as _patches  # noqa: F401 — apply torch meta-device patches
from ._post_import import on_import
from .clustering import ClanClustering
from .clustering import Clustering
from .clustering import LevelClustering
from .clustering import NameClustering
from .clustering import NoClustering
from .clustering import apply_clan_clustering
from .graph import GraphBase
from .validation import validate_graph

__all__ = [
    "GraphBase",
    "Clustering",
    "ClanClustering",
    "LevelClustering",
    "NameClustering",
    "NoClustering",
    "apply_clan_clustering",
    "validate_graph",
    "on_import",
]
