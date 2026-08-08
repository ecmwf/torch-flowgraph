# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Clustering strategies for computation graphs.

This module provides:

* :class:`Clustering` — abstract base class for all clustering strategies.
* :class:`ClanClustering` — re-clusters using the McCreary clan algorithm.
* :class:`LevelClustering` — shows only cluster boxes at a specific depth.
* :class:`NameClustering` — shows only cluster boxes with specific names.

All concrete classes implement :meth:`~Clustering.build_clusters`, which
receives a :class:`~torch_flowgraph.graph.GraphBase` and returns a new
:class:`~torch_flowgraph.clusters.Cluster` root.  Use
:meth:`~torch_flowgraph.graph.GraphBase.recluster` to apply a strategy::

    from torch_flowgraph.clustering import LevelClustering
    shallow = graph.recluster(LevelClustering(level=1))

Backward-compatibility shim
----------------------------
:func:`apply_clan_clustering` is kept as a convenience wrapper so that
existing code does not break::

    from torch_flowgraph import apply_clan_clustering
    clan_graph = apply_clan_clustering(graph)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .clusters import Cluster


class Clustering(ABC):
    """Abstract base class for graph reclustering strategies.

    Subclasses implement :meth:`build_clusters`, which returns a new
    :class:`~torch_flowgraph.clusters.Cluster` root built from *graph*.
    The graph's nodes and edges are *not* modified; only the cluster
    hierarchy changes.
    """

    @abstractmethod
    def build_clusters(self, graph) -> Cluster:
        """Build and return a new cluster root for *graph*.

        Parameters
        ----------
        graph:
            Source :class:`~torch_flowgraph.graph.GraphBase`.

        Returns
        -------
        Cluster
            Root of the new cluster hierarchy.  The returned tree must
            reference only node-ids that exist in *graph*.
        """


# ---------------------------------------------------------------------------
# Clan clustering
# ---------------------------------------------------------------------------

class ClanClustering(Clustering):
    """Re-cluster using the McCreary clan-based graph-parsing algorithm.

    The resulting cluster hierarchy reflects the *dataflow* decomposition
    of the graph (linear chains, independent sub-graphs, primitives) rather
    than the original ``nn.Module`` code hierarchy.

    Clan kinds assigned to clusters: ``LINEAR``, ``INDEPENDENT``,
    ``PRIMITIVE``.

    Trivial clusters are dissolved after parsing: single-node clusters
    have their node promoted to the parent, and single-child wrapper
    clusters (with no own nodes) are replaced by their only child.
    """

    def build_clusters(self, graph) -> Cluster:
        from .parsing import Parser

        G = graph.to_op_networkx()
        parser = Parser(G)
        parse_tree = parser.parse()
        root = _parse_tree_to_cluster(parse_tree, parent=None)
        root.update_cluster_nodes()
        _dissolve_leaf_clusters(root)
        return root


def _parse_tree_to_cluster(node, parent) -> Cluster:
    """Convert a :class:`~torch_flowgraph.parsing.ParseTree` node to a Cluster."""
    from .parsing import Kind

    kind_name = node.clan.kind.name if node.clan.kind != Kind.UNDEF else "UNKNOWN"
    cluster = Cluster(name=kind_name, parent=parent, nodes=list(node.clan.nodes))
    cluster.children = [_parse_tree_to_cluster(child, cluster) for child in node.children]
    return cluster


def _dissolve_leaf_clusters(cluster):
    """Remove trivial clusters, attaching their contents to the parent.

    A cluster is trivial when it contains only a single node (no
    sub-clusters) or only a single sub-cluster (no direct nodes).
    Runs repeatedly until no more trivial clusters remain.
    """
    changed = True
    while changed:
        changed = False
        kept = []
        for child in cluster.children:
            _dissolve_leaf_clusters(child)
            if not child.children and len(child.nodes) <= 1:
                # Single-node (or empty) cluster — promote nodes to parent.
                cluster.nodes.extend(child.nodes)
                changed = True
            elif len(child.children) == 1 and not child.nodes:
                # Single-child wrapper with no own nodes — replace with the child.
                grandchild = child.children[0]
                grandchild.parent = cluster
                kept.append(grandchild)
                changed = True
            else:
                kept.append(child)
        cluster.children = kept


# ---------------------------------------------------------------------------
# Selective clustering (level / name filters)
# ---------------------------------------------------------------------------

class _SelectiveClustering(Clustering):
    """Base for strategies that select a subset of existing cluster boxes.

    Matching clusters are kept as labeled boxes.  All other clusters are
    dropped: their nodes are absorbed into the flat root so the complete
    node set is still rendered — just without grouping boxes.
    """

    @abstractmethod
    def _find_matching(self, root_cluster: Cluster) -> list[Cluster]:
        """Return clusters from *root_cluster*'s tree that should be drawn."""

    def build_clusters(self, graph) -> Cluster:
        matching = self._find_matching(graph.clusters)

        # Collect all node-ids claimed by matching cluster subtrees.
        matching_nodes: set[int] = set()
        for c in matching:
            matching_nodes.update(c.all_nodes)

        # Root gets every node *not* in a matching subtree (shown ungrouped).
        root = Cluster(name=None)
        root.nodes = [n for n in graph.nodes if n not in matching_nodes]
        root.children = []

        for c in matching:
            child = Cluster(name=c.name, nodes=list(c.all_nodes), parent=root)
            root.children.append(child)

        return root


class LevelClustering(_SelectiveClustering):
    """Show only cluster boxes at one or more specific depths.

    All nodes remain in the graph; those not inside a matching cluster
    are rendered without a surrounding box.

    Parameters
    ----------
    level : int or collection of int
        Target cluster level(s), 0-indexed from the outermost real cluster.
        ``0`` means the top-level cluster boxes (the root's direct children),
        ``1`` their children, etc.  The nameless structural root is not a
        selectable level.  Pass a single ``int`` or any iterable of ``int``
        values to select multiple levels simultaneously.
    """

    def __init__(self, level) -> None:
        if isinstance(level, int):
            self.levels: frozenset[int] = frozenset([level])
        else:
            self.levels = frozenset(level)

    def _find_matching(self, root_cluster: Cluster) -> list[Cluster]:
        matching: list[Cluster] = []
        # User-facing level L corresponds to internal cluster depth L + 1,
        # since the root (depth 0) is a nameless structural container.
        target_depths = frozenset(level + 1 for level in self.levels)
        max_depth = max(target_depths) if target_depths else 0

        def _walk(c: Cluster) -> None:
            if c.depth in target_depths:
                matching.append(c)
                # Don't recurse: the whole subtree is kept in this box.
            elif c.depth < max_depth:
                for child in c.children:
                    _walk(child)

        _walk(root_cluster)
        return matching


class NameClustering(_SelectiveClustering):
    """Show only cluster boxes whose name is in *names*.

    All nodes remain in the graph; nodes not inside a named cluster are
    rendered without a surrounding box.

    Parameters
    ----------
    names : collection of str
        Cluster names to keep.  When a matching cluster is found, its
        children are not inspected (the whole subtree is included in the box).
    """

    def __init__(self, names) -> None:
        self.names: frozenset[str] = frozenset(names)

    def _find_matching(self, root_cluster: Cluster) -> list[Cluster]:
        matching: list[Cluster] = []

        def _walk(c: Cluster) -> None:
            if c.name in self.names:
                matching.append(c)
            else:
                for child in c.children:
                    _walk(child)

        _walk(root_cluster)
        return matching


# ---------------------------------------------------------------------------
# No-op clustering — clears all cluster boxes
# ---------------------------------------------------------------------------

class NoClustering(Clustering):
    """Remove all cluster boxes, rendering every node at the top level.

    The returned cluster hierarchy is a single flat root containing all
    nodes and no children.  The graph itself is unchanged; only the cluster
    boxes disappear from the rendered output.
    """

    def build_clusters(self, graph) -> Cluster:
        root = Cluster(name=None)
        root.nodes = list(graph.nodes)
        return root


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------

def apply_clan_clustering(graph):
    """Re-cluster *graph* using the McCreary clan algorithm.

    Convenience wrapper around ``graph.recluster(ClanClustering())``.
    Kept for backward compatibility with existing code.

    Parameters
    ----------
    graph:
        Source :class:`~torch_flowgraph.graph.GraphBase`.

    Returns
    -------
    GraphBase
        New graph with clan-derived cluster hierarchy.
    """
    return graph.recluster(ClanClustering())
