# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

"""Cluster data structure for hierarchical graph grouping.

A :class:`Cluster` represents a named group of graph nodes, arranged in a
tree via parent/children references.  Clusters are used to generate Graphviz
``subgraph cluster_*`` blocks and to control the collapsing/expansion of
module hierarchies during rendering.
"""

from functools import cached_property


class Cluster:
    """A named, hierarchical group of graph nodes.

    Clusters form a tree: each cluster has a *parent* and zero or more
    *children* (sub-clusters).  The *nodes* list holds the integer node-ids
    that belong directly to this cluster (not to its children).

    Attributes
    ----------
    name : str or None
        Human-readable cluster name shown in Graphviz DOT output as the
        ``label`` attribute.
    nodes : list[int]
        Integer node-ids that are direct members of this cluster.
    children : list[Cluster]
        Sub-clusters nested inside this cluster.
    parent : Cluster or None
        Parent cluster, or ``None`` for the root.
    depth : int
        Depth of this cluster in the tree (set by :class:`ModelGraph`
        during graph finalisation; ``-1`` until then).
    visible : bool or None
        Explicit visibility override.  ``None`` means visibility is
        determined automatically by :meth:`show_cluster`.
    kwargs : dict
        Additional Graphviz DOT attributes passed through to
        :attr:`dot_attributes`.
    """

    def __init__(self, *, name=None, parent=None, visible=None, nodes=None, children=None, **kwargs):
        """Initialise a Cluster.

        Parameters
        ----------
        name:
            Display label for the cluster in DOT output.
        parent:
            Parent :class:`Cluster`, or ``None`` for the root.
        visible:
            Explicit visibility flag.  When ``None`` the cluster is always
            shown (see :meth:`show_cluster`).
        nodes:
            Initial list of direct member node-ids.  Defaults to ``[]``.
        children:
            Initial list of child :class:`Cluster` instances.  Defaults to
            ``[]``.
        **kwargs:
            Additional Graphviz DOT attributes (e.g. ``bgcolor``,
            ``pencolor``).
        """
        self.name = name
        self.nodes = [] if nodes is None else nodes
        self.children = [] if children is None else children
        self.parent = parent
        self.depth = -1
        self.visible = visible
        self.is_repeated = False
        self.kwargs = kwargs

    @property
    def dot_attributes(self):
        """Graphviz DOT attribute dictionary for this cluster's subgraph block.

        Starts from any extra keyword arguments supplied at construction time
        and adds a ``"label"`` key when :attr:`name` is set.

        Returns
        -------
        dict
            Mapping of DOT attribute name to value.
        """
        attr = self.kwargs.copy()
        if self.name is not None:
            attr["label"] = self.name
        return attr

    def __len__(self):
        """Return the number of direct member nodes."""
        return len(self.nodes)

    def __repr__(self):
        return f"Cluster({self.name})"

    def as_dict(self):
        """Serialise the cluster tree to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Dictionary with ``"name"``, ``"nodes"``, and ``"children"``
            keys.  Children are recursively serialised.
        """
        return {
            "name": self.name,
            "nodes": self.nodes,
            "children": [c.as_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data, parent):
        """Deserialise a cluster tree from a dictionary.

        Handles both the current ``"children"`` key and the legacy
        ``"clusters"`` key for backwards compatibility.

        Parameters
        ----------
        data:
            Dictionary as produced by :meth:`as_dict`.
        parent:
            Parent :class:`Cluster` to assign to the newly created cluster
            (``None`` for the root).

        Returns
        -------
        Cluster
            Fully reconstructed cluster tree.
        """
        data = data.copy()
        children = data.pop("children", data.get("clusters", []))
        cluster = cls(parent=parent, **data)
        cluster.children = [cls.from_dict(c, cluster) for c in children]
        return cluster

    @cached_property
    def all_nodes(self):
        """Set of all node-ids contained anywhere in this cluster subtree.

        Computed once and cached.  Includes nodes from this cluster and all
        descendant clusters.

        Returns
        -------
        set[int]
        """
        result = set()

        def _all_nodes(c):
            for n in c.nodes:
                result.add(n)
            for c in c.children:
                _all_nodes(c)

        _all_nodes(self)
        return result

    def show_cluster(self, depth):
        """Determine whether this cluster should be rendered as a subgraph.

        Returns :attr:`visible` when it is explicitly set, otherwise always
        returns ``True``.

        Parameters
        ----------
        depth:
            Current rendering depth (not used in the default implementation
            but available for subclasses).

        Returns
        -------
        bool
        """
        if self.visible is not None:
            return self.visible

        return True

    def update_cluster_nodes(self):
        """Redistribute nodes so each node appears in exactly one cluster.

        Starting from the full set of nodes in this subtree (:attr:`all_nodes`),
        recursively assigns each node to the deepest cluster that contains it,
        removing it from all ancestor clusters' :attr:`nodes` lists.

        Returns
        -------
        Cluster
            *self*, mutated in-place.
        """
        nodes = set(self.all_nodes)

        def _update_cluster_nodes(c):
            nonlocal nodes

            for c in c.children:
                _update_cluster_nodes(c)

            c.nodes = list(set(c.nodes).intersection(nodes))
            nodes = nodes.difference(c.nodes)

            return c

        _update_cluster_nodes(self)

        return self
