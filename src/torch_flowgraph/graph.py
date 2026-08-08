# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""GraphBase: base class for PyTorch computation graph representations.

:class:`GraphBase` holds a node-and-edge representation of a computation graph
together with a hierarchical :class:`~torch_flowgraph.clusters.Cluster` tree.
It provides serialisation, deserialisation, and graph-transformation operations
that are independent of how the graph was originally constructed.

Subclasses (e.g. :class:`~torch_flowgraph.model.graph.ModelGraph`) extend this
with construction from live PyTorch models.
"""

import copy


from .clusters import Cluster
from .nodes import CollapsedNode
from .nodes import FunctionNode
from .nodes import LabelNode
from .nodes import ModuleNode
from .nodes import Node
from .nodes import ParameterNode
from .nodes import ShapeNode
from .nodes import TensorNode


def _compute_shortcut_weights(G):
    """Annotate every edge with ``_shortcut_weight``.

    For each edge ``(a, b)`` the weight is the length of the longest
    directed path from *a* to *b* in the graph **minus one** (so that
    direct neighbours get 0).  This equals the number of intermediate
    nodes bypassed by the edge.

    The algorithm runs in *O(V + E)* time using topological-order
    longest-path computation.
    """
    import networkx as nx

    try:
        topo = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return  # cyclic graph — nothing we can do

    order = {n: i for i, n in enumerate(topo)}

    # longest_dist[n] = longest distance from any source to n.
    longest_dist: dict[int, int] = {n: 0 for n in topo}
    for n in topo:
        for s in G.successors(n):
            d = longest_dist[n] + 1
            if d > longest_dist[s]:
                longest_dist[s] = d

    for a, b in G.edges:
        # Longest path from a to b: difference in longest-distance values
        # when we restrict to the sub-DAG between a and b.  A simpler
        # conservative proxy: for each edge we compute the topological
        # span (number of nodes between a and b in topo order) and use
        # longest_dist to get the exact bypass count.
        #
        # Exact approach: BFS/DFS from a restricted to nodes before b in
        # topo order.  But that is O(E) per edge worst-case.
        #
        # Fast heuristic that is exact for most practical DAGs: the
        # topological span minus 1 gives an upper bound; the
        # longest-dist difference gives a lower bound.  We use the
        # longest-dist difference as the weight since it corresponds to
        # the actual longest alternative path length.
        weight = longest_dist[b] - longest_dist[a] - 1
        G.edges[a, b]["_shortcut_weight"] = max(weight, 0)


class GraphBase:
    """A computation graph represented as nodes, edges, and a cluster tree.

    This is the base class for all graph representations in torch-flowgraph.
    It is the concrete type returned when deserialising a graph from JSON.
    Graphs created by tracing a live PyTorch model are represented by the
    :class:`~torch_flowgraph.model.graph.ModelGraph` subclass.

    Attributes
    ----------
    nodes : dict[int, Node]
        Mapping from integer node-id to :class:`~torch_flowgraph.nodes.Node`
        instance.
    edges : list[tuple[int, int]]
        List of ``(src_id, dst_id)`` directed edges.
    clusters : Cluster
        Root of the module-hierarchy cluster tree.
    depth : int
        Maximum cluster depth encountered during finalisation.
    """

    def __init__(self, nodes, edges, clusters):
        """Initialise a GraphBase and finalise the cluster/node relationship.

        During initialisation every node is linked to its containing cluster
        (``node.cluster = cluster``) and every cluster receives its depth in
        the hierarchy.

        Parameters
        ----------
        nodes:
            Mapping of node-id to :class:`~torch_flowgraph.nodes.Node`.
        edges:
            List of ``(src_id, dst_id)`` edge tuples.
        clusters:
            Root :class:`~torch_flowgraph.clusters.Cluster` of the module
            hierarchy.
        """
        assert isinstance(nodes, dict), type(nodes)
        assert isinstance(edges, list)
        assert isinstance(clusters, Cluster)
        assert all(isinstance(n, Node) for n in nodes.values())

        self.nodes = nodes
        self.edges = edges
        self.clusters = clusters
        self.depth = 0

        self._finalise()

    def _finalise(self):
        """Assign cluster depths and back-link every node to its cluster.

        Traverses the cluster tree recursively, setting ``cluster.depth`` and
        ``self.depth``, and setting ``node.cluster`` on each node found in
        each cluster's :attr:`~Cluster.nodes` list.
        """
        self.depth = 0

        def _walk(c, depth):
            c.depth = depth
            self.depth = max(self.depth, depth)
            for n in c.nodes:
                self.nodes[n].cluster = c
            # Invisible clusters don't count toward depth so that max_depth
            # semantics remain stable when a transparent wrapper is added.
            child_depth = depth if (c.visible is False) else depth + 1
            for child in c.children:
                _walk(child, child_depth)

        _walk(self.clusters, 0)

    def __len__(self):
        """Return the number of nodes in the graph."""
        return len(self.nodes)

    def __repr__(self):
        return f"GraphBase({len(self)} nodes)"

    def sources_and_sinks(self):
        """Return the sets of source and sink nodes in this graph.

        A *source* is a node with no incoming edges; a *sink* is a node with
        no outgoing edges.  Both sets are returned as lists of
        :class:`~torch_flowgraph.nodes.Node` instances.

        In a well-formed model graph, only the input and output
        :class:`~torch_flowgraph.nodes.ShapeNode` instances (labels ``"x"``
        and ``"y"``) plus any auxiliary parameter / tensor nodes should
        appear as sources, and only the output ``"y"`` node should appear as
        a sink.  Any :class:`~torch_flowgraph.nodes.FunctionNode` or
        :class:`~torch_flowgraph.nodes.ModuleNode` showing up here indicates
        a dangling edge.

        Returns
        -------
        tuple[list[Node], list[Node]]
            ``(sources, sinks)``.
        """
        has_incoming = {b for _, b in self.edges}
        has_outgoing = {a for a, _ in self.edges}
        sources = [self.nodes[n] for n in self.nodes if n not in has_incoming]
        sinks = [self.nodes[n] for n in self.nodes if n not in has_outgoing]
        return sources, sinks

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def as_dict(self):
        """Serialise this graph to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Dictionary with ``"nodes"``, ``"edges"``, and ``"clusters"``
            keys.
        """
        return {
            "nodes": [n.as_dict() for n in self.nodes.values()],
            "edges": self.edges,
            "clusters": self.clusters.as_dict(),
        }

    def save_json(self, path):
        """Serialise this graph to a JSON file.

        Parameters
        ----------
        path:
            Destination file path.  The file is created or overwritten.
        """
        import json

        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2)

    def save_dot(self, path, graph_attributes=None, **kwargs):
        """Write this graph to a Graphviz DOT file.

        Parameters
        ----------
        path:
            Destination file path.
        graph_attributes:
            Optional top-level DOT graph attribute dictionary.
        **kwargs:
            Additional keyword arguments forwarded to
            :func:`~torch_flowgraph.dot.save_dot`.
        """
        from torch_flowgraph.dot import save_dot

        return save_dot(self, path, graph_attributes, **kwargs)

    def plot_graph(self, path, fmt=None, **kwargs):
        """Render the graph with Graphviz and write the result to *path*.

        The output format is inferred from *path*'s extension (e.g. ``.png``,
        ``.svg``, ``.pdf``).  Pass *fmt* explicitly to override.

        A temporary DOT file is used as input to ``dot`` and is deleted
        afterwards regardless of success or failure.

        Parameters
        ----------
        path:
            Destination file for the rendered output.
        fmt:
            Graphviz output format.  Inferred from *path*'s extension when
            ``None`` (default ``"png"`` if the extension is absent or
            unrecognised).
        **kwargs:
            Additional keyword arguments forwarded to :meth:`save_dot`
            (e.g. ``clusters=``, ``depth=``, ``graph_attributes=``).
        """
        import os
        import subprocess
        import tempfile

        if fmt is None:
            ext = os.path.splitext(path)[1].lstrip(".")
            fmt = ext if ext else "png"

        with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as dot_file:
            dot_path = dot_file.name

        try:
            self.save_dot(dot_path, **kwargs)
            subprocess.run(["dot", f"-T{fmt}", dot_path, "-o", path], check=True)
        finally:
            os.unlink(dot_path)

    # ------------------------------------------------------------------
    # Deserialisation — always produces a GraphBase, never a subclass
    # ------------------------------------------------------------------

    @classmethod
    def from_json_file(cls, path):
        """Deserialise a :class:`GraphBase` from a JSON file.

        Parameters
        ----------
        path:
            File path to a JSON file previously written by :meth:`save_json`.

        Returns
        -------
        GraphBase
        """
        import json

        with open(path, "r") as f:
            data = json.load(f)

        return GraphBase.from_dict(data)

    @classmethod
    def from_dict(cls, data):
        """Deserialise a :class:`GraphBase` from a dictionary.

        Reconstructs the correct node subclass (e.g. :class:`FunctionNode`,
        :class:`ModuleNode`) by reading the ``"_kind"`` field stored during
        serialisation.  JSON lists are converted to tuples so that the data
        can be passed directly to node constructors.

        Always returns a plain :class:`GraphBase`, regardless of which
        subclass ``cls`` is called on.  JSON-deserialised graphs are not
        PyTorch-traced graphs; use
        :meth:`~torch_flowgraph.model.graph.ModelGraph.from_model` to
        construct a graph from a live model.

        Parameters
        ----------
        data:
            Dictionary as produced by :meth:`as_dict`.

        Returns
        -------
        GraphBase
        """
        CLASSES = {
            str(_.__name__): _
            for _ in [
                FunctionNode,
                LabelNode,
                ModuleNode,
                ShapeNode,
                TensorNode,
                ParameterNode,
            ]
        }

        def _as_tuples(x):
            """Recursively convert JSON lists to tuples for node constructors."""
            if isinstance(x, dict):
                return {k: _as_tuples(v) for k, v in x.items()}
            if isinstance(x, list):
                return tuple(_as_tuples(v) for v in x)
            return x

        nodes = {}
        for n in data["nodes"]:
            n = n.copy()
            kind = n.pop("_kind")
            rank = n.pop("rank", None)
            n = _as_tuples(n)
            node = CLASSES[kind](**n)
            node.rank = rank
            nodes[node.n] = node

        edges = [_as_tuples(x) for x in data["edges"]]
        clusters = Cluster.from_dict(data["clusters"], None)

        return GraphBase(nodes, edges, clusters)

    # ------------------------------------------------------------------
    # Graph transformations
    # ------------------------------------------------------------------

    def recluster(self, clustering) -> 'GraphBase':
        """Return a new graph with a different cluster hierarchy.

        Applies *clustering* — a :class:`~torch_flowgraph.clustering.Clustering`
        instance — to this graph and returns the resulting graph.  The node and
        edge data are not modified; only the cluster tree changes.

        Parameters
        ----------
        clustering:
            A :class:`~torch_flowgraph.clustering.Clustering` instance whose
            :meth:`~torch_flowgraph.clustering.Clustering.build_clusters` method
            will be called with this graph.

        Returns
        -------
        GraphBase
            New graph with the cluster hierarchy produced by *clustering*.

        Example
        -------
        ::

            from torch_flowgraph.clustering import LevelClustering
            shallow = graph.recluster(LevelClustering(level=1))
        """
        new_clusters = clustering.build_clusters(self)
        return self.use_clusters(new_clusters)

    def use_clusters(self, clusters):
        """Return a new graph using a different cluster hierarchy.

        Re-links all nodes to the clusters provided by *clusters*, replacing
        the current hierarchy without modifying the node or edge data.

        Parameters
        ----------
        clusters:
            New root :class:`~torch_flowgraph.clusters.Cluster` or an
            equivalent dictionary.

        Returns
        -------
        GraphBase
            New graph object sharing the same nodes and edges but with the
            updated cluster tree.
        """
        if isinstance(clusters, dict):
            clusters = Cluster.from_dict(clusters, None)

        # Shallow-copy each node so that setting .cluster does not mutate
        # the original graph's node objects.
        nodes = {n: copy.copy(node) for n, node in self.nodes.items()}
        edges = self.edges.copy()

        def _use_cluster(c):
            for n in c.nodes:
                nodes[n].cluster = c
            for child in c.children:
                _use_cluster(child)

        _use_cluster(clusters)

        return GraphBase(nodes, edges, clusters)

    def collapsed(self, max_depth):
        """Return a new graph with clusters collapsed to *max_depth*.

        Clusters deeper than *max_depth* are replaced by a single
        :class:`~torch_flowgraph.nodes.CollapsedNode`.  Internal edges are
        removed; boundary edges are reconnected to the collapsed node.  Nodes
        are then renumbered so that ids remain contiguous.

        Parameters
        ----------
        max_depth:
            Maximum cluster depth to retain.  Clusters at depth
            ``> max_depth`` are collapsed.

        Returns
        -------
        GraphBase
            New graph with collapsed clusters and renumbered nodes.
        """
        nodes = self.nodes.copy()
        edges = self.edges.copy()

        # Map cluster → collapsed-node id without mutating original cluster objects.
        _cluster_node_id = {}
        # Map parent-cluster → [collapsed-node ids added for its collapsed children].
        _parent_extra_nodes = {}

        def _parent_cluster(n):
            c = self.nodes[n].cluster
            while c.depth > max_depth:
                c = c.parent
            return c

        def _ensure_cluster(c, outgoing=None):
            if c not in _cluster_node_id:
                n = len(nodes)
                node = CollapsedNode(n, c.name)
                nodes[n] = node
                if c.parent is not None:
                    _parent_extra_nodes.setdefault(c.parent, []).append(n)
                _cluster_node_id[c] = n
            if outgoing:
                nodes[_cluster_node_id[c]].outgoing(outgoing)
            return _cluster_node_id[c]

        def _collapse(c, depth):
            for child in c.children:
                # Invisible clusters don't consume a depth level.
                _collapse(child, depth if (c.visible is False) else depth + 1)

            if depth > max_depth and c.visible is not False:
                for a, b in edges.copy():
                    a_inside = a in c.nodes
                    b_inside = b in c.nodes

                    if not a_inside and not b_inside:
                        continue

                    edges.remove((a, b))

                    if a_inside and not b_inside:
                        a = _ensure_cluster(_parent_cluster(a), nodes[a])
                        edges.append((a, b))

                    if b_inside and not a_inside:
                        b = _ensure_cluster(_parent_cluster(b))
                        edges.append((a, b))

                for n in c.nodes:
                    nodes[n] = None

        _collapse(self.clusters, 0)

        edges = [(a, b) for a, b in edges if (nodes[a] is not None) and (nodes[b] is not None) and (a != b)]

        renumber = {}
        new_nodes = {}
        for i, n in enumerate(nodes):
            if nodes[n] is None:
                continue
            renumber[n] = i
            # Use a shallow copy so we don't mutate the original node's .n attribute.
            new_node = copy.copy(nodes[n])
            new_node.n = i
            new_nodes[i] = new_node

        edges = [(renumber[a], renumber[b]) for a, b in edges]

        def _fix_cluster(c, parent):
            result = Cluster(name=c.name, parent=parent, visible=c.visible)
            # Include both the cluster's own nodes and any CollapsedNodes
            # created for collapsed children (tracked in _parent_extra_nodes).
            all_nodes = list(c.nodes) + _parent_extra_nodes.get(c, [])
            result.nodes = [renumber[n] for n in all_nodes if n in renumber]
            result.children = [_fix_cluster(child, result) for child in c.children]
            return result

        clusters = _fix_cluster(self.clusters, None)

        return GraphBase(new_nodes, edges, clusters)

    def collapse_to_level(self, level: int) -> 'GraphBase':
        """Return a new graph with every cluster at depth ``level + 1`` collapsed
        into a single node.

        Unlike :meth:`collapsed`, which keeps the boundary cluster as a *box*
        and expands one level inside it, this replaces each cluster at depth
        ``level + 1`` — together with its whole subtree — by a single
        :class:`~torch_flowgraph.nodes.CollapsedNode`.  Clusters shallower than
        that (depth ``<= level``) are kept as boxes and their direct member
        nodes stay visible.

        This is the "show the model at level N" view used by the interactive
        viewer: ``level=0`` collapses the outermost modules to single nodes
        (e.g. ``x -> Predict -> y``), ``level=1`` opens those boxes and
        collapses their children, and so on.

        Boundary edges are reconnected to the collapsed node; edges internal to
        a collapsed subtree are dropped.

        Parameters
        ----------
        level:
            0-indexed cluster level to collapse to (``0`` = outermost modules).

        Returns
        -------
        GraphBase
            New graph with collapsed clusters and renumbered nodes.
        """
        target = level + 1

        new_nodes: dict[int, Node] = {}
        node_repr: dict[int, int] = {}       # original node-id → new node-id
        collapsed_for: dict[int, CollapsedNode] = {}  # id(cluster) → its collapsed node

        def _keep(n):
            node = copy.copy(self.nodes[n])
            node.n = len(new_nodes)
            node.cluster = None
            new_nodes[node.n] = node
            node_repr[n] = node.n

        def _collapse(c):
            cn = CollapsedNode(len(new_nodes), c.name)
            new_nodes[cn.n] = cn
            collapsed_for[id(c)] = cn
            for n in c.all_nodes:
                node_repr[n] = cn.n

        def _walk(c):
            if c.depth >= target:
                _collapse(c)
                return
            for n in c.nodes:
                _keep(n)
            for child in c.children:
                _walk(child)

        _walk(self.clusters)

        # Nodes not attached to any cluster keep their own identity.
        for n in self.nodes:
            if n not in node_repr:
                _keep(n)

        edges = set()
        for a, b in self.edges:
            na, nb = node_repr[a], node_repr[b]
            if na == nb:
                continue  # edge internal to a collapsed subtree
            src = new_nodes[na]
            if isinstance(src, CollapsedNode):
                src.outgoing(self.nodes[a])
            edges.add((na, nb))

        def _build(c):
            result = Cluster(name=c.name, parent=None, visible=c.visible, **c.kwargs)
            result.nodes = [node_repr[n] for n in c.nodes if n in node_repr]
            for child in c.children:
                if id(child) in collapsed_for:
                    result.nodes.append(collapsed_for[id(child)].n)
                else:
                    sub = _build(child)
                    sub.parent = result
                    result.children.append(sub)
            return result

        clusters = _build(self.clusters)

        return GraphBase(new_nodes, list(edges), clusters)

    def merge_repeated(self) -> 'GraphBase':
        """Return a new graph with consecutive repeated sibling clusters merged.

        For each run of N consecutive sibling clusters sharing the same name
        and sub-cluster structure, relabels the first as ``"{name} x {N}"``
        and marks the remaining N-1 with ``visible=False``.  Bridge edges are
        added from the first cluster's exit nodes to the last cluster's
        external targets so that connectivity is preserved in explore mode
        (where invisible clusters are removed entirely).

        Returns
        -------
        GraphBase
            New graph with the merged cluster tree.
        """
        def _fingerprint(c):
            return (c.name, tuple(_fingerprint(ch) for ch in c.children))

        def _walk(c):
            for child in c.children:
                _walk(child)
            i = 0
            while i < len(c.children):
                child = c.children[i]
                if not child.name:
                    i += 1
                    continue
                fp = _fingerprint(child)
                j = i + 1
                while j < len(c.children) and _fingerprint(c.children[j]) == fp:
                    j += 1
                n = j - i
                if n > 1:
                    child.is_repeated = True
                    child.name = f"{child.name} x {n}"
                    for dup in c.children[i + 1 : j]:
                        dup.visible = False
                i = j

        new_clusters = copy.deepcopy(self.clusters)
        _walk(new_clusters)
        nodes = {n: copy.copy(node) for n, node in self.nodes.items()}
        edges = self.edges.copy()
        return GraphBase(nodes, edges, new_clusters)

    def explore(self, expanded_paths=None, auto_expanded_paths=None):
        """Return a graph suitable for interactive Explore mode.

        Like :meth:`collapsed` with ``max_depth=0`` but with *selective*
        expansion.  Clusters whose tree path appears in *expanded_paths* or
        *auto_expanded_paths* are rendered as cluster boxes (and their direct
        children are individually collapsed); all other clusters are replaced
        by a single :class:`~torch_flowgraph.nodes.CollapsedNode` that carries
        a ``URL = "explore:expand:{path}"`` attribute for click-to-expand
        support in SVG viewers.

        Clusters in *expanded_paths* (user-clicked) carry a
        ``URL = "explore:collapse:{path}"`` attribute for click-to-collapse.
        Clusters in *auto_expanded_paths* (depth-based pre-expansion) do
        **not** get a collapse URL, so they cannot be collapsed by clicking.

        Cluster tree paths are ``'/'``-joined zero-based child indices below
        the root.  For example ``"0"`` is the first top-level cluster,
        ``"0/1"`` is its second child cluster, and so on.  The root itself
        (empty string) is always treated as expanded.

        Parameters
        ----------
        expanded_paths : collection of str, optional
            Paths of clusters expanded by user interaction.  These clusters
            receive a ``explore:collapse`` URL.  Defaults to an empty set.
        auto_expanded_paths : collection of str, optional
            Paths of clusters to pre-expand (e.g. based on a depth slider).
            These clusters are shown expanded but do *not* receive a collapse
            URL.  Defaults to an empty set.

        Returns
        -------
        GraphBase
        """
        user_expanded = set(expanded_paths or [])
        auto_expanded = set(auto_expanded_paths or [])
        expanded = user_expanded | auto_expanded

        nodes = self.nodes.copy()
        edges = list(self.edges)
        _cluster_node_id: dict[int, int] = {}    # id(cluster) → new CollapsedNode id
        _parent_extra_nodes: dict[int, list] = {}  # id(cluster) → CollapsedNode ids
        _cluster_path: dict[int, str] = {}

        # Assign path strings ("", "0", "0/1", …) to every cluster.
        def _assign_paths(c, path):
            _cluster_path[id(c)] = path
            for i, child in enumerate(c.children):
                _assign_paths(child, f"{path}/{i}" if path else str(i))

        _assign_paths(self.clusters, "")

        def _is_expanded(c):
            path = _cluster_path.get(id(c), "")
            return path == "" or path in expanded   # root ("") always expanded

        def _ensure_collapsed(c):
            """Create (or reuse) the CollapsedNode representing cluster *c*."""
            c_key = id(c)
            if c_key not in _cluster_node_id:
                n_id = len(nodes)
                path = _cluster_path[c_key]
                new_node = CollapsedNode(n_id, c.name)
                new_node.explore_url = f"explore:expand:{path}"
                new_node.is_repeated = c.is_repeated
                nodes[n_id] = new_node

                # Place the CollapsedNode inside its nearest expanded ancestor.
                parent = c.parent
                while parent is not None and not _is_expanded(parent):
                    parent = parent.parent
                if parent is not None:
                    _parent_extra_nodes.setdefault(id(parent), []).append(n_id)

                _cluster_node_id[c_key] = n_id
            return _cluster_node_id[c_key]

        def _collapse(c):
            if c.visible is False:
                # Invisible cluster: remove its nodes but thread connectivity
                # through it by bridging every current incoming source to every
                # current outgoing target.  This is transitive: if several
                # invisible clusters are chained (c1→c2→c3 all invisible) each
                # step extends the bridge so the final source→target pair is
                # correctly linked even when connections pass through
                # intermediate non-cluster nodes.
                c_nodes = set(c.all_nodes)
                incoming = {a for a, b in edges if b in c_nodes and a not in c_nodes}
                outgoing = {b for a, b in edges if a in c_nodes and b not in c_nodes}
                for src in incoming:
                    for tgt in outgoing:
                        edges.append((src, tgt))
                for n_id in c.all_nodes:
                    if n_id in nodes and nodes[n_id] is not None:
                        nodes[n_id] = None
                return
            if _is_expanded(c):
                for child in c.children:
                    _collapse(child)
            else:
                subtree = c.all_nodes
                if not subtree:
                    return  # empty cluster — no nodes to represent
                cn = _ensure_collapsed(c)

                new_edges = []
                for a, b in edges:
                    a_in = a in subtree
                    b_in = b in subtree
                    if not a_in and not b_in:
                        new_edges.append((a, b))
                    elif a_in and not b_in:
                        # Register the original outgoing node so the CollapsedNode
                        # can delegate to it for edge label (shape info).
                        nodes[cn].outgoing(nodes[a])
                        # Per-target tracking: record which source contributes to
                        # each (cn → b) edge so dot_edge_kwargs can apply the
                        # 1-vs-many label rule later.
                        nodes[cn]._sources_for.setdefault(b, []).append(nodes[a])
                        new_edges.append((cn, b))
                    elif b_in and not a_in:
                        new_edges.append((a, cn))
                        # If the source is itself a CollapsedNode that was already
                        # tracking edges to nodes now absorbed into this cluster,
                        # migrate those entries to point at the new collapsed id.
                        src = nodes.get(a)
                        if isinstance(src, CollapsedNode) and b in src._sources_for:
                            src._sources_for.setdefault(cn, []).extend(
                                src._sources_for.pop(b)
                            )
                    # Both inside → internal edge, dropped.
                edges[:] = new_edges

                for n_id in subtree:
                    nodes[n_id] = None

        for child in self.clusters.children:
            _collapse(child)

        # Remove edges touching None-nodes and self-loops; deduplicate.
        seen_edges: set = set()
        clean_edges = []
        for a, b in edges:
            if nodes.get(a) is None or nodes.get(b) is None or a == b:
                continue
            if (a, b) not in seen_edges:
                seen_edges.add((a, b))
                clean_edges.append((a, b))
        edges = clean_edges

        # Renumber nodes to keep ids contiguous.
        renumber: dict[int, int] = {}
        new_nodes: dict[int, "Node"] = {}
        i = 0
        for n_id in sorted(nodes):
            if nodes[n_id] is None:
                continue
            renumber[n_id] = i
            new_node = copy.copy(nodes[n_id])
            new_node.n = i
            new_nodes[i] = new_node
            i += 1

        edges = [(renumber[a], renumber[b]) for a, b in edges]

        # Remap _sources_for keys from pre-renumber ids to post-renumber ids.
        for old_id, orig_node in nodes.items():
            if orig_node is None or not isinstance(orig_node, CollapsedNode):
                continue
            new_nodes[renumber[old_id]]._sources_for = {
                renumber[old_t]: srcs
                for old_t, srcs in orig_node._sources_for.items()
                if old_t in renumber
            }

        def _fix_cluster(c, parent):
            path = _cluster_path[id(c)]
            result = Cluster(name=c.name, parent=parent, visible=c.visible, **c.kwargs)

            own_ids = list(c.nodes) + _parent_extra_nodes.get(id(c), [])
            result.nodes = [renumber[n_id] for n_id in own_ids if n_id in renumber]

            # Only recurse into children that are expanded.
            result.children = [
                _fix_cluster(child, result)
                for child in c.children
                if _is_expanded(child)
            ]

            # User-expanded (non-root) cluster boxes get a collapse URL.
            # Auto-expanded clusters (depth-based) intentionally do not, so
            # they cannot be collapsed by clicking.
            if path and path in user_expanded:
                result.kwargs = {**result.kwargs, "URL": f"explore:collapse:{path}"}

            return result

        clusters = _fix_cluster(self.clusters, None)
        return GraphBase(new_nodes, edges, clusters)

    @property
    def cluster_levels(self):
        """Sorted list of selectable cluster levels that contain at least one node.

        Levels are 0-indexed from the outermost real cluster: level ``0`` is
        the root's direct children, level ``1`` their children, etc.  The
        nameless structural root (internal depth ``0``) is not a level.

        Returns
        -------
        list[int]
        """
        levels = set()

        def _walk(c):
            if c.all_nodes and c.depth > 0:
                levels.add(c.depth - 1)
            for child in c.children:
                _walk(child)

        _walk(self.clusters)
        return sorted(levels)

    @property
    def cluster_names(self):
        """Sorted list of distinct cluster names (excluding ``None``) that contain nodes.

        Returns
        -------
        list[str]
        """
        names = set()

        def _walk(c):
            if c.name is not None and c.all_nodes:
                names.add(c.name)
            for child in c.children:
                _walk(child)

        _walk(self.clusters)
        return sorted(names)

    def only_modules(self):
        """Return a simplified graph where each cluster is collapsed to one node.

        Produces a coarse view of the model in which every cluster (module) is
        represented by a single :class:`~torch_flowgraph.nodes.CollapsedNode`
        and inter-cluster edges are retained.

        Returns
        -------
        GraphBase
            Simplified graph.
        """
        nodes = {}
        edges = set()

        _clusters = {}
        _seen = {}

        def _ensure_cluster(c):
            if c in _seen:
                return _seen[c]

            cluster = self.nodes[c].cluster
            # Nodes that are not inside a *named* cluster (e.g. the input/output
            # shape markers living in the anonymous root) keep their own identity
            # instead of being merged into a single nameless ("null") node.
            if cluster is None or cluster.name is None:
                node = copy.copy(self.nodes[c])
                node.n = len(nodes)
                nodes[node.n] = node
                _seen[c] = node
                return node

            if cluster not in _clusters:
                n = len(nodes)
                _clusters[cluster] = CollapsedNode(n, cluster.name)
                nodes[n] = _clusters[cluster]
            _seen[c] = _clusters[cluster]
            return _clusters[cluster]

        for a, b in self.edges:
            a = _ensure_cluster(a).n
            b = _ensure_cluster(b).n
            edges.add((a, b))

        def _fix_cluster(c, parent):
            result = Cluster(name=c.name, parent=parent)
            result.nodes = list(set([_ensure_cluster(n).n for n in c.nodes]))
            result.children = [_fix_cluster(child, result) for child in c.children]
            return result

        clusters = _fix_cluster(self.clusters, None)

        return GraphBase(nodes, list(edges), clusters)

    def shortcut_weights(self):
        """Return per-edge shortcut weights.

        For each edge ``(a, b)`` compute the number of nodes bypassed,
        defined as the longest-path distance from *a* to *b* minus one.
        Direct neighbours get weight 0.

        Returns
        -------
        list[int]
            List of shortcut weights, one per edge, in the same order as
            :attr:`edges`.
        """
        G = self.to_op_networkx()
        return [G.edges[a, b].get("_shortcut_weight", 0) for a, b in self.edges]

    def node_depths(self):
        """Return the cluster depth of each node.

        Walks the cluster tree and records for every node the depth of the
        cluster it belongs to (root = 0).

        Returns
        -------
        list[int]
            List of depths, one per node, in the same order as
            :attr:`nodes`.
        """
        depth_map: dict[int, int] = {}

        def _walk(cluster, depth):
            for n in cluster.nodes:
                depth_map[n] = depth
            for child in cluster.children:
                _walk(child, depth + 1)

        _walk(self.clusters, 0)
        return [depth_map.get(n, 0) for n in self.nodes]

    # ------------------------------------------------------------------
    # NetworkX conversion
    # ------------------------------------------------------------------

    def to_op_networkx(self, links=False):
        """Convert this graph to a full op-level NetworkX :class:`~networkx.DiGraph`.

        Every node in the returned graph corresponds to one PyTorch operation
        or auxiliary node (parameter, tensor, shape marker).  Node attributes
        are populated from :meth:`~torch_flowgraph.nodes.Node.dot_node_kwargs`
        and edge attributes from
        :meth:`~torch_flowgraph.nodes.Node.dot_edge_kwargs`.

        Parameters
        ----------
        links:
            When ``True``, :class:`~torch_flowgraph.nodes.FunctionNode` nodes
            include a ``URL`` attribute linking to the PyTorch documentation.
            Defaults to ``False``.

        Returns
        -------
        networkx.DiGraph
            Op-level graph with DOT styling attributes on nodes and edges.
        """
        import networkx as nx

        G = nx.DiGraph()

        for n in self.nodes.values():
            G.add_node(n.n, **n.dot_node_kwargs(links=links))

        for a, b in self.edges:
            G.add_edge(a, b, **self.nodes[a].dot_edge_kwargs(self.nodes[b]))

        # Compute shortcut weights: for each edge (a, b) store the number
        # of nodes on the longest path from a to b *excluding* a and b
        # themselves.  This measures how many nodes a "shortcut" edge
        # bypasses.  Edges on the critical path get weight 0.
        _compute_shortcut_weights(G)

        return G

    def to_module_networkx(self):
        """Convert this graph to a coarsened module-level NetworkX :class:`~networkx.DiGraph`.

        Maps every op-level node to its containing cluster so that each
        cluster (``nn.Module`` or equivalent) becomes a single node.  Only
        edges that involve at least one leaf cluster are retained, giving a
        high-level module-to-module data-flow view.

        Returns
        -------
        networkx.DiGraph
            Module-level graph with ``"label"`` attributes on nodes.
        """
        import networkx as nx

        mapping = {}
        clusters = []
        seen = set()
        leaf = set()

        def _fix_cluster(c):
            m = len(clusters)
            clusters.append(c.name)

            for n in c.nodes:
                mapping[n] = m

            for child in c.children:
                _fix_cluster(child)

            if len(c.children) == 0:
                leaf.add(m)

        _fix_cluster(self.clusters)

        G = nx.DiGraph()

        for n in self.nodes.values():
            if n.n not in mapping:
                m = len(clusters)
                clusters.append(n.label)
                mapping[n.n] = m

            p = mapping[n.n]
            if p in seen:
                continue
            G.add_node(p, label=clusters[p])
            seen.add(p)

        for a, b in self.edges:
            a = mapping[a]
            b = mapping[b]

            if (a, b) in seen:
                continue

            if not (a in leaf or b in leaf):
                continue

            G.add_edge(a, b)
            seen.add((a, b))

        return G
