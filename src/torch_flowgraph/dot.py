# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

"""High-level DOT export helper for :class:`ModelGraph`.

This module provides :func:`save_dot`, a convenience wrapper around
:func:`~torch_flowgraph.plotting.plot_graph` that accepts a
:class:`~torch_flowgraph.model.graph.ModelGraph` and writes a Graphviz DOT
file suitable for rendering with ``dot -Tpdf``.
"""

from torch_flowgraph.clusters import Cluster
from torch_flowgraph.plotting import plot_graph


def save_dot(
    graph,
    path,
    graph_attributes=None,
    clusters=None,
    depth=None,
    links=False,
    show_clusters=True,
    highlight_fn=None,
    cluster_bg=True,
):
    """Write a :class:`ModelGraph` to a Graphviz DOT file.

    Optionally replaces the graph's cluster hierarchy with *clusters* and/or
    collapses the graph to *depth* levels before rendering.

    Parameters
    ----------
    graph:
        :class:`~torch_flowgraph.model.graph.ModelGraph` instance to export.
    path:
        Destination file path for the ``.dot`` output.
    graph_attributes:
        Optional dictionary of top-level Graphviz graph attributes (e.g.
        ``{"rankdir": "LR"}``).
    clusters:
        Controls subgraph box rendering:

        * ``None`` (default) — use the graph's own cluster hierarchy.
        * A :class:`~torch_flowgraph.clusters.Cluster` instance — re-cluster
          the graph with this hierarchy before rendering.
        * ``[]`` (empty list) — suppress all subgraph boxes.
    depth:
        Maximum cluster depth to render.  When given, the graph is collapsed
        to *depth* levels via
        :meth:`~torch_flowgraph.model.graph.ModelGraph.collapsed`.
    links:
        When ``True``, function nodes include ``URL`` attributes linking to
        the PyTorch documentation.  Defaults to ``False``.
    show_clusters:
        When ``False``, all cluster subgraph blocks are rendered as invisible
        so they guide the layout engine without drawing visible boxes.
        Forwarded to :func:`~torch_flowgraph.plotting.plot_graph`.
        Defaults to ``True``.
    cluster_bg:
        When ``False``, cluster boxes are drawn without a background fill.
        Forwarded to :func:`~torch_flowgraph.plotting.plot_graph`.
        Defaults to ``True``.
    """
    # Re-cluster only when a real Cluster object is provided.
    if isinstance(clusters, Cluster) and clusters is not graph.clusters:
        graph = graph.use_clusters(clusters)

    graph = graph.collapsed(depth) if depth is not None else graph

    G = graph.to_op_networkx(links=links)
    # None → show own clusters (skip root wrapper, use children directly);
    # [] → no boxes; Cluster already applied above.
    display_clusters = graph.clusters.children if clusters is None else clusters
    plot_graph(
        path,
        G,
        display_clusters,
        graph_attributes=graph_attributes,
        cluster_bg=cluster_bg,
        show_clusters=show_clusters,
        highlight_fn=highlight_fn,
    )
