# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Graphviz DOT file generation for graphs and parse trees.

This module provides two rendering functions:

* :func:`plot_graph` — writes a ``digraph`` DOT file from a NetworkX
  :class:`~networkx.DiGraph`, optionally grouping nodes into Graphviz
  ``subgraph cluster_*`` blocks.
* :func:`plot_tree` — writes a ``digraph`` DOT file visualising the
  hierarchical structure of a :class:`~torch_flowgraph.parsing.ParseTree`.
"""

import json

from torch_flowgraph.attributes import EDGE_ATTRIBUTES
from torch_flowgraph.attributes import NODE_ATTRIBUTES
from torch_flowgraph.clusters import Cluster


def attribute_str(attributes, collection):
    """Serialise a dictionary of DOT attributes to a bracketed attribute string.

    Only entries whose key appears in *collection* are included.  If
    *collection* is ``None`` all entries are included.

    Parameters
    ----------
    attributes:
        Dictionary mapping attribute name to value.
    collection:
        Allowlist of attribute names, or ``None`` to allow all.

    Returns
    -------
    str or None
        A string of the form ``[key="value", ...]``, or ``None`` when there
        are no matching attributes.
    """
    a = []
    for k, v in attributes.items():
        if collection is None or k in collection:
            a.append(f"{k}={json.dumps(v)}")

    if a:
        return f"[{', '.join(a)}]"
    return None


_DEFAULT_PASSIVE_TYPES = ("TensorNode", "ParameterNode")


def build_plan(
    G,
    clusters=None,
    graph_attributes=None,
    show_clusters=True,
    highlight_fn=None,
    cluster_bg=True,
    passive_types=_DEFAULT_PASSIVE_TYPES,
):
    """Build a JSON-serialisable plan describing the rendered graph.

    The plan carries **all** the information needed both to:

    * generate the Graphviz DOT file (consumed by :func:`plot_graph`), and
    * drive the React data-flow animation in the Models MFE (consumed via
      ``/flow/plan``).

    Parameters
    ----------
    G, clusters, graph_attributes, show_clusters, highlight_fn, cluster_bg:
        Same semantics as :func:`plot_graph`.
    passive_types:
        Sequence of node *type* names (matched against the ``_kind``
        attribute injected by :meth:`~torch_flowgraph.nodes.Node.dot_node_kwargs`)
        that should be marked ``"passive": true`` in the plan.
        These nodes are rendered normally in the SVG but are ignored by
        the data-flow simulation.  Defaults to ``("TensorNode",
        "ParameterNode")``.  Active nodes omit the field entirely.

    Returns
    -------
    dict
        Plan dictionary; see :func:`_render_plan` for its structure.
    """
    if clusters is None:
        clusters = []
    if not isinstance(clusters, list):
        clusters = [clusters]
    clusters = [Cluster.from_dict(c, None) if isinstance(c, dict) else c for c in clusters]

    graph_attributes = graph_attributes.copy() if graph_attributes else {}
    graph_attributes.setdefault("rankdir", "TB")

    passive = set(passive_types)

    # ── Nodes ─────────────────────────────────────────────────────────
    nodes = []
    nodes_by_sn = {}
    start_sns = []
    end_sns = []
    for n, attrs in G.nodes(data=True):
        sn = int(n)
        label = str(attrs.get("label", n))
        kind = attrs.get("_kind", "")
        rank = attrs.get("_rank")
        filtered = {k: v for k, v in attrs.items() if k in NODE_ATTRIBUTES}
        entry = {
            "sn": sn,
            "label": label,
            "attrs": filtered,
        }
        if kind in passive:
            entry["passive"] = True
        nodes.append(entry)
        nodes_by_sn[sn] = entry
        if rank == "min":
            start_sns.append(sn)
        elif rank == "max":
            end_sns.append(sn)

    # Fallback: if no _rank attributes were found, detect sources/sinks
    # from graph topology (nodes with no incoming / no outgoing edges,
    # excluding passive nodes).  This handles graph JSONs serialised before
    # the rank field was persisted.
    if not start_sns or not end_sns:
        has_incoming = {int(b) for _, b in G.edges}
        has_outgoing = {int(a) for a, _ in G.edges}
        passive_sns = {e["sn"] for e in nodes if e.get("passive")}
        if not start_sns:
            start_sns = [e["sn"] for e in nodes if e["sn"] not in has_incoming and e["sn"] not in passive_sns]
        if not end_sns:
            end_sns = [e["sn"] for e in nodes if e["sn"] not in has_outgoing and e["sn"] not in passive_sns]

    # ── Edges ─────────────────────────────────────────────────────────
    edges = []
    for a, b in G.edges:
        e_attrs = G.get_edge_data(a, b)
        filtered = {k: v for k, v in e_attrs.items() if k in EDGE_ATTRIBUTES}
        edges.append({"src": int(a), "dst": int(b), "attrs": filtered})

    # ── Clusters (recursive) ──────────────────────────────────────────
    nodes_in_G = set(int(n) for n in G.nodes)

    def _cluster_to_plan(c):
        invisible = (not show_clusters) or (c.visible is False) or (not c.name)
        highlight = (not invisible) and highlight_fn is not None and highlight_fn(c)
        return {
            "name": c.name,
            "invisible": invisible,
            "highlight": highlight,
            "dot_attributes": dict(c.dot_attributes),
            "node_sns": [int(n) for n in c.nodes if int(n) in nodes_in_G],
            "children": [_cluster_to_plan(child) for child in c.children],
        }

    cluster_plans = [_cluster_to_plan(c) for c in clusters]

    # ── Rank groups ───────────────────────────────────────────────────
    rank_groups = {}
    for n, data in G.nodes(data=True):
        r = data.get("_rank")
        if r is not None:
            rank_groups.setdefault(r, []).append(int(n))
    rank_plan = [{"rank": r, "members": members} for r, members in rank_groups.items()]

    return {
        "graph_attributes": graph_attributes,
        "show_clusters": show_clusters,
        "cluster_bg": cluster_bg,
        "clusters": cluster_plans,
        "nodes": nodes,
        "edges": edges,
        "rank_groups": rank_plan,
        "start_sns": start_sns,
        "end_sns": end_sns,
    }


def _render_plan(path, plan):
    """Write the DOT file *path* from a plan produced by :func:`build_plan`.

    The plan is the **only** input — no NetworkX graph or :class:`Cluster`
    objects are consulted here.  This guarantees that the DOT file (and
    hence the SVG) and the JSON plan exposed to the front-end describe
    exactly the same graph.
    """
    nodes_by_sn = {n["sn"]: n for n in plan["nodes"]}
    seen = set()
    cluster_bg = plan["cluster_bg"]

    with open(path, "w") as f:

        def _emit_node(sn):
            if sn in seen or sn not in nodes_by_sn:
                return
            seen.add(sn)
            entry = nodes_by_sn[sn]
            a = attribute_str(entry["attrs"], NODE_ATTRIBUTES)
            if a:
                print(f"{sn} {a}", file=f)
            else:
                print(f"{sn}", file=f)

        def _emit_cluster(c, *path_idx):
            invisible = c["invisible"]
            highlight = c["highlight"]
            name = "_".join(map(str, path_idx))

            print(f"subgraph cluster_{name} {{", file=f)
            if invisible:
                print('style="invis";', file=f)
                print('label="";', file=f)
            else:
                if cluster_bg:
                    print('style="filled,dashed";', file=f)
                else:
                    print('style="dashed";', file=f)
                print('fontname="Helvetica-Bold,Arial-Bold,sans-serif";', file=f)
                print('fontcolor="#343a40";', file=f)
                print("fontsize=11;", file=f)
                print('labeljust="l";', file=f)
                print("margin=10;", file=f)
                for k, v in c["dot_attributes"].items():
                    print(f"{k}={json.dumps(v)};", file=f)
                if highlight:
                    print('color="#ffc9c9";', file=f)
                    if cluster_bg:
                        print('fillcolor="#fff5f5";', file=f)
                    print("penwidth=2;", file=f)
                else:
                    print('color="#dee2e6";', file=f)
                    if cluster_bg:
                        fillcolor = "#f8f9fa" if len(path_idx) % 2 == 1 else "#fafbfc"
                        print(f'fillcolor="{fillcolor}";', file=f)

            for i, child in enumerate(c["children"]):
                _emit_cluster(child, *path_idx, i)
            for sn in c["node_sns"]:
                _emit_node(sn)
            print("}", file=f)

        print("digraph {", file=f)

        # Global graph attributes
        print('fontname="Helvetica,Arial,sans-serif";', file=f)
        print("nodesep=0.5;", file=f)
        print("ranksep=0.8;", file=f)
        print('bgcolor="transparent";', file=f)

        for k, v in plan["graph_attributes"].items():
            if isinstance(v, dict):
                a = attribute_str(v, None)
                print(f"{k} {a}", file=f)
            else:
                print(f"{k}={json.dumps(v)};", file=f)

        # Default node / edge styles
        print(
            'node [fontname="Helvetica,Arial,sans-serif", shape="ellipse",'
            ' style="filled", fillcolor="#e7f5ff", color="#a5d8ff",'
            ' fontcolor="#212529", penwidth=1.5, margin="0.15,0.08"];',
            file=f,
        )
        print(
            'edge [fontname="Helvetica,Arial,sans-serif", color="#adb5bd",'
            " arrowsize=0.8, penwidth=1.5, fontsize=9];",
            file=f,
        )

        for i, c in enumerate(plan["clusters"]):
            _emit_cluster(c, i)

        # Loose nodes (those not placed in any cluster)
        for entry in plan["nodes"]:
            _emit_node(entry["sn"])

        # Edges
        for e in plan["edges"]:
            a, b = e["src"], e["dst"]
            _emit_node(a)
            _emit_node(b)
            attr = [f"{k}={json.dumps(v)}" for k, v in e["attrs"].items()]
            if attr:
                print(f"{a} -> {b}  [{', '.join(attr)}]", file=f)
            else:
                print(f"{a} -> {b};", file=f)

        # Rank constraints
        for rg in plan["rank_groups"]:
            members = "; ".join(str(n) for n in rg["members"])
            print(f"{{ rank={rg['rank']}; {members}; }}", file=f)

        print("}", file=f)


def plot_graph(
    path,
    G,
    clusters=None,
    graph_attributes=None,
    show_clusters=True,
    highlight_fn=None,
    cluster_bg=True,
    passive_types=_DEFAULT_PASSIVE_TYPES,
):
    """Write a NetworkX directed graph to a Graphviz DOT file.

    This is a thin wrapper that delegates to :func:`build_plan` for the
    JSON-serialisable plan and then to :func:`_render_plan` for the DOT
    serialisation.  The DOT file is generated from the plan alone — no
    NetworkX or :class:`~torch_flowgraph.clusters.Cluster` instances are
    consulted by the renderer.  See :func:`build_plan` for parameter
    semantics.
    """
    plan = build_plan(
        G,
        clusters=clusters,
        graph_attributes=graph_attributes,
        show_clusters=show_clusters,
        highlight_fn=highlight_fn,
        cluster_bg=cluster_bg,
        passive_types=passive_types,
    )
    _render_plan(path, plan)


def plot_tree(path, tree):
    """Write a :class:`~torch_flowgraph.parsing.ParseTree` to a Graphviz DOT file.

    Each node in the parse tree becomes a DOT node labelled with the clan
    kind, the set of member nodes, and the number of children.  Edges reflect
    the parent-child relationships in the tree.  The graph uses a left-to-right
    layout (``rankdir=LR``) to make the hierarchy easy to read.

    Parameters
    ----------
    path:
        Destination file path for the DOT output.
    tree:
        Root :class:`~torch_flowgraph.parsing.ParseTree` to render.
    """
    seen = set()

    def _plot_cluster(c, path, depth=0):
        """Recursively emit parse-tree nodes and edges."""
        name = "_".join(map(str, path))

        assert name not in seen
        seen.add(name)
        kind = c.clan.kind
        nodes = sorted(c.clan.nodes)
        print(f'{name} [label="{kind} {nodes} [{len(c.children)}]"];', file=f)

        for i, n in enumerate(set(c.children)):
            m = _plot_cluster(n, path + [i], depth + 1)
            print(f"{name} -> {m}", file=f)

        return name

    with open(path, "w") as f:
        print("digraph {\nrankdir=LR", file=f)

        _plot_cluster(tree, ["n"])

        print("}", file=f)
