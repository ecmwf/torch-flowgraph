# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Structural validation for computation graphs.

:func:`validate_graph` checks a :class:`~torch_flowgraph.graph.GraphBase` for
inconsistencies and returns a boolean success flag.  It is intended both as a
command-line tool (``python -m torch_flowgraph validate graph.json``) and as a
building-block to be called right after a graph is constructed.

Checks performed
----------------
* **edges** — every edge endpoint refers to an existing node, and no edge is
  a self-loop.
* **cluster membership** — every node listed by a cluster exists, no node is
  claimed by more than one cluster, and every node is attached to some cluster.
* **acyclic** — the graph is a DAG (a computation graph must not contain
  directed cycles).
"""

from __future__ import annotations


def _iter_clusters(root):
    """Yield every cluster in *root*'s subtree (root first)."""
    stack = [root]
    while stack:
        c = stack.pop()
        yield c
        stack.extend(c.children)


def _check_edges(graph):
    ids = set(graph.nodes)
    problems = []
    for a, b in graph.edges:
        if a not in ids:
            problems.append(f"edge ({a} -> {b}) references unknown source node {a}")
        if b not in ids:
            problems.append(f"edge ({a} -> {b}) references unknown target node {b}")
        if a == b:
            problems.append(f"self-loop edge on node {a} ({graph.nodes[a].label})")
    return problems


def _check_cluster_membership(graph):
    ids = set(graph.nodes)
    owners: dict[int, list] = {}
    problems = []

    for c in _iter_clusters(graph.clusters):
        for n in c.nodes:
            if n not in ids:
                problems.append(f"cluster {c.name!r} lists unknown node {n}")
            owners.setdefault(n, []).append(c)

    for n, clusters in owners.items():
        if len(clusters) > 1:
            names = ", ".join(repr(c.name) for c in clusters)
            problems.append(f"node {n} ({graph.nodes[n].label}) belongs to {len(clusters)} clusters: {names}")

    for n in ids:
        if n not in owners:
            problems.append(f"node {n} ({graph.nodes[n].label}) is not attached to any cluster")

    return problems


def _check_acyclic(graph):
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(graph.nodes)
    G.add_edges_from(graph.edges)
    if nx.is_directed_acyclic_graph(G):
        return []

    problems = []
    for cycle in nx.simple_cycles(G):
        labels = " -> ".join(f"{n} ({graph.nodes[n].label})" for n in cycle)
        problems.append(f"directed cycle: {labels}")
    return problems


# Ordered mapping of check-name → callable(graph) -> list[str] of problems.
CHECKS = {
    "edges": _check_edges,
    "cluster membership": _check_cluster_membership,
    "acyclic": _check_acyclic,
}

_MAX_SHOWN = 20


def _report(graph, problems_by_check):
    if not problems_by_check:
        print(f"✓ graph is consistent ({len(graph)} nodes, {len(graph.edges)} edges)")
        return

    total = sum(len(v) for v in problems_by_check.values())
    print(f"✗ {total} inconsistency(ies) found:")
    for name, problems in problems_by_check.items():
        print(f"\n  [{name}] {len(problems)}")
        for p in problems[:_MAX_SHOWN]:
            print(f"    - {p}")
        if len(problems) > _MAX_SHOWN:
            print(f"    … and {len(problems) - _MAX_SHOWN} more")


def validate_graph(graph, *, verbose: bool = True) -> bool:
    """Validate *graph* and return ``True`` when it is consistent.

    Parameters
    ----------
    graph:
        A :class:`~torch_flowgraph.graph.GraphBase` (or subclass) to check.
    verbose:
        When ``True`` (default), print a human-readable report of any
        inconsistencies found.

    Returns
    -------
    bool
        ``True`` if no inconsistencies were found, ``False`` otherwise.
    """
    problems_by_check = {}
    for name, check in CHECKS.items():
        problems = check(graph)
        if problems:
            problems_by_check[name] = problems

    if verbose:
        _report(graph, problems_by_check)

    return not problems_by_check
