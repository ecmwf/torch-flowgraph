# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Regression tests for in-place-mutation cycles in the traced op graph.

Background
----------
In-place tensor mutation makes the tracer alias several *versions* of one
tensor onto a single node.  The canonical offender is ``aifs``'s
``aifs/model/gnn.py``::

    x_out = einops.rearrange(x_out, "(b n) f -> b n f", ...).to(dtype=x.dtype).clone()
    x_out[..., self._internal_output_idx] += x[:, -1, :, self._internal_input_idx]
    return x_out

combined with the ``normalize -> forward -> denormalize`` rollout in
``predict_step`` (each an in-place clone+setitem buffer).  When such a buffer
is reused/returned, the pre-write reads and the post-write output of the *same*
tensor object attach to the same node, and the value-flow graph picks up a
back-edge (an op pointing at one of its own ancestors).

The underlying computation is always a DAG — the cycles are purely a
representation artefact.  :meth:`GraphBuilder._break_cycles` strips the minimal
set of back-edges so downstream consumers that need a DAG
(:func:`networkx.topological_generations`, clan clustering) work.

The model ``joint-worm`` (checkpoint ``a5275e04-…``) exhibited exactly two
such cycles and crashed layout computation with
``networkx.NetworkXUnfeasible: Graph contains a cycle``.  The two edge sets
below reproduce those two cycles' topologies (node ids follow the real trace's
raw builder ids).
"""

import networkx as nx

from torch_flowgraph.model.pytorch import GraphBuilder
from torch_flowgraph.nodes import FunctionNode


def _builder(edges):
    """Return a :class:`GraphBuilder` populated with *edges* and dummy nodes."""
    b = GraphBuilder()
    ids = {n for e in edges for n in e}
    b.nodes = {i: FunctionNode(i, f"op{i}") for i in ids}
    b.edges = [tuple(e) for e in edges]
    return b


def _to_nx(builder):
    g = nx.DiGraph()
    g.add_nodes_from(builder.nodes)
    g.add_edges_from(builder.edges)
    return g


# ---------------------------------------------------------------------------
# The two real joint-worm cycles.
# ---------------------------------------------------------------------------

# Small SCC (12 nodes): the normalize/denormalize + gnn in-place buffer ring
#   1:y(output) -> 7:reshape -> 6:to -> 5:clone -> 4:setitem
#                -> 3:clone -> 2:setitem -> 1:y
# plus the feeder tail (149:div/150:sub/151:getitem, 146:add_/147:getitem).
SMALL_SCC_EDGES = [
    (1, 7),
    (7, 6),
    (6, 5),
    (5, 4),
    (4, 3),
    (3, 2),
    (2, 1),  # the ring
    (151, 150),
    (150, 149),
    (149, 2),  # normalize feeders
    (147, 146),
    (146, 4),  # denormalize feeder
    (3, 151),
    (5, 147),  # getitem views into buffers
]

# Big SCC (residual stream): a ring of transformer-block residual adds.
#   ... 33:add -> 30:add -> 25:add -> 22:add -> 18:linear -> 134:add -> ... -> 33:add
RESIDUAL_RING_EDGES = [
    (62, 57),
    (57, 54),
    (54, 49),
    (49, 46),
    (46, 41),
    (41, 38),
    (38, 33),
    (33, 30),
    (30, 25),
    (25, 22),
    (22, 18),
    (18, 134),
    (134, 129),
    (129, 126),
    (126, 121),
    (121, 118),
    (118, 113),
    (113, 110),
    (110, 105),
    (105, 65),
    (65, 62),  # closes the ring
]


def test_small_scc_becomes_dag():
    b = _builder(SMALL_SCC_EDGES)
    assert not nx.is_directed_acyclic_graph(_to_nx(b))  # cyclic before

    removed = b._break_cycles()

    assert removed, "expected at least one back-edge to be removed"
    assert nx.is_directed_acyclic_graph(_to_nx(b)), "graph still cyclic after fix"


def test_residual_ring_becomes_dag():
    b = _builder(RESIDUAL_RING_EDGES)
    assert not nx.is_directed_acyclic_graph(_to_nx(b))

    removed = b._break_cycles()

    assert removed
    assert nx.is_directed_acyclic_graph(_to_nx(b))


def test_both_cycles_reproduce_joint_worm_counts():
    """Both real cycles together need exactly 2 back-edges removed."""
    b = _builder(SMALL_SCC_EDGES + RESIDUAL_RING_EDGES)
    assert not nx.is_directed_acyclic_graph(_to_nx(b))

    removed = b._break_cycles()

    # One independent cycle per SCC -> one back-edge each.
    assert len(removed) == 2, removed
    assert nx.is_directed_acyclic_graph(_to_nx(b))


def test_break_cycles_is_noop_on_dag():
    """A graph that is already acyclic must be left completely unchanged."""
    dag_edges = [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3), (3, 4)]
    b = _builder(dag_edges)
    before = list(b.edges)

    removed = b._break_cycles()

    assert removed == []
    assert b.edges == before


def test_break_cycles_only_removes_back_edges():
    """Every removed edge must be a genuine back-edge (dst reaches src)."""
    b = _builder(SMALL_SCC_EDGES + RESIDUAL_RING_EDGES)
    original = set(b.edges)

    removed = b._break_cycles()
    kept = _to_nx(b)

    for a, c in removed:
        assert (a, c) in original
        # A back-edge (a -> c) closes a cycle: c must still reach a via kept edges.
        assert nx.has_path(kept, c, a), (a, c)


def test_self_loop_is_removed():
    """A degenerate self-loop (a -> a) is a back-edge and must be dropped."""
    b = _builder([(0, 1), (1, 2), (2, 2), (2, 3)])
    removed = b._break_cycles()
    assert (2, 2) in removed
    assert nx.is_directed_acyclic_graph(_to_nx(b))


def test_break_cycles_deterministic():
    """Repeated runs on identical input remove the same edge set."""
    r1 = _builder(SMALL_SCC_EDGES + RESIDUAL_RING_EDGES)._break_cycles()
    r2 = _builder(SMALL_SCC_EDGES + RESIDUAL_RING_EDGES)._break_cycles()
    assert r1 == r2


# ---------------------------------------------------------------------------
# Integration: a repaired graph must clan-cluster without NetworkXUnfeasible.
# ---------------------------------------------------------------------------


def test_repaired_graph_clan_clusters():
    """The layout step that crashed on joint-worm must succeed after the fix.

    Before the fix, ``ClanClustering`` -> ``levels`` ->
    ``nx.topological_generations`` raised ``NetworkXUnfeasible`` on the cyclic
    graph.  After ``_break_cycles`` the graph is a DAG and clustering runs.
    """
    from torch_flowgraph.clustering import ClanClustering
    from torch_flowgraph.graph import GraphBase

    b = _builder(SMALL_SCC_EDGES + RESIDUAL_RING_EDGES)
    b._break_cycles()

    flow = {
        "nodes": [n.as_dict() for n in b.nodes.values()],
        "edges": list(b.edges),
        "clusters": {},
    }
    graph = GraphBase.from_dict(flow)
    # Must not raise networkx.NetworkXUnfeasible.
    graph.recluster(ClanClustering())
