# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Regression test for the ``GraphTransformerMapperBlock`` orphan-node bug.

The original bug appeared when a sub-module returned a tuple containing a
plain (non-``Tracker``) ``torch.Tensor`` in addition to tracked tensors, and
the caller discarded that plain tensor.  ``leave_node`` would create a
*bridge* :class:`~torch_flowgraph.nodes.FunctionNode` so that the inputs
remained connected to *something*, but because the wrapped output was never
consumed downstream, the bridge ended up with incoming edges and **no
outgoing edges** — an orphan node in the middle of the graph.

This test reproduces the same shape with a tiny synthetic model and asserts,
via :meth:`GraphBase.sources_and_sinks`, that only ``ShapeNode`` instances
appear as sources or sinks.
"""

import pytest

torch = pytest.importorskip("torch")

from torch_flowgraph.model.graph import ModelGraph  # noqa: E402
from torch_flowgraph.nodes import ParameterNode  # noqa: E402
from torch_flowgraph.nodes import ShapeNode  # noqa: E402
from torch_flowgraph.nodes import TensorNode  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic reproduction of the GraphTransformerMapperBlock pattern.
# ---------------------------------------------------------------------------


class TrainableBuffer(torch.nn.Module):
    """Module whose forward returns a function of a registered buffer.

    The output never has a ``Tracker`` provenance, so it is a plain
    ``torch.Tensor`` from the tracer's point of view (mirrors
    ``anemoi.models.layers.utils.TrainableTensor``).
    """

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("buf", torch.randn(dim))

    def forward(self):
        return self.buf * 1.0


class SubBlock(torch.nn.Module):
    """Block returning ``(tracked_output, plain_extra)``.

    Mirrors ``GraphTransformerMapperBlock.forward`` which returns
    ``((nodes_new_src, nodes_new_dst), edge_attr)`` where ``edge_attr`` is the
    same plain tensor that was passed in.
    """

    def __init__(self, dim):
        super().__init__()
        self.lin = torch.nn.Linear(dim, dim)

    def forward(self, x, extra):
        # ``x`` is a Tracker; ``extra`` is a plain Tensor with no Tracker
        # provenance.  We return both, but the caller will discard ``extra``.
        return self.lin(x), extra


class OuterModule(torch.nn.Module):
    """Outer module that discards the plain return value of ``SubBlock``."""

    def __init__(self, dim):
        super().__init__()
        self.trainable = TrainableBuffer(dim)
        self.sub = SubBlock(dim)
        self.head = torch.nn.Linear(dim, dim)

    def forward(self, x):
        extra = self.trainable()  # plain Tensor (no Tracker provenance)
        x, _discarded = self.sub(x, extra)  # plain ``extra`` is dropped
        return self.head(x)


# ---------------------------------------------------------------------------
# Synthetic reproduction of the ``get_shape_shards`` orphan bug.
#
# ``anemoi.models.distributed.shapes.get_shape_shards`` calls
# ``torch.tensor_split(tracked_tensor, 1, dim=...)`` and inspects only ``.shape``
# on each chunk — never feeding the tracked outputs into another torch op.
# The naive tracer records a ``tensor_split`` :class:`FunctionNode` with input
# edges but no output edges.
# ---------------------------------------------------------------------------


class ShapeProbeModule(torch.nn.Module):
    """Module that mimics ``get_shape_shards`` then continues processing."""

    def __init__(self, dim):
        super().__init__()
        self.head = torch.nn.Linear(dim, dim)

    def forward(self, x):
        # Side-channel use: split, read shapes, throw the chunks away.
        _shapes = [list(c.shape) for c in torch.tensor_split(x, 1, dim=0)]
        # Real data flow continues from the original ``x``.
        return self.head(x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sources_and_sinks_basic():
    """Sanity check: well-formed graph has only ShapeNode sources/sinks."""
    model = OuterModule(dim=8)
    graph = ModelGraph.from_model(model, shape=(1, 8), add_extra_nodes=True)

    sources, sinks = graph.sources_and_sinks()

    bad_sources = [s for s in sources if not isinstance(s, (ShapeNode, ParameterNode, TensorNode))]
    bad_sinks = [s for s in sinks if not isinstance(s, ShapeNode)]

    assert not bad_sources, (
        f"Non-Shape/Parameter/Tensor sources (no incoming edges): "
        f"{[(n.n, type(n).__name__, n.label) for n in bad_sources]}"
    )
    assert not bad_sinks, (
        f"Non-Shape sinks (no outgoing edges): " f"{[(n.n, type(n).__name__, n.label) for n in bad_sinks]}"
    )


def test_no_orphan_bridge_node():
    """The discarded plain tensor must not produce an orphan FunctionNode.

    Specifically, no node labelled ``"SubBlock"`` (the bridge name) should
    exist with incoming edges but no outgoing edges.
    """
    model = OuterModule(dim=8)
    graph = ModelGraph.from_model(model, shape=(1, 8), add_extra_nodes=True)

    has_outgoing = {a for a, _ in graph.edges}
    has_incoming = {b for _, b in graph.edges}

    orphans = [
        node
        for nid, node in graph.nodes.items()
        if nid in has_incoming and nid not in has_outgoing and not isinstance(node, ShapeNode)
    ]

    assert not orphans, (
        f"Found orphan node(s) with incoming but no outgoing edges: "
        f"{[(n.n, type(n).__name__, n.label) for n in orphans]}"
    )


def test_shape_probe_does_not_create_orphan():
    """``tensor_split`` whose outputs are only ``.shape``-inspected must not be recorded.

    Reproduces the ``get_shape_shards`` pattern from
    ``anemoi.models.distributed.shapes``: a torch op whose output trackers
    never reach another torch op should not appear as a node in the graph
    (or, equivalently, must not appear as a graph sink).
    """
    model = ShapeProbeModule(dim=8)
    graph = ModelGraph.from_model(model, shape=(4, 8), add_extra_nodes=True)

    sources, sinks = graph.sources_and_sinks()

    bad_sources = [s for s in sources if not isinstance(s, (ShapeNode, ParameterNode, TensorNode))]
    bad_sinks = [s for s in sinks if not isinstance(s, ShapeNode)]

    assert not bad_sources, (
        f"Non-Shape/Parameter/Tensor sources: " f"{[(n.n, type(n).__name__, n.label) for n in bad_sources]}"
    )
    assert not bad_sinks, (
        f"Non-Shape sinks (tensor_split orphan?): " f"{[(n.n, type(n).__name__, n.label) for n in bad_sinks]}"
    )
