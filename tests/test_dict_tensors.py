# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Tests for dict-of-tensors input and output support.

Verifies that :meth:`ModelGraph.from_model` correctly handles models whose
``forward()`` accepts a ``dict[str, Tensor]`` and returns either a
``dict[str, Tensor]`` or a single ``Tensor``.
"""

import pytest

torch = pytest.importorskip("torch")

from torch_flowgraph.model.graph import ModelGraph  # noqa: E402
from torch_flowgraph.nodes import ShapeNode  # noqa: E402


# ---------------------------------------------------------------------------
# Dummy models
# ---------------------------------------------------------------------------


class DictToDictModel(torch.nn.Module):
    """Takes a dict of tensors and returns a dict of tensors.

    Each branch has its own linear layer; the output dict has the same keys.
    """

    def __init__(self, dim):
        super().__init__()
        self.branch_a = torch.nn.Linear(dim, dim)
        self.branch_b = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return {
            "alpha": self.branch_a(x["alpha"]),
            "beta": self.branch_b(x["beta"]),
        }


class DictToTensorModel(torch.nn.Module):
    """Takes a dict of tensors and returns a single tensor (sum of branches)."""

    def __init__(self, dim):
        super().__init__()
        self.branch_a = torch.nn.Linear(dim, dim)
        self.branch_b = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.branch_a(x["alpha"]) + self.branch_b(x["beta"])


class DictWithInternalDictModel(torch.nn.Module):
    """A sub-module returns a dict, but the outer module collapses to a tensor."""

    def __init__(self, dim):
        super().__init__()
        self.inner = DictToDictModel(dim)
        self.head = torch.nn.Linear(dim, dim)

    def forward(self, x):
        intermediates = self.inner(x)
        return self.head(intermediates["alpha"] + intermediates["beta"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SHAPE_DICT = {"alpha": (1, 8), "beta": (1, 8)}


class TestDictToDict:
    """Model: dict of tensors -> dict of tensors."""

    @pytest.fixture()
    def graph(self):
        model = DictToDictModel(dim=8)
        return ModelGraph.from_model(model, shape=SHAPE_DICT, add_extra_nodes=True)

    def test_graph_is_built(self, graph):
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

    def test_input_nodes_named_correctly(self, graph):
        """Input ShapeNodes should be x_alpha and x_beta."""
        shape_labels = {
            node.label for node in graph.nodes.values() if isinstance(node, ShapeNode)
        }
        assert "x_alpha" in shape_labels, f"Missing x_alpha in {shape_labels}"
        assert "x_beta" in shape_labels, f"Missing x_beta in {shape_labels}"

    def test_output_nodes_named_correctly(self, graph):
        """Output ShapeNodes should be y_alpha and y_beta."""
        shape_labels = {
            node.label for node in graph.nodes.values() if isinstance(node, ShapeNode)
        }
        assert "y_alpha" in shape_labels, f"Missing y_alpha in {shape_labels}"
        assert "y_beta" in shape_labels, f"Missing y_beta in {shape_labels}"

    def test_sources_and_sinks(self, graph):
        """Only ShapeNodes should be sources and sinks."""
        sources, sinks = graph.sources_and_sinks()
        bad_sources = [s for s in sources if not isinstance(s, ShapeNode)]
        bad_sinks = [s for s in sinks if not isinstance(s, ShapeNode)]
        assert not bad_sources, (
            f"Non-Shape sources: {[(n.n, type(n).__name__, n.label) for n in bad_sources]}"
        )
        assert not bad_sinks, (
            f"Non-Shape sinks: {[(n.n, type(n).__name__, n.label) for n in bad_sinks]}"
        )


class TestDictToTensor:
    """Model: dict of tensors -> single tensor."""

    @pytest.fixture()
    def graph(self):
        model = DictToTensorModel(dim=8)
        return ModelGraph.from_model(model, shape=SHAPE_DICT, add_extra_nodes=True)

    def test_graph_is_built(self, graph):
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

    def test_input_nodes_named_correctly(self, graph):
        shape_labels = {
            node.label for node in graph.nodes.values() if isinstance(node, ShapeNode)
        }
        assert "x_alpha" in shape_labels, f"Missing x_alpha in {shape_labels}"
        assert "x_beta" in shape_labels, f"Missing x_beta in {shape_labels}"

    def test_single_output_node(self, graph):
        """Output should be a single 'y' ShapeNode."""
        shape_labels = {
            node.label for node in graph.nodes.values() if isinstance(node, ShapeNode)
        }
        assert "y" in shape_labels, f"Missing y in {shape_labels}"
        # Should NOT have y_alpha / y_beta
        assert "y_alpha" not in shape_labels
        assert "y_beta" not in shape_labels

    def test_sources_and_sinks(self, graph):
        sources, sinks = graph.sources_and_sinks()
        bad_sources = [s for s in sources if not isinstance(s, ShapeNode)]
        bad_sinks = [s for s in sinks if not isinstance(s, ShapeNode)]
        assert not bad_sources, (
            f"Non-Shape sources: {[(n.n, type(n).__name__, n.label) for n in bad_sources]}"
        )
        assert not bad_sinks, (
            f"Non-Shape sinks: {[(n.n, type(n).__name__, n.label) for n in bad_sinks]}"
        )


class TestDictWithInternalDict:
    """Model: dict input -> internal dict intermediate -> single tensor output."""

    @pytest.fixture()
    def graph(self):
        model = DictWithInternalDictModel(dim=8)
        return ModelGraph.from_model(model, shape=SHAPE_DICT, add_extra_nodes=True)

    def test_graph_is_built(self, graph):
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

    def test_input_nodes_named_correctly(self, graph):
        shape_labels = {
            node.label for node in graph.nodes.values() if isinstance(node, ShapeNode)
        }
        assert "x_alpha" in shape_labels
        assert "x_beta" in shape_labels

    def test_single_output_node(self, graph):
        shape_labels = {
            node.label for node in graph.nodes.values() if isinstance(node, ShapeNode)
        }
        assert "y" in shape_labels

    def test_sources_and_sinks(self, graph):
        sources, sinks = graph.sources_and_sinks()
        bad_sources = [s for s in sources if not isinstance(s, ShapeNode)]
        bad_sinks = [s for s in sinks if not isinstance(s, ShapeNode)]
        assert not bad_sources, (
            f"Non-Shape sources: {[(n.n, type(n).__name__, n.label) for n in bad_sources]}"
        )
        assert not bad_sinks, (
            f"Non-Shape sinks: {[(n.n, type(n).__name__, n.label) for n in bad_sinks]}"
        )
