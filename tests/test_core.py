# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Tests for the torch_flowgraph package.

These tests exercise the core graph data structures, serialisation round-trips,
clustering strategies, and graph transformations using the bundled
``tinynet.json`` fixture (no PyTorch installation required).
"""

import json
import os
import tempfile

import pytest

from torch_flowgraph import ClanClustering
from torch_flowgraph import GraphBase
from torch_flowgraph import LevelClustering
from torch_flowgraph import NameClustering
from torch_flowgraph import NoClustering
from torch_flowgraph import apply_clan_clustering
from torch_flowgraph import validate_graph
from torch_flowgraph.clusters import Cluster
from torch_flowgraph.nodes import CollapsedNode
from torch_flowgraph.nodes import FunctionNode
from torch_flowgraph.nodes import LabelNode
from torch_flowgraph.nodes import ModuleNode
from torch_flowgraph.nodes import Node
from torch_flowgraph.nodes import ParameterNode
from torch_flowgraph.nodes import ShapeNode
from torch_flowgraph.nodes import url

FIXTURE = os.path.join(os.path.dirname(__file__), os.pardir, "tinynet.json")


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def tinynet_dict():
    """Raw dictionary loaded from tinynet.json."""
    with open(FIXTURE) as f:
        return json.load(f)


@pytest.fixture
def graph(tinynet_dict):
    """A GraphBase built from the tinynet.json fixture."""
    return GraphBase.from_dict(tinynet_dict)


# ===================================================================
# Node types
# ===================================================================


class TestNode:
    def test_repr(self):
        n = Node(0, "hello")
        assert repr(n) == "hello"

    def test_as_dict(self):
        n = Node(42, "test")
        d = n.as_dict()
        assert d["n"] == 42
        assert d["label"] == "test"
        assert d["_kind"] == "Node"

    def test_dot_node_kwargs_default(self):
        n = Node(0, "x")
        assert n.dot_node_kwargs()["label"] == "x"

    def test_dot_node_kwargs_with_rank(self):
        n = Node(0, "x")
        n.rank = "min"
        kw = n.dot_node_kwargs()
        assert kw["_rank"] == "min"

    def test_dot_edge_kwargs_empty(self):
        a = Node(0, "a")
        b = Node(1, "b")
        assert a.dot_edge_kwargs(b) == {}


class TestShapeNode:
    def test_shape_stored_as_tuple(self):
        sn = ShapeNode(0, "x", [1, 3, 32, 32])
        assert sn.shape == (1, 3, 32, 32)

    def test_as_dict(self):
        sn = ShapeNode(0, "x", (1, 3))
        d = sn.as_dict()
        assert d["shape"] == (1, 3)
        assert d["_kind"] == "ShapeNode"

    def test_dot_edge_kwargs_contains_shape(self):
        sn = ShapeNode(0, "x", (1, 3, 32, 32))
        other = Node(1, "y")
        kw = sn.dot_edge_kwargs(other)
        assert "(1, 3, 32, 32)" in kw["label"]

    def test_dot_node_kwargs_colours(self):
        sn = ShapeNode(0, "x", (1,))
        kw = sn.dot_node_kwargs()
        assert "fillcolor" in kw
        assert "color" in kw


class TestFunctionNode:
    def test_basic(self):
        fn = FunctionNode(0, "relu")
        assert repr(fn) == "relu"

    def test_dot_label_without_params(self):
        fn = FunctionNode(0, "relu")
        assert fn.dot_label == "relu"

    def test_dot_label_with_params(self):
        fn = FunctionNode(0, "conv2d", params={"stride": (1, 1), "padding": (1, 1)})
        label = fn.dot_label
        assert "conv2d" in label
        assert "stride" in label

    def test_as_dict_includes_params(self):
        fn = FunctionNode(0, "linear", params={"bias": True})
        d = fn.as_dict()
        assert d["params"] == {"bias": True}
        assert d["_kind"] == "FunctionNode"

    def test_dot_node_kwargs_with_links(self):
        fn = FunctionNode(0, "relu")
        kw = fn.dot_node_kwargs(links=True)
        assert "URL" in kw

    def test_dot_node_kwargs_without_links(self):
        fn = FunctionNode(0, "relu")
        kw = fn.dot_node_kwargs(links=False)
        assert "URL" not in kw


class TestModuleNode:
    def test_dot_colours(self):
        mn = ModuleNode(0, "Linear")
        kw = mn.dot_node_kwargs()
        assert kw["fillcolor"] == "#f3f0ff"


class TestParameterNode:
    def test_dot_shape(self):
        pn = ParameterNode(0, "P", (16, 3, 3, 3))
        kw = pn.dot_node_kwargs()
        assert kw["shape"] == "invhouse"


class TestCollapsedNode:
    def test_dot_style(self):
        cn = CollapsedNode(0, "block")
        kw = cn.dot_node_kwargs()
        assert kw["shape"] == "rect"
        assert "rounded" in kw["style"]

    def test_explore_url(self):
        cn = CollapsedNode(0, "block")
        cn.explore_url = "explore:expand:0"
        kw = cn.dot_node_kwargs()
        assert kw["URL"] == "explore:expand:0"

    def test_outgoing_only_accepts_function_nodes(self):
        cn = CollapsedNode(0, "block")
        cn.outgoing(FunctionNode(1, "relu"))
        cn.outgoing(Node(2, "other"))  # silently ignored
        assert len(cn._outgoing) == 1

    def test_dot_edge_kwargs_fallback(self):
        cn = CollapsedNode(0, "block")
        other = Node(1, "y")
        kw = cn.dot_edge_kwargs(other)
        assert kw["label"] == "?"


class TestLabelNode:
    def test_creation(self):
        ln = LabelNode(0, "entry")
        assert repr(ln) == "entry"


class TestUrl:
    def test_known_op(self):
        u = url("relu")
        assert "torch.nn.functional.relu" in u

    def test_suppressed_op(self):
        assert url("__getitem__") is None

    def test_unknown_op_gets_search_url(self):
        u = url("some_unknown_op")
        assert "search.html" in u

    def test_single_char_returns_none(self):
        assert url("x") is None


# ===================================================================
# Cluster
# ===================================================================


class TestCluster:
    def test_basic_creation(self):
        c = Cluster(name="root", nodes=[0, 1, 2])
        assert c.name == "root"
        assert len(c) == 3

    def test_repr(self):
        c = Cluster(name="Linear")
        assert "Linear" in repr(c)

    def test_dot_attributes(self):
        c = Cluster(name="block", bgcolor="#fff")
        attr = c.dot_attributes
        assert attr["label"] == "block"
        assert attr["bgcolor"] == "#fff"

    def test_as_dict_round_trip(self):
        root = Cluster(name="root", nodes=[0, 1])
        child = Cluster(name="child", parent=root, nodes=[2, 3])
        root.children = [child]

        d = root.as_dict()
        restored = Cluster.from_dict(d, None)
        assert restored.name == "root"
        assert len(restored.children) == 1
        assert restored.children[0].name == "child"
        assert restored.children[0].nodes == [2, 3]

    def test_all_nodes(self):
        root = Cluster(name="root", nodes=[0, 1])
        child = Cluster(name="child", parent=root, nodes=[2, 3])
        root.children = [child]
        assert root.all_nodes == {0, 1, 2, 3}

    def test_show_cluster_default(self):
        c = Cluster(name="x")
        assert c.show_cluster(depth=5) is True

    def test_show_cluster_explicit_false(self):
        c = Cluster(name="x", visible=False)
        assert c.show_cluster(depth=0) is False


# ===================================================================
# GraphBase — construction and properties
# ===================================================================


class TestGraphBase:
    def test_len(self, graph):
        assert len(graph) == 26

    def test_edges(self, graph):
        assert len(graph.edges) == 25

    def test_depth(self, graph):
        assert graph.depth == 3

    def test_repr(self, graph):
        assert "26" in repr(graph)

    def test_nodes_have_cluster(self, graph):
        for node in graph.nodes.values():
            assert node.cluster is not None


# ===================================================================
# Serialisation round-trips
# ===================================================================


class TestSerialisation:
    def test_json_file_round_trip(self, graph):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            graph.save_json(path)
            restored = GraphBase.from_json_file(path)
            assert len(restored) == len(graph)
            assert len(restored.edges) == len(graph.edges)
            assert restored.depth == graph.depth
        finally:
            os.unlink(path)

    def test_dict_round_trip(self, graph):
        d = graph.as_dict()
        restored = GraphBase.from_dict(d)
        assert len(restored) == len(graph)
        assert len(restored.edges) == len(graph.edges)

    def test_from_json_file(self):
        graph = GraphBase.from_json_file(FIXTURE)
        assert len(graph) == 26

    def test_node_kinds_preserved(self, tinynet_dict):
        graph = GraphBase.from_dict(tinynet_dict)
        kinds = {type(n).__name__ for n in graph.nodes.values()}
        assert "FunctionNode" in kinds
        assert "ShapeNode" in kinds


# ===================================================================
# Graph transformations
# ===================================================================


class TestCollapsed:
    def test_collapsed_depth_1(self, graph):
        c = graph.collapsed(max_depth=1)
        assert len(c) < len(graph)

    def test_collapsed_depth_0(self, graph):
        c = graph.collapsed(max_depth=0)
        assert len(c) < len(graph)

    def test_collapsed_preserves_connectivity(self, graph):
        c = graph.collapsed(max_depth=1)
        assert len(c.edges) > 0

    def test_collapsed_at_full_depth_preserves_nodes(self, graph):
        c = graph.collapsed(max_depth=graph.depth)
        assert len(c) == len(graph)


class TestOnlyModules:
    def test_produces_fewer_nodes(self, graph):
        m = graph.only_modules()
        assert len(m) < len(graph)

    def test_edges_exist(self, graph):
        m = graph.only_modules()
        assert len(m.edges) > 0


class TestExplore:
    def test_default_explore(self, graph):
        e = graph.explore()
        assert len(e) > 0

    def test_expand_a_path(self, graph):
        e1 = graph.explore()
        e2 = graph.explore(expanded_paths={"0"})
        assert len(e2) > len(e1)


class TestUseClusters:
    def test_with_dict(self, graph):
        d = graph.clusters.as_dict()
        g2 = graph.use_clusters(d)
        assert len(g2) == len(graph)
        assert g2.depth == graph.depth


# ===================================================================
# NetworkX conversion
# ===================================================================


class TestNetworkX:
    def test_to_op_networkx(self, graph):
        G = graph.to_op_networkx()
        assert G.number_of_nodes() == 26
        assert G.number_of_edges() == 25

    def test_to_module_networkx(self, graph):
        G = graph.to_module_networkx()
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0


# ===================================================================
# Clustering strategies
# ===================================================================


class TestClanClustering:
    def test_apply_clan_clustering(self, graph):
        clan = apply_clan_clustering(graph)
        assert len(clan) == len(graph)
        assert clan.depth > 0

    def test_clan_clustering_class(self, graph):
        clan = graph.recluster(ClanClustering())
        assert len(clan) == len(graph)


class TestLevelClustering:
    def test_level_0(self, graph):
        # Level 0 is the outermost real cluster: the single "TinyNet" box,
        # with the input/output nodes (x, y) left outside it.
        g = graph.recluster(LevelClustering(level=0))
        assert g.depth == 1
        assert g.cluster_names == ["TinyNet"]

    def test_level_1(self, graph):
        # Level 1 is TinyNet's children: the Sequential and Linear boxes.
        g = graph.recluster(LevelClustering(level=1))
        assert g.depth == 1
        assert g.cluster_names == ["Linear", "Sequential"]


class TestNameClustering:
    def test_filter_by_name(self, graph):
        g = graph.recluster(NameClustering(names=["Sequential"]))
        assert g.depth >= 1


class TestNoClustering:
    def test_flat(self, graph):
        g = graph.recluster(NoClustering())
        assert g.depth == 0
        assert len(g) == len(graph)


class TestCollapseToLevel:
    def test_level_0_single_node(self, graph):
        # Level 0 collapses the outermost module (TinyNet) into one node:
        # x -> TinyNet -> y.
        g = graph.collapse_to_level(0)
        labels = sorted(n.label for n in g.nodes.values())
        assert labels == ["TinyNet", "x", "y"]
        assert g.cluster_names == []

    def test_level_1_opens_top_box(self, graph):
        # Level 1 keeps the TinyNet box and collapses its children to nodes.
        g = graph.collapse_to_level(1)
        assert g.cluster_names == ["TinyNet"]
        labels = {n.label for n in g.nodes.values()}
        assert {"Sequential", "Linear"} <= labels

    def test_deep_level_is_full_graph(self, graph):
        # Collapsing past the deepest level returns every node.
        g = graph.collapse_to_level(graph.depth + 1)
        assert len(g) == len(graph)


class TestValidation:
    def test_fixture_is_consistent(self, graph):
        assert validate_graph(graph, verbose=False) is True

    def test_missing_edge_endpoint(self, graph):
        graph.edges.append((0, 999999))
        assert validate_graph(graph, verbose=False) is False

    def test_self_loop(self, graph):
        graph.edges.append((0, 0))
        assert validate_graph(graph, verbose=False) is False

    def test_cycle(self, graph):
        # Introduce a 2-cycle between two existing nodes.
        a, b = list(graph.nodes)[:2]
        graph.edges.append((a, b))
        graph.edges.append((b, a))
        assert validate_graph(graph, verbose=False) is False
