# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""ModelGraph: traced PyTorch model as a computation graph.

:class:`ModelGraph` extends :class:`~torch_flowgraph.graph.GraphBase` with
construction from a live PyTorch model via
:meth:`ModelGraph.from_model`.  It also supports optional injection of
auxiliary tensor and parameter nodes during initialisation.

For graphs deserialised from JSON (i.e. not produced by PyTorch tracing),
use :class:`~torch_flowgraph.graph.GraphBase` directly.
"""

from ..clusters import Cluster
from ..graph import GraphBase
from ..nodes import Node


class ModelGraph(GraphBase):
    """A traced PyTorch model represented as nodes, edges, and a cluster tree.

    Extends :class:`~torch_flowgraph.graph.GraphBase` with:

    * Construction from a live PyTorch model (:meth:`from_model`).
    * Optional injection of auxiliary tensor/parameter nodes
      (``add_extra_nodes`` parameter).

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

    def __init__(self, nodes, edges, clusters, add_extra_nodes=False):
        """Initialise a ModelGraph and finalise the cluster/node relationship.

        Calls the parent :class:`~torch_flowgraph.graph.GraphBase` initialiser
        and then, when requested, injects auxiliary tensor and parameter nodes
        into the graph before finalising a second time.

        Parameters
        ----------
        nodes:
            Mapping of node-id to :class:`~torch_flowgraph.nodes.Node`.
        edges:
            List of ``(src_id, dst_id)`` edge tuples.
        clusters:
            Root :class:`~torch_flowgraph.clusters.Cluster` of the module
            hierarchy.
        add_extra_nodes:
            When ``True``, call
            :meth:`~torch_flowgraph.nodes.Node.add_extra_nodes` on every node
            to inject auxiliary :class:`~torch_flowgraph.nodes.TensorNode` and
            :class:`~torch_flowgraph.nodes.ParameterNode` instances.
        """
        assert isinstance(nodes, dict), type(nodes)
        assert isinstance(edges, list)
        assert isinstance(clusters, Cluster)
        assert all(isinstance(n, Node) for n in nodes.values())

        # Store attributes before calling super().__init__ which calls _finalise.
        self.nodes = nodes
        self.edges = edges
        self.clusters = clusters
        self.depth = 0

        self._finalise()

        print("ModelGraph", len(self.nodes), len(self.edges), self.depth)

        if add_extra_nodes:
            data = {}

            extra_nodes = self.nodes.copy()
            extra_edges = self.edges.copy()

            for n in self.nodes.values():
                n.add_extra_nodes(extra_nodes, extra_edges, data)

            self.nodes = extra_nodes
            self.edges = extra_edges

            self._finalise()

            print("ModelGraph (with extra nodes)", len(self.nodes), len(self.edges), self.depth)

    def __repr__(self):
        return f"Model({len(self)} nodes)"

    @classmethod
    def from_model(cls, model, shape, add_extra_nodes=False):
        """Construct a :class:`ModelGraph` by tracing a PyTorch model.

        Delegates to :func:`~torch_flowgraph.model.pytorch.build_model_graph`.

        Parameters
        ----------
        model:
            A ``torch.nn.Module`` whose forward pass will be traced.
        shape:
            Input tensor shape (e.g. ``(1, 3, 224, 224)``).
        add_extra_nodes:
            Passed through to :meth:`__init__`.

        Returns
        -------
        ModelGraph
        """
        from .pytorch import build_model_graph

        return build_model_graph(model, shape, add_extra_nodes)
