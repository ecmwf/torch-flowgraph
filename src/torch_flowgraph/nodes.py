# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Node types for PyTorch computation graph visualization.

This module defines the node hierarchy used to represent elements of a
PyTorch model graph. Nodes correspond to modules, functions, tensors,
parameters, and other graph elements, each carrying the metadata needed
to render them in Graphviz DOT format.
"""

# Mapping from PyTorch operation names to their PyTorch documentation page ids.
# Operations with a value of None are intentionally undocumented (e.g. dunder
# methods). Entries with a key present but value None suppress the "NO PAGE"
# warning while still omitting a hyperlink.
PAGES = {
    "__getitem__": None,
    "__setitem__": None,
    "adaptive_avg_pool2d": "torch.nn.functional.adaptive_avg_pool2d",
    "add_": "torch.Tensor.add_",
    "add": "torch.add",
    "batch_norm": "torch.nn.functional.batch_norm",
    "cat": "torch.cat",
    "chunk": "torch.Tensor.chunk",
    "clone": "torch.Tensor.clone",
    "conv2d": "torch.nn.functional.conv2d",
    "detach": "torch.Tensor.detach",
    "div": "torch.div",
    "dropout": "torch.nn.functional.dropout",
    "exp": "torch.exp",
    "expand_as": "torch.Tensor.expand_as",
    "flatten": "torch.flatten",
    "gelu": "torch.nn.functional.gelu",
    "index_select": "torch.index_select",
    "layer_norm": "torch.nn.functional.layer_norm",
    "linalg_norm": "torch.linalg.norm",
    "linear": "torch.nn.functional.linear",
    "max_pool2d": "torch.nn.functional.max_pool2d",
    "mul": "torch.mul",
    "new_zeros": "torch.Tensor.new_zeros",
    "permute": "torch.permute",
    "randn_like": "torch.randn_like",
    "relu": "torch.nn.functional.relu",
    "reshape": "torch.Tensor.reshape",
    "scaled_dot_product_attention": "torch.nn.functional.scaled_dot_product_attention",
    "scatter_add_": "torch.Tensor.scatter_add_",
    "scatter_reduce_": "torch.Tensor.scatter_reduce_",
    "silu": "torch.nn.functional.silu",
    "sub": "torch.sub",
    "sum": "torch.sum",
    "tensor_split": "torch.tensor_split",
    "to": "torch.nn.functional.to",
    "type_as": "torch.Tensor.type_as",
    "view_as": "torch.Tensor.view_as",
    "view": "torch.Tensor.view",
    "zeros_like": "torch.zeros_like",
}


def url(label):
    """Return the PyTorch documentation URL for a given operation label.

    Looks up *label* in :data:`PAGES`. Returns a fully-qualified URL when a
    documentation page is known, a search-page URL when the label is
    unrecognised (and prints a warning), or ``None`` when the label is a
    single character or explicitly mapped to ``None``.

    Parameters
    ----------
    label:
        PyTorch operation name (e.g. ``"linear"``, ``"cat"``).

    Returns
    -------
    str or None
        Documentation URL, or ``None`` if no link should be emitted.
    """
    if len(label) > 1:
        _sentinel = object()
        page = PAGES.get(label, _sentinel)
        if page is _sentinel:
            return f"https://pytorch.org/docs/stable/search.html?q={label}"
        if page is not None:
            return f"https://pytorch.org/docs/stable/generated/{page}.html"
        # page is None → label is explicitly silenced, emit no link
        return None


def _merge(a, b):
    """Return a shallow copy of dict *a* updated with all entries from dict *b*.

    Parameters
    ----------
    a, b:
        Dictionaries to merge. *b* takes precedence for duplicate keys.

    Returns
    -------
    dict
        New dictionary containing all key-value pairs from both *a* and *b*.
    """
    a = a.copy()
    a.update(b)
    return a


class Node:
    """Base class for all graph nodes.

    Every node has a unique integer identifier *n*, a human-readable *label*,
    and optional references to its parent node and the cluster it belongs to.
    Subclasses extend this class to carry additional payload and to customise
    the Graphviz DOT attributes used during rendering.
    """

    def __init__(self, n, label):
        """Initialise a node.

        Parameters
        ----------
        n:
            Unique integer identifier for this node within the graph.
        label:
            Human-readable display label.
        """
        assert isinstance(n, int)
        self.n = n
        self.parent = None
        self.label = label
        self.cluster = None
        self.rank = None  # Graphviz rank hint: "min", "max", or None

    def __repr__(self):
        return self.label

    def as_dict(self):
        """Serialise the node to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Dictionary with keys ``"n"``, ``"label"``, and ``"_kind"``
            (the class name, used for deserialisation).
        """
        d = {
            "n": self.n,
            "label": self.label,
            "_kind": self.__class__.__name__,
        }
        if self.rank is not None:
            d["rank"] = self.rank
        return d

    def dot_node_kwargs(self, **kwargs):
        """Return Graphviz DOT attributes for this node.

        Parameters
        ----------
        **kwargs:
            Subclass-specific options (e.g. ``links``).

        Returns
        -------
        dict
            Keyword arguments suitable for passing to a DOT node definition,
            at minimum containing a ``"label"`` entry.  A ``"_rank"`` key may
            also be present when :attr:`rank` is set; it is not a real DOT
            attribute but is used by :func:`~torch_flowgraph.plotting.plot_graph`
            to emit ``{ rank=min; ... }`` / ``{ rank=max; ... }`` blocks.
        """
        d = {"label": self.dot_label, "_kind": self.__class__.__name__}
        if self.rank is not None:
            d["_rank"] = self.rank
        return d

    @property
    def dot_label(self):
        """The label string to use in DOT output (defaults to :attr:`label`)."""
        return self.label

    def dot_edge_kwargs(self, other):
        """Return Graphviz DOT attributes for the edge leading to *other*.

        Parameters
        ----------
        other:
            The target node of the edge.

        Returns
        -------
        dict
            Edge attribute keyword arguments (empty by default).
        """
        return {}

    def add_extra_nodes(self, nodes, edges, clusters):
        """Optionally add auxiliary nodes/edges to the graph during finalisation.

        The default implementation does nothing. Subclasses (e.g.
        :class:`FunctionNode`) override this to inject tensor or parameter
        nodes that are referenced by this node's arguments.

        Parameters
        ----------
        nodes:
            Mutable mapping of node-id to :class:`Node` for the whole graph.
        edges:
            Mutable list of ``(src_id, dst_id)`` edge tuples for the whole graph.
        clusters:
            Auxiliary data passed through from the caller (usage is
            subclass-specific).
        """
        return


class InputOutput(Node):
    """Node that tracks tensor shapes flowing in and out of an operation.

    Extends :class:`Node` with *input* and *output* fields that store the
    serialised representation of the tensors consumed and produced by this
    operation.  The edge label between two nodes is derived from these fields.
    """

    def __init__(self, n, label, input=None, output=None):
        """Initialise an InputOutput node.

        Parameters
        ----------
        n:
            Unique integer identifier.
        label:
            Human-readable display label.
        input:
            Serialised representation of the input tensor(s) (as produced by
            the ``_`` helper in :mod:`torch_flowgraph.model.pytorch`).
        output:
            Serialised representation of the output tensor(s).
        """
        super().__init__(n, label)
        self.input = input
        self.output = output

    def as_dict(self):
        """Serialise to a dictionary, adding ``"input"`` and ``"output"`` entries."""
        return _merge(
            super().as_dict(),
            {
                "input": self.input,
                "output": self.output,
            },
        )

    def dot_edge_kwargs(self, other):
        """Return edge attributes carrying tensor shape information as a label.

        Extracts the shape from a ``("_tracker", shape)`` tuple stored in
        *input* or *output* and uses it as the DOT edge label.

        Parameters
        ----------
        other:
            Target node (unused, present for API consistency).

        Returns
        -------
        dict
            Dictionary with a ``"label"`` key showing the tensor shape, or
            an empty dict if no shape information is available.
        """

        def _(x):
            if isinstance(x, (tuple, list)) and len(x) == 2 and x[0] == "_tracker":
                return x[1]

        # TODO: use code from model.py
        if self.output is None:
            if self.input is None:
                return {}
            return dict(label=f" {_(self.input[0])} ")

        return dict(label=f" {_(self.output)} ")


class ModuleNode(InputOutput):
    """Node representing a ``torch.nn.Module`` in the computation graph.

    Inherits all behaviour from :class:`InputOutput` without modification.
    The distinction from :class:`FunctionNode` is semantic: module nodes
    correspond to named sub-modules (e.g. ``Linear``, ``LayerNorm``) whereas
    function nodes correspond to individual torch operations.
    """

    def dot_node_kwargs(self, **kwargs):
        return _merge(
            super().dot_node_kwargs(**kwargs),
            {
                "fillcolor": "#f3f0ff",
                "color": "#d0bfff",
            },
        )


class FunctionNode(InputOutput):
    """Node representing a single PyTorch functional operation.

    In addition to the input/output shape tracking provided by
    :class:`InputOutput`, a :class:`FunctionNode` stores the keyword
    arguments (*params*) passed to the underlying torch function, and can
    inject auxiliary :class:`TensorNode` / :class:`ParameterNode` nodes into
    the graph for non-tracked tensor arguments.
    """

    def __init__(self, n, label, input=None, output=None, params=None):
        """Initialise a FunctionNode.

        Parameters
        ----------
        n:
            Unique integer identifier.
        label:
            Name of the torch operation (e.g. ``"linear"``, ``"cat"``).
        input:
            Serialised input arguments.
        output:
            Serialised output tensor(s).
        params:
            Serialised keyword arguments passed to the operation.
        """
        super().__init__(n, label, input, output)
        self.params = params

    @property
    def dot_label(self):
        """Multi-line DOT label: operation name followed by non-tensor params."""
        if self.params is None:
            return self.label

        label = [self.label]
        for k, v in self.params.items():
            if not (isinstance(v, (tuple, list)) and len(v) == 3 and v[0] in ("_parameter", "_tensor")):
                label.append(f"{k}={v}")

        return "\n".join(label)

    def as_dict(self):
        """Serialise to a dictionary, adding a ``"params"`` entry."""
        return _merge(
            super().as_dict(),
            {
                "params": self.params,
            },
        )

    def add_extra_nodes(self, nodes, edges, data):
        """Inject :class:`TensorNode` and :class:`ParameterNode` auxiliaries.

        Scans *input* and *params* for ``("_tensor", …)`` / ``("_parameter",
        …)`` tuples and creates the corresponding nodes if they have not been
        seen before (keyed by the object id stored in the tuple). The new
        nodes are added to *nodes*, connected to this node via *edges*, and
        placed inside this node's cluster.

        Parameters
        ----------
        nodes:
            Mutable mapping of node-id to :class:`Node`.
        edges:
            Mutable list of ``(src_id, dst_id)`` edge tuples.
        data:
            Mutable cache mapping object-id to the :class:`Node` that was
            already created for it (avoids duplicate nodes for shared tensors).
        """

        def _(x):
            if isinstance(x, (tuple, list)) and len(x) == 3 and x[0] in ("_tensor", "_parameter"):
                p = data.get(x[2])
                if p is None:
                    p = TensorNode(len(nodes), "T", x[1]) if x[0] == "_tensor" else ParameterNode(len(nodes), "P", x[1])
                    nodes[p.n] = p
                    self.cluster.nodes.append(p.n)
                    data[x[2]] = p
                edges.append((p.n, self.n))
                return

            if isinstance(x, (tuple, list)):
                for y in x:
                    _(y)

            if isinstance(x, dict):
                for y in x.values():
                    _(y)

        _((self.input, self.params))

    def dot_node_kwargs(self, **kwargs):
        """Return DOT node attributes, optionally adding a ``URL`` link.

        Parameters
        ----------
        links:
            When ``True``, add a ``URL`` attribute pointing to the PyTorch
            documentation for the operation.  Defaults to ``False``.
        """
        links = kwargs.pop("links", False)
        if links:
            help = url(self.label)
            if help:
                return _merge(
                    super().dot_node_kwargs(**kwargs),
                    {"URL": help},
                )
        return super().dot_node_kwargs(**kwargs)


class LabelNode(Node):
    """A plain labelled node with no additional payload.

    Used for graph entry/exit annotations such as the input tensor shape
    marker ``"x"`` and output tensor shape marker ``"y"``.
    """

    pass


class CollapsedNode(Node):
    """A node that represents a collapsed sub-cluster during depth-limited rendering.

    When a graph is rendered at a maximum depth that hides the internals of a
    cluster, the cluster is replaced by a single :class:`CollapsedNode`. The
    node accumulates the outgoing :class:`FunctionNode` instances of its
    members so that it can forward edge-label information to adjacent nodes.

    The optional :attr:`explore_url` attribute, when set, is emitted as the
    Graphviz ``URL`` attribute so that the node becomes a clickable hyperlink
    in SVG output.  This is used by the interactive Explore mode to let users
    expand collapsed clusters by clicking on them.
    """

    def __init__(self, *arg, **kwargs):
        """Initialise a CollapsedNode, setting up the outgoing node list."""
        super().__init__(*arg, **kwargs)
        self._outgoing = []
        self.explore_url: str | None = None
        # Maps new target-node-id → list of original source nodes that
        # contributed to the edge (populated & remapped by explore()).
        # Used by dot_edge_kwargs to apply the 1-vs-many label rule.
        self._sources_for: dict[int, list] = {}
        # Set to True by explore() when this node represents a merged-repeated run.
        self.is_repeated: bool = False

    def dot_node_kwargs(self, **kwargs):
        """Return DOT attributes that render this node as a styled rounded rectangle.

        Nodes that represent a merged-repeated run (``is_repeated=True``) use
        a double border and a warm colour to visually distinguish them.
        """
        d = _merge(
            super().dot_node_kwargs(**kwargs),
            {
                "color": "#bac8ff",
                "fontcolor": "#212529",
                "shape": "rect",
                "fillcolor": "#edf2ff",
                "style": "filled,rounded",
                "penwidth": 2,
            },
        )
        if self.is_repeated:
            d.update(
                {
                    "fillcolor": "#fff3e0",
                    "color": "#ff922b",
                    "penwidth": 2.5,
                    "peripheries": 2,
                }
            )
        if self.explore_url is not None:
            d["URL"] = self.explore_url
        return d

    def dot_edge_kwargs(self, other):
        """Return edge attributes for the edge to *other*.

        When per-target source tracking is available (``_sources_for``
        populated by :meth:`~torch_flowgraph.graph.GraphBase.explore`):

        * Exactly one contributing edge → delegate to that source's
          :meth:`dot_edge_kwargs` so the tensor-shape label is preserved.
        * More than one contributing edge → return an empty dict (no label,
          to avoid misleading the reader with an arbitrary single shape).

        Falls back to the legacy ``_outgoing`` mechanism when ``_sources_for``
        has no entry for this target.

        Parameters
        ----------
        other:
            Target node of the edge.

        Returns
        -------
        dict
            Edge attribute dictionary.
        """
        if other.n in self._sources_for:
            sources = self._sources_for[other.n]
            if len(sources) == 1:
                return sources[0].dot_edge_kwargs(other)
            return {}  # multiple contributing edges → omit label

        # Legacy fallback (collapsed() path or nodes without _sources_for data).
        if self._outgoing:
            labelled = sorted(self._outgoing, key=lambda x: x.n)
            return labelled[-1].dot_edge_kwargs(other)

        return _merge(
            super().dot_edge_kwargs(other),
            {
                "label": "?",
            },
        )

    def outgoing(self, node):
        """Register *node* as an outgoing function node of this collapsed cluster.

        Only :class:`FunctionNode` instances are accepted; other node types
        are silently ignored.

        Parameters
        ----------
        node:
            A node that was an internal outgoing edge source before collapsing.
        """
        if isinstance(node, FunctionNode):
            self._outgoing.append(node)


class ShapeNode(Node):
    """A node that carries tensor shape information on its outgoing edge.

    The *shape* tuple is rendered as the label of every edge leaving this node,
    providing a visual indication of tensor dimensions in the graph.
    """

    def __init__(self, n, label, shape):
        """Initialise a ShapeNode.

        Parameters
        ----------
        n:
            Unique integer identifier.
        label:
            Human-readable display label.
        shape:
            Tensor shape as an iterable of integers; stored internally as a
            ``tuple``.
        """
        super().__init__(n, label)
        self.shape = tuple(shape)

    def dot_node_kwargs(self, **kwargs):
        return _merge(
            super().dot_node_kwargs(**kwargs),
            {
                "fillcolor": "#ebfbee",
                "color": "#b2f2bb",
            },
        )

    def dot_edge_kwargs(self, other):
        """Return edge attributes with the tensor shape as label.

        Parameters
        ----------
        other:
            Target node (unused).

        Returns
        -------
        dict
            Dictionary with ``"label"`` set to the string representation of
            :attr:`shape`.
        """
        return _merge(
            super().dot_edge_kwargs(other),
            {
                "label": f" {self.shape} ",
            },
        )

    def as_dict(self):
        """Serialise to a dictionary, adding a ``"shape"`` entry."""
        return _merge(
            super().as_dict(),
            {
                "shape": self.shape,
            },
        )


class ParameterNode(ShapeNode):
    """Node representing a learnable ``torch.nn.Parameter``.

    Rendered as an inverted-house shape with an orange colour scheme so that
    weight/bias tensors are visually distinguishable from data-flow tensors.
    """

    def dot_node_kwargs(self, **kwargs):
        """Return DOT attributes for the inverted-house parameter style."""
        return _merge(
            super().dot_node_kwargs(**kwargs),
            {
                "shape": "invhouse",
                "color": "#ffd8a8",
                "fillcolor": "#fff4e6",
                "style": "filled",
                "fontcolor": "#212529",
            },
        )


class TensorNode(ShapeNode):
    """Node representing a non-tracked ``torch.Tensor`` (e.g. a buffer).

    Rendered as a house shape with a green colour scheme to distinguish
    constant or buffer tensors from learnable parameters and from the main
    data flow.
    """

    def dot_node_kwargs(self, **kwargs):
        """Return DOT attributes for the house-shaped tensor style."""
        return _merge(
            super().dot_node_kwargs(**kwargs),
            {
                "shape": "house",
                "color": "#b2f2bb",
                "fillcolor": "#ebfbee",
                "style": "filled",
                "fontcolor": "#212529",
            },
        )


class UnknownNode(Node):
    """Fallback node for unrecognised node types encountered during deserialisation.

    Any extra keyword arguments supplied at construction time are stored in
    :attr:`kwargs` for inspection but are otherwise ignored.
    """

    def __init__(self, n, label, **kwargs):
        """Initialise an UnknownNode, storing unrecognised keyword arguments.

        Parameters
        ----------
        n:
            Unique integer identifier.
        label:
            Human-readable display label.
        **kwargs:
            Arbitrary extra attributes preserved for debugging.
        """
        super().__init__(n, label)
        self.kwargs = kwargs
