# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""PyTorch model graph tracing via tensor subclassing.

This module traces a PyTorch ``nn.Module``'s forward pass and constructs a
:class:`~torch_flowgraph.model.graph.ModelGraph` that captures:

* Module boundaries (via ``register_forward_hook`` / ``register_forward_pre_hook``).
* Individual tensor operations (via :meth:`Tracker.__torch_function__`).
* Tensor shapes for input, output, and parameter nodes.

The main entry point is :func:`build_model_graph`.

Implementation strategy
-----------------------
A custom :class:`Tracker` subclass of ``torch.Tensor`` intercepts every
torch function call through ``__torch_function__``.  A singleton
:class:`GraphBuilder` instance is used as a global accumulator during the
forward pass; it records every module entry/exit and every function call.

References
----------
* https://pytorch.org/docs/stable/notes/extending.html
* https://discuss.pytorch.org/t/subclassing-torch-tensor/23754
"""

import logging
import os
import threading

import torch

import torch_flowgraph.patches as _patches  # noqa: F401 — apply torch meta-device patches

from torch_flowgraph.clusters import Cluster
from torch_flowgraph.model.graph import ModelGraph
from torch_flowgraph.nodes import FunctionNode
from torch_flowgraph.nodes import ModuleNode
from torch_flowgraph.nodes import Node
from torch_flowgraph.nodes import ShapeNode

logger = logging.getLogger(__name__)
_thread_local = threading.local()


def _rss_mb() -> float:
    """Return current RSS in MiB by reading /proc/self/status."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def _log_mem(label: str) -> None:
    logger.debug("[mem] %-45s  rss=%6.0f MiB  pid=%d", label, _rss_mb(), os.getpid())


def _(output):
    """Recursively serialise a PyTorch value to a JSON-safe representation.

    Converts the rich PyTorch type hierarchy to plain Python primitives and
    tagged lists so that node input/output fields can be stored as JSON.

    * ``torch.nn.parameter.Parameter`` → ``["_parameter", shape, id]``
    * :class:`Tracker` → ``["_tracker", shape]``
    * ``torch.Tensor`` → ``["_tensor", shape, id]``
    * ``slice`` → ``["_slice", [start, stop, step]]``
    * Scalars (int, float, str, bool), ``None``, ``Ellipsis`` → unchanged
    * Everything else → ``str(output)``

    Parameters
    ----------
    output:
        Any value encountered as a torch function argument or return value.

    Returns
    -------
    object
        JSON-serialisable representation of *output*.
    """
    if isinstance(output, tuple):
        if len(output) == 1:
            return _(output[0])
        return tuple(_(o) for o in output)

    if isinstance(output, list):
        if len(output) == 1:
            return _(output[0])
        return list(_(o) for o in output)

    if isinstance(output, dict):
        return {_(k): _(v) for k, v in output.items()}

    if isinstance(output, torch.nn.parameter.Parameter):
        # Ignore learnable parameters — record shape and object identity only.
        return ["_parameter", tuple(_(x) for x in output.shape), id(output)]

    if isinstance(output, Tracker):
        return ["_tracker", tuple(_(x) for x in output.shape)]

    if isinstance(output, torch.Tensor):
        # Non-tracked tensor (e.g. a buffer or constant).
        return ["_tensor", tuple(_(x) for x in output.shape), id(output)]

    if output is None:
        return None

    if isinstance(output, (int, float, str, bool)):
        return output

    if output is Ellipsis:
        return "..."

    if isinstance(output, slice):
        return ["_slice", _((output.start, output.stop, output.step))]

    return str(output)


def _debug_args(x):
    """Recursively convert *x* to a human-readable string for debugging.

    Replaces :class:`Tracker` and ``torch.Tensor`` instances with compact
    string summaries so that large argument lists can be printed without
    flooding the console.

    Parameters
    ----------
    x:
        Value to convert (may be a nested list / tuple / dict).

    Returns
    -------
    object
        Debuggable representation of *x*.
    """
    if isinstance(x, list):
        return [_debug_args(i) for i in x]

    if isinstance(x, tuple):
        return tuple([_debug_args(i) for i in x])

    if isinstance(x, dict):
        return {k: _debug_args(v) for k, v in x.items()}

    if isinstance(x, Tracker):
        return f"Tracker[{id(x)}({x._shape})]"

    if isinstance(x, torch.Tensor):
        return f"Tensor[{id(x)}({x.shape})]"

    return x


class _LazyBridge:
    """Deferred bridge :class:`~torch_flowgraph.nodes.FunctionNode`.

    A *bridge* is a synthetic node inserted by :meth:`GraphBuilder.leave_node`
    when a module returns a plain ``torch.Tensor`` (i.e. the module's C++
    internals bypassed ``__torch_function__``).  Eager creation produced
    orphan nodes when the wrapped output tensor was never consumed by any
    downstream torch op (e.g. when a caller unpacks the return value and
    discards the plain tensor).

    :class:`_LazyBridge` is stored as the ``owner`` attribute of every
    wrapped tensor returned by the bridge logic.  The actual
    :class:`~torch_flowgraph.nodes.FunctionNode` is only allocated, registered
    in :attr:`GraphBuilder.nodes`, attached to its cluster, and wired to its
    inputs the first time :attr:`n` is read — which only happens when
    :meth:`GraphBuilder.connect` actually needs the bridge as an edge source.

    If no wrapped tensor is ever consumed, the bridge is silently dropped.
    """

    def __init__(self, builder, module, current_cluster, input):
        self._builder = builder
        self._module = module
        self._cluster = current_cluster
        self._input = input
        self._node = None

    @property
    def n(self):
        """Materialise the bridge node on first access and return its id."""
        if self._node is None:
            n = len(self._builder.nodes)
            self._node = FunctionNode(n, self._module._get_name())
            self._builder.nodes[n] = self._node
            if self._cluster is not None:
                self._cluster.nodes.append(n)
            # Wire every Tracker in the module's input to this bridge so the
            # graph stays connected even though the internals were opaque.
            self._builder.connect(self._input, self._node)
        return self._node.n


class _LazyOp:
    """Deferred :class:`~torch_flowgraph.nodes.FunctionNode` for a torch op.

    The :class:`FunctionNode` is allocated, registered in
    :attr:`GraphBuilder.nodes`, attached to its cluster, and wired to its
    input :class:`Tracker` arguments **only when** at least one of the op's
    output trackers is actually consumed by another torch operation —
    i.e. the first time :attr:`n` is read by :meth:`GraphBuilder.connect`.

    If none of the op's output trackers are ever fed into a subsequent torch
    function (e.g. ``torch.tensor_split(x, 1, dim=0)`` whose chunks are only
    used to read ``.shape``, as in
    ``anemoi.models.distributed.shapes.get_shape_shards``), the node is
    silently dropped — preventing orphan sinks in the graph.

    The owners of the input :class:`Tracker` arguments are snapshotted at
    construction time, **before** ``attach_owner`` re-points the output
    trackers to ``self``.  This matters for in-place ops where the output
    aliases an input tensor: without the snapshot, materialisation would
    read the (mutated) ``.owner`` and produce a self-loop.
    """

    def __init__(self, builder, func, args, kwargs, result, cluster):
        self._builder = builder
        self._func = func
        self._cluster = cluster
        self._node = None
        # Snapshot input Tracker owners NOW (before attach_owner mutates any).
        self._input_owners: list = []
        self._collect_owners(args)
        self._collect_owners(kwargs)
        # Serialize immediately so the raw tensor objects (potentially large)
        # can be GC'd as soon as this __init__ returns.
        self._s_args = _(args)
        self._s_kwargs = _(kwargs)
        self._s_result = _(result)

    def _collect_owners(self, x):
        """Recursively gather the ``owner`` of every :class:`Tracker` in *x*."""
        if isinstance(x, Tracker):
            owner = getattr(x, "owner", None)
            if owner is not None:
                self._input_owners.append(owner)
            return
        if isinstance(x, (tuple, list)):
            for i in x:
                self._collect_owners(i)
        elif isinstance(x, dict):
            for i in x.values():
                self._collect_owners(i)

    @property
    def n(self):
        """Materialise the function node on first access and return its id."""
        if self._node is None:
            nid = len(self._builder.nodes)
            self._node = FunctionNode(
                nid,
                self._func.__name__,
                input=self._s_args,
                output=self._s_result,
                params=self._s_kwargs,
            )
            self._builder.nodes[nid] = self._node
            if self._cluster is not None:
                self._cluster.nodes.append(nid)
            # Wire incoming edges from the snapshotted input owners.  Reading
            # ``owner.n`` here will recursively materialise any upstream
            # _LazyOp instances that have not yet been materialised.
            for owner in self._input_owners:
                self._builder.edges.append((owner.n, nid))
        return self._node.n


class GraphBuilder:
    """Accumulates graph structure during a single model forward pass.

    One :class:`GraphBuilder` instance acts as the global state for a tracing
    run.  It maintains:

    * A stack of :class:`~torch_flowgraph.clusters.Cluster` objects that
      mirrors the current ``nn.Module`` call stack.
    * Mappings from node-id to :class:`~torch_flowgraph.nodes.Node`.
    * A list of directed edges.
    * Forward hook handles that are removed after tracing.

    Attributes
    ----------
    hooks : list
        Forward hook handles returned by ``register_forward_hook`` /
        ``register_forward_pre_hook``.  Removed after tracing.
    nodes : dict[int, Node]
        Accumulated node mapping.
    edges : list[tuple[int, int]]
        Accumulated edge list.
    top : Cluster or None
        Root cluster created at the start of tracing.
    cluster_stack : list[Cluster]
        Stack of clusters reflecting the current module call depth.
    seen_clusters : set
        Clusters that have already been visited (unused; reserved).
    other : dict
        Reserved for future use.
    """

    def __init__(self):
        """Initialise an empty GraphBuilder."""
        self.hooks = []
        self.nodes = {}
        self.edges = []
        self.top = None
        self.cluster_stack = []
        self.seen_clusters = set()
        self.other = {}
        self._op_count = 0

    def push_cluster(self, module):
        """Push a new cluster onto the stack for the given module.

        Creates a :class:`~torch_flowgraph.clusters.Cluster` named after
        ``module._get_name()``, links it to the current top of the stack as
        its parent, and appends it as a child of the current cluster.

        Parameters
        ----------
        module:
            The ``torch.nn.Module`` being entered.

        Returns
        -------
        Cluster
            The newly created cluster.
        """
        c = Cluster(
            name=module._get_name(),
            parent=self.cluster_stack[-1] if len(self.cluster_stack) > 0 else None,
        )

        if len(self.cluster_stack) > 0:
            self.cluster_stack[-1].children.append(c)

        self.cluster_stack.append(c)

        return c

    def pop_cluster(self, module):
        """Pop the topmost cluster off the stack when leaving *module*.

        Parameters
        ----------
        module:
            The ``torch.nn.Module`` being exited (used for logging only).
        """
        self.cluster_stack.pop()

    def shape_node(self, label, shape):
        """Create and register a :class:`~torch_flowgraph.nodes.ShapeNode`.

        Parameters
        ----------
        label:
            Display label (e.g. ``"x"`` for the input, ``"y"`` for the
            output).
        shape:
            Tensor shape tuple.

        Returns
        -------
        ShapeNode
        """
        n = len(self.nodes)
        shape = ShapeNode(n, label, shape)
        self.nodes[n] = shape
        return shape

    def ensure_payload(self, module):
        """Return the :class:`~torch_flowgraph.nodes.ModuleNode` for *module*, creating it if needed.

        Each ``nn.Module`` instance gets exactly one :class:`ModuleNode` the
        first time it is seen.  Subsequent calls return the same node.

        Parameters
        ----------
        module:
            A ``torch.nn.Module`` instance.

        Returns
        -------
        ModuleNode
        """
        if hasattr(module, "payload"):
            return module.payload

        n = len(self.nodes)
        payload = ModuleNode(n, module._get_name())
        module.payload = payload
        self.nodes[n] = payload

        if len(self.cluster_stack) > 0:
            cluster = self.cluster_stack[-1]
            cluster.nodes.append(n)

        return payload

    def get_owners(self, input, result=None):
        """Collect :class:`~torch_flowgraph.nodes.Node` owners from nested tensor structures.

        Recursively traverses *input* and collects the ``owner`` attribute of
        every :class:`Tracker` found.

        Parameters
        ----------
        input:
            Possibly-nested structure (list, tuple, dict, or tensor).
        result:
            Accumulator list; a new list is created when ``None``.

        Returns
        -------
        list[Node]
            All owner nodes found in *input*.
        """
        if result is None:
            result = []

        if isinstance(input, (tuple, list)):
            for i in input:
                self.get_owners(i, result)

        if isinstance(input, dict):
            for i in input.values():
                self.get_owners(i, result)

        if isinstance(input, torch.Tensor):
            if hasattr(input, "owner"):
                result.append(input.owner)
            else:
                print("." * len(self.cluster_stack), "no owner", type(input), id(input))

        return result

    def set_owner(self, output, owner):
        """Recursively assign *owner* as the ``owner`` attribute of every tensor in *output*.

        Parameters
        ----------
        output:
            Possibly-nested structure (list, tuple, dict, or tensor).
        owner:
            The :class:`~torch_flowgraph.nodes.Node` to assign.
        """
        assert isinstance(owner, Node)

        if isinstance(output, (tuple, list)):
            for i in output:
                self.set_owner(i, owner)

        if isinstance(output, dict):
            for i in output.values():
                self.set_owner(i, owner)

        if isinstance(output, torch.Tensor):
            output.owner = owner

    def connect(self, input, payload):
        """Add edges from every :class:`Tracker` tensor in *input* to *payload*.

        Recursively traverses *input*; for each :class:`Tracker` found, an
        edge ``(tracker.owner.n, payload.n)`` is appended to :attr:`edges`.
        Parameters, plain tensors, and dicts are handled recursively but
        do not create edges.

        Parameters
        ----------
        input:
            Possibly-nested structure of tensors and other values.
        payload:
            Target :class:`~torch_flowgraph.nodes.Node` for all edges.
        """
        if isinstance(input, (tuple, list)):
            for i in input:
                self.connect(i, payload)

        if isinstance(input, dict):
            for i in input.values():
                self.connect(i, payload)

        if isinstance(input, torch.nn.parameter.Parameter):
            # Ignore learnable parameters.
            return "parameter"

        if isinstance(input, Tracker):
            self.edges.append((input.owner.n, payload.n))

        if isinstance(input, torch.Tensor):
            # Non-tracked tensor.
            return "tensor"

    def enter_node(self, module, input):
        """Forward pre-hook: push the module's cluster onto the stack.

        Called by PyTorch immediately before the module's ``forward`` method.

        Parameters
        ----------
        module:
            The entering ``torch.nn.Module``.
        input:
            The (unwrapped) input to the module (unused here).
        """
        self.push_cluster(module)

    def register_forward_pre_hook(self, module):
        """Register :meth:`enter_node` as a forward pre-hook on *module*.

        Guarded against double-registration: if the same module instance
        appears as a shared grandchild under multiple parents,
        ``model.apply()`` would visit it more than once.  A flag on the
        module prevents a second hook from being attached.

        Parameters
        ----------
        module:
            ``torch.nn.Module`` to hook.
        """
        if not getattr(module, "_tfg_pre_hooked", False):
            module._tfg_pre_hooked = True
            self.hooks.append(module.register_forward_pre_hook(self.enter_node))

    def leave_node(self, module, input, output):
        """Forward hook: pop the module's cluster and fix broken tracking.

        Called by PyTorch immediately after the module's ``forward`` method
        returns.

        If the output contains any plain ``torch.Tensor`` (i.e. not a
        :class:`Tracker`), the module's C++ internals bypassed
        ``__torch_function__`` entirely (e.g. ``torch_scatter.scatter``).
        In that case a bridge :class:`~torch_flowgraph.nodes.FunctionNode` is
        created, the module's Tracker inputs are connected to it, and the
        plain-tensor outputs are promoted to Trackers so the ownership chain
        survives into the next module.

        Parameters
        ----------
        module:
            The exiting ``torch.nn.Module``.
        input:
            The (unwrapped) inputs to the module.
        output:
            The module's return value.
        """
        # Capture the cluster for this module BEFORE popping, so the bridge
        # node ends up inside the right subgraph.
        current_cluster = self.cluster_stack[-1] if self.cluster_stack else None
        self.pop_cluster(module)

        def _has_plain_tensor(x):
            if isinstance(x, torch.Tensor) and not isinstance(x, Tracker):
                return True
            if isinstance(x, (tuple, list)):
                return any(_has_plain_tensor(i) for i in x)
            if isinstance(x, dict):
                return any(_has_plain_tensor(i) for i in x.values())
            return False

        def _has_tracker(x):
            if isinstance(x, Tracker):
                return True
            if isinstance(x, (tuple, list)):
                return any(_has_tracker(i) for i in x)
            if isinstance(x, dict):
                return any(_has_tracker(i) for i in x.values())
            return False

        if not _has_plain_tensor(output):
            return  # all outputs already tracked — nothing to do

        if not _has_tracker(input):
            # No Tracker inputs to wire up: the module is a data source
            # (e.g. TrainableTensor / positional encoding) whose output is
            # static data, not a step in the data-flow.  Leave it as a plain
            # tensor; it will be picked up as an extra node when needed.
            return

        logger.debug("[bridge] %s output is untracked — inserting bridge node", module._get_name())

        # The bridge is created LAZILY: only when at least one of the wrapped
        # output tensors is actually consumed downstream (i.e. when its
        # ``owner.n`` is read by ``connect()``).  If none of the wrapped
        # tensors are ever used (e.g. the caller unpacks the return value and
        # discards the offending plain tensor, as happens with ``edge_attr``
        # returned but unused by ``GraphTransformerBaseMapper``), no node is
        # added to the graph and no spurious incoming edges are recorded.
        bridge = _LazyBridge(self, module, current_cluster, input)

        def _wrap(x):
            if isinstance(x, torch.Tensor) and not isinstance(x, Tracker):
                x = x.as_subclass(Tracker)
                x.owner = bridge
            elif isinstance(x, (tuple, list)):
                x = type(x)(_wrap(i) for i in x)
            elif isinstance(x, dict):
                x = {k: _wrap(v) for k, v in x.items()}
            return x

        return _wrap(output)

    def register_forward_hook(self, module):
        """Register :meth:`leave_node` as a forward hook on *module*.

        Guarded against double-registration for the same reason as
        :meth:`register_forward_pre_hook`.

        Parameters
        ----------
        module:
            ``torch.nn.Module`` to hook.
        """
        if not getattr(module, "_tfg_post_hooked", False):
            module._tfg_post_hooked = True
            self.hooks.append(module.register_forward_hook(self.leave_node))

    def register_function(self, func, args, kwargs, result):
        """Record a torch function call as a :class:`~torch_flowgraph.nodes.FunctionNode`.

        Creates a new :class:`FunctionNode`, adds it to the current cluster,
        and connects all :class:`Tracker` arguments to it via :meth:`connect`.

        Parameters
        ----------
        func:
            The torch function that was called.
        args:
            Positional arguments passed to *func*.
        kwargs:
            Keyword arguments passed to *func*.
        result:
            Return value of *func*.

        Returns
        -------
        FunctionNode
            The newly created node.
        """
        n = len(self.nodes)
        name = func.__name__

        if n % 100 == 0:
            _log_mem(f"register_function node={n} ({name})")

        f = FunctionNode(n, name, input=_(args), output=_(result), params=_(kwargs))
        self.nodes[n] = f
        cluster = self.cluster_stack[-1]
        cluster.nodes.append(n)

        self.connect(args, f)
        self.connect(kwargs, f)

        return f

    def _fix_inplace_dead_ends(self):
        """Add edges from dead-end in-place nodes to their next consumer.

        ``__setitem__`` and no-op ``to`` operations modify a VIEW of a base
        tensor in place.  Because ``args[0]`` in ``__torch_function__`` is the
        VIEW (not the base), the base tensor's ``owner`` is never updated, so
        downstream reads of the base don't trace through these nodes.

        Post-hoc fix: for each dead-end setitem/to node S, BFS up the ancestor
        chain until an ancestor is found that has a child with a higher node-id
        than S (the "next consumer" of the same base).  Add edge S → first
        such child.
        """
        from collections import deque

        has_child = {a for a, b in self.edges}
        parent_of = {}
        for a, b in self.edges:
            parent_of.setdefault(b, []).append(a)
        child_of = {}
        for a, b in self.edges:
            child_of.setdefault(a, []).append(b)

        for nid, node in list(self.nodes.items()):
            if nid in has_child:
                continue
            if node.label not in {"__setitem__", "to"}:
                continue
            visited = {nid}
            queue = deque(parent_of.get(nid, []))
            while queue:
                p = queue.popleft()
                if p in visited:
                    continue
                visited.add(p)
                next_uses = [c for c in child_of.get(p, []) if c > nid and c not in visited]
                if next_uses:
                    self.edges.append((nid, min(next_uses)))
                    has_child.add(nid)
                    break
                queue.extend(parent_of.get(p, []))

    def build_model_graph(self, model, shape, add_extra_nodes=False):
        """Run the model forward pass and construct the :class:`~torch_flowgraph.model.graph.ModelGraph`.

        Steps:

        1. Push a root cluster for the top-level ``nn.Module``.
        2. Register forward hooks on all sub-modules.
        3. Create a :class:`Tracker` input tensor and run the model.
        4. Connect the output tensor to a sink :class:`~torch_flowgraph.nodes.ShapeNode`.
        5. Remove all hooks.
        6. Wrap the accumulated data in a :class:`~torch_flowgraph.model.graph.ModelGraph`.

        Parameters
        ----------
        model:
            ``torch.nn.Module`` to trace.
        shape:
            Input tensor shape.
        add_extra_nodes:
            Forwarded to :class:`~torch_flowgraph.model.graph.ModelGraph`.

        Returns
        -------
        ModelGraph
        """
        _log_mem("build_model_graph: start")
        top = self.push_cluster(torch.nn.Module())

        model.apply(self.register_forward_hook)
        model.apply(self.register_forward_pre_hook)
        _log_mem(f"build_model_graph: hooks registered ({len(self.hooks)})")

        with torch.no_grad():
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

            if isinstance(shape, dict):
                x_node = {name: self.shape_node(f'x_{name}', s) for name, s in shape.items()}
                input = {name: Tracker(torch.empty(s, device='meta')) for name, s in shape.items()}
                for name, node in x_node.items():
                    input[name].owner = node
                    node.rank = "min"
            else:
                input = Tracker(torch.empty(shape, device='meta'))
                x_node = self.shape_node("x", shape)
                input.owner = x_node
                x_node.rank = "min"

            _log_mem("build_model_graph: forward pass starting")
            output = model(input)
            _log_mem(f"build_model_graph: forward pass done  nodes={len(self.nodes)}  edges={len(self.edges)}")

            if isinstance(output, dict):
                y_nodes = []
                for name, tensor in output.items():
                    assert getattr(tensor, "owner", None) is not None, (name, type(tensor), tensor)
                    yn = self.shape_node(f"y_{name}", tensor.shape)
                    yn.rank = "max"
                    self.connect(tensor, yn)
                    y_nodes.append(yn)
            elif isinstance(output, tuple):
                y_nodes = []
                for i, tensor in enumerate(output):
                    assert getattr(tensor, "owner", None) is not None, (i, type(tensor), tensor)
                    yn = self.shape_node(f"y_{i}", tensor.shape)
                    yn.rank = "max"
                    self.connect(tensor, yn)
                    y_nodes.append(yn)
            else:
                assert getattr(output, "owner", None) is not None, (type(output), output)
                y_node = self.shape_node("y", output.shape)
                y_node.rank = "max"
                self.connect(output, y_node)
                y_nodes = [y_node]

        assert len(top.nodes) == 0
        assert len(top.children) == 1

        self._fix_inplace_dead_ends()
        _log_mem(f"build_model_graph: dead-end fix done  edges={len(self.edges)}")

        for h in self.hooks:
            h.remove()

        model_cluster = top.children[0]
        root = Cluster(name=None, visible=False)
        if isinstance(x_node, dict):
            root.nodes = [xn.n for xn in x_node.values()] + [yn.n for yn in y_nodes]
        else:
            root.nodes = [x_node.n] + [yn.n for yn in y_nodes]
        root.children = [model_cluster]
        model_cluster.parent = root

        _log_mem("build_model_graph: building ModelGraph")
        graph = ModelGraph(self.nodes, self.edges, root, add_extra_nodes)
        _log_mem("build_model_graph: done")
        return graph


# Set of operation names that modify a tensor in-place but return None.
IN_PLACE_OPS = {"__setitem__"}


class Tracker(torch.Tensor):
    """A ``torch.Tensor`` subclass that records every operation in the graph.

    :class:`Tracker` overrides ``__torch_function__`` to intercept every
    PyTorch operation that involves a tracked tensor.  On each call it:

    1. Executes the original operation via ``super().__torch_function__``.
    2. Creates a :class:`~torch_flowgraph.nodes.FunctionNode` via the global
       :data:`BUILDER`.
    3. Assigns the new node as the ``owner`` of every :class:`Tracker` in the
       result.

    The ``_track`` flag allows shape queries (e.g. inside :attr:`_shape`) to
    bypass recording and avoid infinite recursion.
    """

    _track = True

    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        """Create a new Tracker wrapping tensor *x*."""
        return super().__new__(cls, x, *args, **kwargs)  # noqa

    @property
    def _shape(self):
        """Return the tensor shape without triggering operation tracking."""
        self._track = False
        result = self.shape
        self._track = True
        return result

    def __repr__(self):
        """Include the owner node label in the repr when available."""
        if hasattr(self, "owner"):
            return f"{super().__repr__()} -> {self.owner}"
        return f"{super().__repr__()}"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Intercept every torch function call and record it in the graph.

        Skips recording when the first argument's :attr:`_track` flag is
        ``False`` (e.g. during shape queries).

        For in-place operations (name ends with ``_``, has ``inplace=True``,
        or is in :data:`IN_PLACE_OPS`) the first argument is treated as the
        result so its owner is updated correctly.

        Parameters
        ----------
        func:
            The torch function being called.
        types:
            Collection of tensor types present in *args* (passed through to
            ``super``).
        args:
            Positional arguments to *func*.
        kwargs:
            Keyword arguments to *func*.

        Returns
        -------
        object
            The result of the original torch function call.
        """
        if len(args) > 0 and isinstance(args[0], Tracker) and not args[0]._track:
            ret = super().__torch_function__(func, types, args, kwargs)
            return ret

        if kwargs is None:
            kwargs = {}

        builder = _thread_local.builder
        builder._op_count += 1
        if builder._op_count % 100 == 0:
            _log_mem(f"op #{builder._op_count} nodes={len(builder.nodes)} {func.__name__}")

        def is_in_place(func, kwargs):
            """Return True when *func* modifies its first argument in-place."""
            name = func.__name__
            return (
                (name.endswith("_") and not name.startswith("_"))
                or kwargs.get("inplace", False)
                or name in IN_PLACE_OPS
            )

        # Use meta tensors to compute output shapes without allocating real storage.
        # _to_meta uses DisableTorchFunctionSubclass to access .size() and .dtype
        # without triggering recursive __torch_function__ dispatch (accessing .dtype
        # on a Tracker subclass dispatches through __torch_function__, which would
        # call _to_meta again, eventually hitting RecursionError and leaving
        # _track=False on the original tensor — breaking all subsequent ops).
        def _to_meta(x):
            if isinstance(x, Tracker):
                with torch._C.DisableTorchFunctionSubclass():
                    shape = x.size()
                    dtype = x.dtype
                return torch.empty(shape, dtype=dtype, device="meta")
            if isinstance(x, torch.Tensor):
                return torch.empty(x.shape, dtype=x.dtype, device="meta")
            return x

        def _meta_to_tracker(x):
            if isinstance(x, torch.Tensor):
                return x.as_subclass(Tracker)  # zero-cost: reinterpret, no copy
            return x

        def _map_nested(fn, x):
            if isinstance(x, (list, tuple)):
                return type(x)(_map_nested(fn, i) for i in x)
            if isinstance(x, dict):
                return {k: _map_nested(fn, v) for k, v in x.items()}
            return fn(x)

        inp = is_in_place(func, kwargs)
        try:
            meta_result = func(*_map_nested(_to_meta, args), **_map_nested(_to_meta, kwargs))
            if meta_result is None:
                result = None
            elif inp:
                # In-place: return the original Tracker (not a fresh meta view)
                result = args[0] if isinstance(args[0], Tracker) else _map_nested(_meta_to_tracker, meta_result)
            else:
                result = _map_nested(_meta_to_tracker, meta_result)
        except Exception as _meta_err:
            # Log which ops fail meta so we can identify OOM sources.
            _shapes = [getattr(a, 'shape', None) for a in (args if isinstance(args, (list, tuple)) else [args]) if hasattr(a, 'shape')]
            logger.debug("[meta-fail] %s  shapes=%s  err=%s", func.__name__, _shapes, _meta_err)
            result = super().__torch_function__(func, types, args, kwargs)

        f = None

        def attach_owner(x):
            """Assign *f* as the owner of every Tracker tensor in *x*.

            *f* is a :class:`_LazyOp` proxy: the underlying
            :class:`FunctionNode` is only created (and its incoming edges
            wired) the first time a downstream op reads ``f.n``.  If none of
            the produced trackers are ever consumed by another torch op, the
            node is silently dropped.
            """
            nonlocal f
            if isinstance(x, Tracker):
                if f is None:
                    cluster = builder.cluster_stack[-1] if builder.cluster_stack else None
                    f = _LazyOp(builder, func, args, kwargs, result, cluster)
                x.owner = f
                return

            if isinstance(x, (tuple, list)):
                for i in x:
                    attach_owner(i)

            if isinstance(x, dict):
                for i in x.values():
                    attach_owner(i)

        if result is None and inp:
            attach_owner(args[0])
        else:
            attach_owner(result)

        return result


def build_model_graph(model, shape, add_extra_nodes=False):
    """Trace *model* and return a :class:`~torch_flowgraph.model.graph.ModelGraph`.

    Creates a fresh :class:`GraphBuilder`, runs the forward pass with a
    :class:`Tracker` input of the given *shape*, and returns the resulting
    graph.  The global ``BUILDER`` variable is set so that
    :meth:`Tracker.__torch_function__` can record operations.

    Parameters
    ----------
    model:
        ``torch.nn.Module`` to trace.
    shape:
        Input tensor shape (e.g. ``(1, 3, 224, 224)``).
    add_extra_nodes:
        When ``True``, auxiliary :class:`~torch_flowgraph.nodes.TensorNode`
        and :class:`~torch_flowgraph.nodes.ParameterNode` nodes are injected
        into the graph for non-tracked tensor arguments.

    Returns
    -------
    ModelGraph
    """
    _thread_local.builder = GraphBuilder()
    return _thread_local.builder.build_model_graph(model, shape, add_extra_nodes)
