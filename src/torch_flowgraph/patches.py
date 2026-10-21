# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

"""Monkey-patches that make torch work on meta devices.

Each patch is registered via :func:`on_import` so it fires only after the
target module is actually loaded by some other code path.  This keeps this
module side-effect-free at import time: no third-party package is imported
or required just by importing ``torch_flowgraph``.
"""

from ._post_import import on_import as _on_import


# ---------------------------------------------------------------------------
# Patch 1: torchvision FakeImplHolder.register
# ---------------------------------------------------------------------------
# Older torchvision versions call
# torch._C._dispatch_has_kernel_for_dispatch_key() on operators that have not
# been registered yet (e.g. torchvision::nms), which raises RuntimeError on
# newer PyTorch builds instead of returning False.  Patch the registration
# helper before torchvision is ever imported so that missing-operator errors
# are silently ignored rather than propagated.
def _patch_fake_impl(_fake_impl_mod):
    try:
        _orig_register = _fake_impl_mod.FakeImplHolder.register
    except AttributeError:
        return

    class _NullHandle:
        """No-op replacement for a registration handle whose op didn't exist."""

        def destroy(self):
            pass

    def _safe_register(self, fn, *args, _orig=_orig_register, _null=_NullHandle, **kwargs):
        try:
            return _orig(self, fn, *args, **kwargs)
        except RuntimeError as _e:
            if "does not exist" in str(_e):
                return _null()  # operator not loaded yet; return a no-op handle
            raise

    _fake_impl_mod.FakeImplHolder.register = _safe_register


_on_import("torch._library.fake_impl", _patch_fake_impl)


# ---------------------------------------------------------------------------
# Patch 2: torch.unique meta device support
# ---------------------------------------------------------------------------
# aten::_unique2 has no fake/meta kernel in PyTorch, so torch.unique() raises
# NotImplementedError when called on meta tensors during graph tracing.
# This patch intercepts calls on meta tensors and returns plausibly shaped
# meta tensors without attempting the real computation.
#
# Output shapes (upper-bound approximation — exact count of unique elements is
# data-dependent and unknowable at trace time):
#   output         : same shape as input (dim=None: flattened length)
#   inverse_indices: same numel as input along the traced dimension
#   counts         : same length as output
def _patch_unique(_torch):
    _orig_unique = _torch.unique

    def _meta_safe_unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
        if input.device.type == "meta":
            if dim is None:
                n = input.numel()
                out = _torch.empty(n, dtype=input.dtype, device="meta")
                inv_size = n
            else:
                n = input.shape[dim]
                shape = list(input.shape)
                out = _torch.empty(shape, dtype=input.dtype, device="meta")
                inv_size = n
            results = [out]
            if return_inverse:
                results.append(_torch.empty(inv_size, dtype=_torch.long, device="meta"))
            if return_counts:
                results.append(_torch.empty(n, dtype=_torch.long, device="meta"))
            if return_inverse or return_counts:
                return tuple(results)
            return out
        return _orig_unique(
            input,
            sorted=sorted,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dim=dim,
        )

    _torch.unique = _meta_safe_unique


_on_import("torch", _patch_unique)


# ---------------------------------------------------------------------------
# Patch 3: torch._utils._validate_loaded_sparse_tensors meta device support
# ---------------------------------------------------------------------------
# When a checkpoint is loaded onto the meta device (e.g. for graph tracing),
# any sparse COO tensors it contains land on meta as well.
# torch._utils._validate_loaded_sparse_tensors() then calls
# torch._validate_sparse_coo_tensor_args() on them, which raises
#   NotImplementedError: Cannot copy out of meta tensor; no data!
# because validation requires materialising indices/values.
# Patch the validator to skip meta-device sparse tensors silently.
def _patch_validate_sparse(_torch_utils):
    try:
        _orig_validate_sparse = _torch_utils._validate_loaded_sparse_tensors
    except AttributeError:
        return

    def _meta_safe_validate_sparse(_orig=_orig_validate_sparse):
        import torch as _t

        safe = []
        for tensor in _t._utils._sparse_tensors_to_validate:
            if getattr(tensor, "device", None) and tensor.device.type == "meta":
                continue  # skip — cannot inspect data on meta device
            safe.append(tensor)
        _t._utils._sparse_tensors_to_validate[:] = safe
        try:
            _orig()
        except NotImplementedError:
            pass
        finally:
            _t._utils._sparse_tensors_to_validate[:] = []

    _torch_utils._validate_loaded_sparse_tensors = _meta_safe_validate_sparse
    import sys as _sys

    if "torch" in _sys.modules:
        _sys.modules["torch"]._utils._validate_loaded_sparse_tensors = _meta_safe_validate_sparse


_on_import("torch._utils", _patch_validate_sparse)


# ---------------------------------------------------------------------------
# Patch 4: Tensor scalar-coercion meta device support
# ---------------------------------------------------------------------------
# aten::_local_scalar_dense is invoked whenever a tensor is coerced to a
# Python scalar.  Depending on the PyTorch version its meta kernel either
# does not exist (NotImplementedError) or deliberately raises RuntimeError
# ("Tensor.item() cannot be called on meta tensors", meta_local_scalar_dense
# in torch/_meta_registrations.py).  Both abort graph tracing.
#
# The meta kernel is registered on the ATen dispatcher and cannot be
# overridden (py_impl / Library.impl both refuse a second registration), so
# we intercept at the Python Tensor-method level instead — before the call
# ever reaches the ATen dispatcher.  Every scalar-coercion entry point routes
# through _local_scalar_dense, so all of them must be covered, not just
# .item():
#   .item()        -> _meta_safe_item
#   bool(t) / if t -> Tensor.__bool__
#   float(t)       -> Tensor.__float__
#   int(t)         -> Tensor.__int__
#   t as an index  -> Tensor.__index__   (e.g. some_list[t], x[t] with 0-d t)
#   complex(t)     -> Tensor.__complex__
# The exact value does not affect graph topology — only which branch of a
# conditional is traced.
def _patch_item_and_bool(_torch):
    def _meta_scalar(self):
        """Plausible Python scalar for a meta tensor, matching its dtype."""
        if self.dtype == _torch.bool:
            return True
        return 1.0 if self.dtype.is_floating_point else 1

    # .item(): dtype-appropriate scalar.
    try:
        _orig_item = _torch.Tensor.item

        def _meta_safe_item(self, _orig=_orig_item):
            if self.device.type == "meta":
                return _meta_scalar(self)
            return _orig(self)

        _torch.Tensor.item = _meta_safe_item
    except (AttributeError, TypeError):
        pass

    # Coercion dunders: each must return the exact Python type it promises,
    # regardless of the tensor's own dtype (float(int_tensor) is legal).
    for _dunder, _const in (
        ("__bool__", True),
        ("__float__", 1.0),
        ("__int__", 1),
        ("__index__", 1),
        ("__complex__", complex(1, 0)),
    ):
        try:
            _orig = getattr(_torch.Tensor, _dunder)
        except AttributeError:
            continue

        def _make(_orig=_orig, _const=_const):
            def _meta_safe(self):
                if self.device.type == "meta":
                    return _const
                return _orig(self)

            return _meta_safe

        try:
            setattr(_torch.Tensor, _dunder, _make())
        except (AttributeError, TypeError):
            pass


_on_import("torch", _patch_item_and_bool)


# ---------------------------------------------------------------------------
# Patch 5: torch.nonzero / Tensor.nonzero meta device support
# ---------------------------------------------------------------------------
# aten::nonzero has no meta kernel in many PyTorch versions, so any code
# that boolean-indexes a tensor (e.g. edge_index[:, edge_mask]) raises
# NotImplementedError during graph tracing.  The meta kernel for
# aten::__getitem__ calls nonzero() internally to determine the output shape.
#
# Output shape (upper-bound approximation — exact count of True elements is
# data-dependent and unknowable at trace time):
#   as_tuple=False : (numel, ndim)     ← assumes every element is nonzero
#   as_tuple=True  : tuple of ndim tensors each of shape (numel,)
def _patch_nonzero(_torch):
    _orig_nonzero = _torch.nonzero

    def _meta_safe_nonzero(input, *, out=None, as_tuple=False):
        if input.device.type == "meta":
            n = input.numel()
            ndim = input.dim()
            if as_tuple:
                return tuple(_torch.empty(n, dtype=_torch.long, device="meta") for _ in range(ndim))
            return _torch.empty((n, ndim), dtype=_torch.long, device="meta")
        return _orig_nonzero(input, out=out, as_tuple=as_tuple)

    _torch.nonzero = _meta_safe_nonzero
    _torch.Tensor.nonzero = lambda self, *, as_tuple=False, _fn=_meta_safe_nonzero: _fn(self, as_tuple=as_tuple)


_on_import("torch", _patch_nonzero)


# ---------------------------------------------------------------------------
# Patch 6: torch_geometric.utils.bipartite_subgraph meta device support
# ---------------------------------------------------------------------------
# bipartite_subgraph() calls map_index() which calls src.cpu().numpy() to
# build a pandas Series for remapping edge indices.  On meta tensors there
# is no data to copy, so this raises:
#   NotImplementedError: Cannot copy out of meta tensor; no data!
# When edge_index lives on the meta device we short-circuit the call and
# return zero tensors of the same shape — graph topology tracing does not
# need real edge values.
def _patch_torch_geometric_bipartite_subgraph(_tg_subgraph):
    import torch as _torch

    _orig_bipartite_subgraph = _tg_subgraph.bipartite_subgraph

    def _meta_safe_bipartite_subgraph(
        subset,
        edge_index,
        edge_attr=None,
        relabel_nodes=False,
        size=None,
        return_edge_mask=False,
        _orig=_orig_bipartite_subgraph,
    ):
        if edge_index.device.type == "meta":
            out_edge_index = _torch.zeros_like(edge_index)
            out_edge_attr = _torch.zeros_like(edge_attr) if edge_attr is not None else None
            if return_edge_mask:
                mask = _torch.ones(edge_index.size(1), dtype=_torch.bool, device="meta")
                return out_edge_index, out_edge_attr, mask
            return out_edge_index, out_edge_attr
        return _orig(
            subset,
            edge_index,
            edge_attr=edge_attr,
            relabel_nodes=relabel_nodes,
            size=size,
            return_edge_mask=return_edge_mask,
        )

    _tg_subgraph.bipartite_subgraph = _meta_safe_bipartite_subgraph
    try:
        import torch_geometric.utils as _tg_utils

        if hasattr(_tg_utils, "bipartite_subgraph"):
            _tg_utils.bipartite_subgraph = _meta_safe_bipartite_subgraph
    except ImportError:
        pass


_on_import("torch_geometric.utils._subgraph", _patch_torch_geometric_bipartite_subgraph)


# ---------------------------------------------------------------------------
# Patch 7: torch.get_autocast_dtype meta device support
# ---------------------------------------------------------------------------
# torch.get_autocast_dtype(device_type) looks up the autocast dtype registered
# for a device type.  "meta" is not a recognised autocast device, so the
# lookup raises:
#   RuntimeError: unsupported scalarType
# aborting any autocast-aware code path during graph tracing.  Intercept the
# "meta" device type and fall back to the CPU autocast dtype (a plausible
# default); the exact value does not affect graph topology.
def _patch_get_autocast_dtype(_torch):
    try:
        _orig_get_autocast_dtype = _torch.get_autocast_dtype
    except AttributeError:
        return

    def _meta_safe_get_autocast_dtype(device_type, _orig=_orig_get_autocast_dtype):
        if device_type == "meta":
            try:
                return _orig("cpu")  # borrow the CPU autocast dtype as a stand-in
            except RuntimeError:
                return _torch.bfloat16
        return _orig(device_type)

    _torch.get_autocast_dtype = _meta_safe_get_autocast_dtype


_on_import("torch", _patch_get_autocast_dtype)


# ---------------------------------------------------------------------------
# Patch 8: torch.allclose meta device support
# ---------------------------------------------------------------------------
# torch.allclose(a, b) returns a Python bool, which it computes in C++ by
# reducing the elementwise comparison and calling _local_scalar_dense on the
# result.  On meta tensors that raises
#   RuntimeError: Tensor.item() cannot be called on meta tensors
# (meta_local_scalar_dense) — and because the coercion happens inside the C++
# allclose kernel, it never touches Tensor.item/__bool__/__float__, so the
# Python-level Patch 4 can't intercept it.  Patch allclose itself.
#
# In practice allclose gates a sanity-check warning or assertion (e.g.
# anemoi diffusion sample(): "if not torch.allclose(sigmas_i, sigmas_ref):
# warn schedules differ").  The boolean does not affect graph topology, so on
# meta tensors we return True ("all close") — the no-warning / no-assert happy
# path that keeps tracing.
def _patch_allclose(_torch):
    _orig_allclose = _torch.allclose

    def _meta_safe_allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False, _orig=_orig_allclose):
        if getattr(input, "device", None) is not None and input.device.type == "meta":
            return True
        if getattr(other, "device", None) is not None and other.device.type == "meta":
            return True
        return _orig(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan)

    _torch.allclose = _meta_safe_allclose
    _torch.Tensor.allclose = lambda self, other, rtol=1e-05, atol=1e-08, equal_nan=False, _fn=_meta_safe_allclose: _fn(
        self, other, rtol=rtol, atol=atol, equal_nan=equal_nan
    )


_on_import("torch", _patch_allclose)


# ---------------------------------------------------------------------------
# Patch 9: accept PEP 585 builtin generics in torch.library custom-op schemas
# ---------------------------------------------------------------------------
# `@torch.library.custom_op(...)` infers the operator schema from the wrapped
# function's type annotations via `torch._library.infer_schema.infer_schema`
# (called as `torch._custom_op.impl.infer_schema`).  On torch < 2.5 that lookup
# only recognises the `typing.*` spellings (`typing.List[torch.Tensor]`, …) and
# REJECTS the equivalent PEP 585 builtin generics (`list[torch.Tensor]`,
# `dict[...]`, PEP 604 `X | None`) with
#   ValueError: infer_schema(func): Parameter <p> has unsupported type
#               list[torch.Tensor].
# Newer anemoi code uses the builtin spelling (e.g.
# `anemoi/models/distributed/primitives.py` registers `anemoi_distributed::
# alltoall` with `input_list: list[torch.Tensor]`), so importing `anemoi.models`
# aborts at registration time on a torch-2.4 env — long before torch.load even
# unpickles the model.  This blocks tracing regardless of which anemoi SHA is
# installed (the syntax is a property of the code, not a wrong version).
#
# The fix is purely cosmetic for schema inference: translate the wrapped
# function's builtin-generic annotations to their `typing` equivalents
# (`list`->`List`, `dict`->`Dict`, `tuple`->`Tuple`, `set`->`Set`,
# `X | Y`->`Union[X, Y]`) just for the duration of the infer_schema call, then
# restore the originals.  We wrap with `*args/**kwargs` so it is robust to the
# small signature differences of infer_schema across torch versions, and we only
# ever touch the FIRST positional arg (the prototype function).
def _make_infer_schema_patch():
    import types as _types
    import typing as _t

    _MAP = {list: _t.List, dict: _t.Dict, tuple: _t.Tuple, set: _t.Set, frozenset: _t.FrozenSet}

    def _translate(ann):
        origin = _t.get_origin(ann)
        args = _t.get_args(ann)
        if origin is getattr(_types, "UnionType", None) and args:  # X | Y (PEP 604)
            return _t.Union[tuple(_translate(a) for a in args)]
        if origin in _MAP and args:  # list[...] etc.
            return _MAP[origin][tuple(_translate(a) for a in args)]
        return ann

    def _wrap(orig):
        def _patched(*args, **kwargs):
            if args and callable(args[0]):
                func = args[0]
                orig_ann = getattr(func, "__annotations__", None)
                if orig_ann:
                    try:
                        new_ann = {n: _translate(t) for n, t in orig_ann.items()}
                    except Exception:
                        new_ann = None
                    if new_ann and new_ann != orig_ann:
                        func.__annotations__ = new_ann
                        try:
                            return orig(*args, **kwargs)
                        finally:
                            func.__annotations__ = orig_ann
            return orig(*args, **kwargs)

        return _patched

    return _wrap


def _patch_one_infer_schema(_mod):
    wrap = _make_infer_schema_patch()
    fn = getattr(_mod, "infer_schema", None)
    if fn is not None and not getattr(fn, "_tfg_pep585", False):
        patched = wrap(fn)
        patched._tfg_pep585 = True
        _mod.infer_schema = patched


def _patch_infer_schema(_torch):
    # Patch EAGERLY off the `torch` import (which torch_flowgraph's own import
    # guarantees is already loaded), NOT via `on_import("torch._custom_op.impl")`.
    # `torch.library.custom_op(...)` calls `torch._custom_op.impl.infer_schema`
    # SYNCHRONOUSLY from inside the decorator, i.e. during a NESTED import context;
    # `on_import` defers its callback until the outermost import returns, so the
    # patch would land AFTER the very first custom-op registration had already
    # raised (anemoi.models.distributed.primitives at import time).  Importing the
    # two modules here and patching them now avoids that race — they exist by the
    # time any anemoi code registers an op.
    import importlib

    for name in ("torch._library.infer_schema", "torch._custom_op.impl"):
        try:
            _patch_one_infer_schema(importlib.import_module(name))
        except Exception:
            pass


_on_import("torch", _patch_infer_schema)


# ---------------------------------------------------------------------------
# Patch 10: torch.amp.autocast meta device support
# ---------------------------------------------------------------------------
# `torch.amp.autocast(device_type=..., ...)` validates its device_type in
# __init__ (torch/amp/autocast_mode.py) against the set of autocast-capable
# backends.  "meta" is not one of them, so any model code that opens an
# autocast context keyed on the input's own device — e.g.
#   with torch.amp.autocast(device_type=x.device.type, enabled=self.autocast):
# (anemoi/models/layers/sparse_projector.py) — raises
#   RuntimeError: User specified an unsupported autocast device_type 'meta'
# the moment it runs on a meta tensor during graph tracing.
#
# This is the same class of gap as Patch 7 (get_autocast_dtype): autocast is a
# numerical-precision concern irrelevant to graph topology, so we borrow the
# CPU backend as a stand-in — rewrite a 'meta' device_type to 'cpu' (whether it
# arrives positionally or as a keyword) just for the __init__ call.  Every
# autocast entry point (torch.amp.autocast, torch.autocast, and the
# torch.cuda/cpu.amp.autocast wrappers) routes through this one class __init__,
# so patching it here covers them all.
def _patch_autocast(_autocast_mod):
    try:
        _autocast = _autocast_mod.autocast
        _orig_init = _autocast.__init__
    except AttributeError:
        return
    if getattr(_orig_init, "_tfg_meta_safe", False):
        return

    def _meta_safe_init(self, *args, _orig=_orig_init, **kwargs):
        if args and args[0] == "meta":
            args = ("cpu",) + tuple(args[1:])
        elif kwargs.get("device_type") == "meta":
            kwargs = dict(kwargs)
            kwargs["device_type"] = "cpu"
        return _orig(self, *args, **kwargs)

    _meta_safe_init._tfg_meta_safe = True
    _autocast.__init__ = _meta_safe_init


_on_import("torch.amp.autocast_mode", _patch_autocast)


# ---------------------------------------------------------------------------
# Patch 11: torch.sparse.mm meta device support
# ---------------------------------------------------------------------------
# torch.sparse.mm(sparse, dense) lowers to aten::addmm on the SPARSE tensor's
# backend.  There is no SparseMeta kernel for addmm/mm, so a sparse @ dense
# matmul on the meta device (e.g. anemoi/models/layers/sparse_projector.py:
#   out = torch.sparse.mm(projection_matrix, x_rhs)
# where projection_matrix is a sparse COO tensor) aborts graph tracing with
#   NotImplementedError: Could not run 'aten::addmm' with arguments from the
#   'SparseMeta' backend.
#
# The result of sparse[M, N] @ dense[N, K] is a DENSE tensor of shape [M, K]
# whose dtype is the promotion of the two operands.  On the meta device the
# values are irrelevant to graph topology, so short-circuit to a correctly
# shaped dense meta tensor.  The (M, K) shape is fully determined by the two
# inputs' shapes, which are known on meta, so the traced topology is exact.
def _patch_sparse_mm(_sparse_mod):
    import torch as _torch

    try:
        _orig_mm = _sparse_mod.mm
    except AttributeError:
        return
    if getattr(_orig_mm, "_tfg_meta_safe", False):
        return

    def _meta_safe_sparse_mm(mat1, mat2, *args, _orig=_orig_mm, **kwargs):
        d1 = getattr(mat1, "device", None)
        d2 = getattr(mat2, "device", None)
        if (d1 is not None and d1.type == "meta") or (d2 is not None and d2.type == "meta"):
            dtype = _torch.promote_types(mat1.dtype, mat2.dtype)
            return _torch.empty((mat1.shape[0], mat2.shape[1]), dtype=dtype, device="meta")
        return _orig(mat1, mat2, *args, **kwargs)

    _meta_safe_sparse_mm._tfg_meta_safe = True
    _sparse_mod.mm = _meta_safe_sparse_mm


_on_import("torch.sparse", _patch_sparse_mm)
