# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

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
# Patch 4: Tensor.item() / Tensor.__bool__() meta device support
# ---------------------------------------------------------------------------
# aten::_local_scalar_dense is invoked whenever a tensor is coerced to a
# Python scalar (.item(), __bool__(), __float__(), __int__()).  Depending on
# the PyTorch version it either has no meta kernel (NotImplementedError) or
# has one that deliberately raises RuntimeError.  Both cases abort graph
# tracing.
#
# Patch both Tensor.item() and Tensor.__bool__() at the Python class level
# to return dummy values for meta-device tensors, intercepting the call
# before it reaches the ATen dispatcher.  The exact value does not affect
# graph topology — only which branch of a conditional is traced.
def _patch_item_and_bool(_torch):
    try:
        _orig_item = _torch.Tensor.item

        def _meta_safe_item(self, _orig=_orig_item):
            if self.device.type == "meta":
                if self.dtype == _torch.bool:
                    return True
                return 1.0 if self.dtype.is_floating_point else 1
            return _orig(self)

        _torch.Tensor.item = _meta_safe_item
    except (AttributeError, TypeError):
        pass

    try:
        _orig_bool = _torch.Tensor.__bool__

        def _meta_safe_bool(self, _orig=_orig_bool):
            if self.device.type == "meta":
                return True
            return _orig(self)

        _torch.Tensor.__bool__ = _meta_safe_bool
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
                return tuple(
                    _torch.empty(n, dtype=_torch.long, device="meta")
                    for _ in range(ndim)
                )
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
        subset, edge_index, edge_attr=None, relabel_nodes=False, size=None,
        return_edge_mask=False, _orig=_orig_bipartite_subgraph,
    ):
        if edge_index.device.type == "meta":
            out_edge_index = _torch.zeros_like(edge_index)
            out_edge_attr = _torch.zeros_like(edge_attr) if edge_attr is not None else None
            if return_edge_mask:
                mask = _torch.ones(edge_index.size(1), dtype=_torch.bool, device="meta")
                return out_edge_index, out_edge_attr, mask
            return out_edge_index, out_edge_attr
        return _orig(
            subset, edge_index, edge_attr=edge_attr,
            relabel_nodes=relabel_nodes, size=size,
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
