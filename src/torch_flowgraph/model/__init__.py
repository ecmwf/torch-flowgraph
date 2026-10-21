# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0


"""PyTorch model tracing sub-package.

Public API
----------
:class:`ModelGraph`
    A :class:`~torch_flowgraph.GraphBase` constructed by tracing a live
    ``torch.nn.Module``.  Use :meth:`~ModelGraph.from_model` to build one.
"""

from .graph import ModelGraph

__all__ = [
    "ModelGraph",
]
