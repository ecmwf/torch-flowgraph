# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

"""Regression test: a failed tracing attempt must not flatten the retry.

Background
----------
The forward-hook cluster builder de-duplicates hooked modules so a shared
module is not hooked twice in one ``model.apply`` pass.  This guard used to be
a persistent flag written onto the module instance
(``module._tfg_pre_hooked = True``).  Because a *fresh* ``GraphBuilder`` is
created per :func:`build_model_graph` call, the flag leaked across builds:

* callers such as ``get-model-graph.py`` trace ``Predict(model)`` with one input
  shape, and if the forward raises (e.g. a dict shape indexed as a tensor) they
  retry with a different shape on the **same** inner modules;
* after the first (failed) attempt every inner module still carried
  ``_tfg_pre_hooked = True``, so the retry's ``register_forward_pre_hook`` skipped
  them all and no inner cluster was ever pushed;
* the retry therefore succeeded but produced a single flat cluster containing
  every op node (``depth == 1``) with the whole nn.Module hierarchy discarded.

This affected a large minority of the collected model graphs (any model whose
first trace attempt failed and was retried).  The fix scopes the guard to the
builder (an ``id(module)`` set on the ``GraphBuilder``) and removes hook handles
if the forward raises, so each attempt starts from clean module state.
"""

import pytest
import torch.nn as nn

from torch_flowgraph.model.graph import ModelGraph


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.lin1 = nn.Linear(d, 2 * d)
        self.lin2 = nn.Linear(2 * d, d)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.lin2(self.act(self.lin1(self.ln(x))))


class _Inner(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(d) for _ in range(3)])
        self.head = nn.Linear(d, 4)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)


class _Wrapper(nn.Module):
    """Mimics get-model-graph.py's ``Predict(model)`` — a fresh wrapper per
    attempt around shared inner modules."""

    def __init__(self, inner):
        super().__init__()
        self.model = inner

    def forward(self, x):
        return self.model(x)


def _depth(cluster, d=0):
    children = cluster.get("children") or []
    return d if not children else max(_depth(c, d + 1) for c in children)


def test_retry_after_failed_attempt_keeps_hierarchy():
    d = 16
    inner = _Inner(d).eval()  # shared across both attempts, like the real pipeline

    # First attempt fails mid-forward *after* hooks are registered (wrong
    # feature size makes LayerNorm raise).
    with pytest.raises(Exception):
        ModelGraph.from_model(_Wrapper(inner), (1, d // 2), add_extra_nodes=True)

    # Retry on the same inner modules with a correct shape must still build the
    # full module hierarchy — not a single flat cluster.
    g = ModelGraph.from_model(_Wrapper(inner), (1, d), add_extra_nodes=True)
    clusters = g.as_dict()["clusters"]

    assert _depth(clusters) >= 4, (
        f"retry produced a flat/shallow graph (depth {_depth(clusters)}); "
        "the failed first attempt leaked its hook-dedup guard"
    )

    # And no persistent guard flags should linger on the modules.
    for m in inner.modules():
        assert not getattr(m, "_tfg_pre_hooked", False)
        assert not getattr(m, "_tfg_post_hooked", False)
