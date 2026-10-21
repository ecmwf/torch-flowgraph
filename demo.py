# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import os

import torch.nn as nn

from torch_flowgraph import GraphBase
from torch_flowgraph.model import ModelGraph


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Linear(32 * 16, 10)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


# ── Trace ─────────────────────────────────────────────────────────────────────

os.makedirs("images", exist_ok=True)

graph = ModelGraph.from_model(TinyNet(), shape=(1, 3, 32, 32), add_extra_nodes=True)
print(f"graph:      {graph}  depth={graph.depth}")

graph.save_json("tinynet.json")
graph2 = GraphBase.from_json_file("tinynet.json")
print(f"reloaded:   {graph2}")

# NetworkX conversions
G_ops = graph.to_op_networkx()
G_modules = graph.to_module_networkx()
print(f"op-level networkx:     {G_ops.number_of_nodes()} nodes, {G_ops.number_of_edges()} edges")
print(f"module-level networkx: {G_modules.number_of_nodes()} nodes, {G_modules.number_of_edges()} edges")

# ── Helpers ───────────────────────────────────────────────────────────────────


def render(g, stem, *, fmt="png", clusters=None, depth=None, links=False):
    """Plot *g* to images/<stem>.<fmt>, forwarding optional kwargs."""
    kwargs = {}
    if clusters is not None:
        kwargs["clusters"] = clusters
    if depth is not None:
        kwargs["depth"] = depth
    kwargs["links"] = links
    path = f"images/{stem}.{fmt}"
    g.plot_graph(path, **kwargs)
    print(f"  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. No subgraph boxes  (pass clusters=[] to suppress them)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Flat (no subgraph boxes) ──────────────────────────────────────────")

render(graph, "flat_full", clusters=[])
render(graph.collapsed(max_depth=1), "flat_depth1", clusters=[])
render(graph.only_modules(), "flat_modules", clusters=[])

# ══════════════════════════════════════════════════════════════════════════════
# 2. Module (nn.Module) cluster subgraphs
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Module clusters ───────────────────────────────────────────────────")

render(graph, "mod_full")
render(graph.collapsed(max_depth=1), "mod_depth1")
render(graph.only_modules(), "mod_modules")

# ══════════════════════════════════════════════════════════════════════════════
# 3. With PyTorch documentation hyperlinks (SVG — links are lost in PNG)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── With PyTorch documentation links ──────────────────────────────────")

render(graph, "links_full", fmt="svg", links=True)
render(graph, "links_flat", fmt="svg", clusters=[], links=True)
