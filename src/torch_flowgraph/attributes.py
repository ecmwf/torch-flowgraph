# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

"""Graphviz DOT attribute name lists.

This module enumerates the valid attribute names for different element types
in Graphviz DOT files.  The lists are used by the plotting functions to
filter out unknown attributes when building DOT output, ensuring that only
recognised attributes are emitted.

Attribute hierarchy
-------------------
``COMMON_ATTRIBUTES``
    Attributes valid for all element types (nodes, edges, graphs, clusters).
``NODE_EDGE_ATTRIBUTES``
    Attributes valid for both nodes and edges (extends ``COMMON_ATTRIBUTES``).
``NODE_ATTRIBUTES``
    Node-specific attributes (extends ``NODE_EDGE_ATTRIBUTES``).
``EDGE_ATTRIBUTES``
    Edge-specific attributes (extends ``NODE_EDGE_ATTRIBUTES``).
``CLUSTER_GRAPH_ATTRIBUTES``
    Attributes valid for both clusters and graphs (extends
    ``COMMON_ATTRIBUTES``).
``CLUSTER_ATTRIBUTES``
    Cluster-specific attributes (extends ``CLUSTER_GRAPH_ATTRIBUTES``).
``GRAPH_ATTRIBUTES``
    Graph-specific attributes (extends ``CLUSTER_GRAPH_ATTRIBUTES``).

Reference: https://graphviz.org/doc/info/attrs.html
"""

# Attributes shared across all Graphviz element types.
COMMON_ATTRIBUTES = [
    "URL",
    "class",
    "colorscheme",
    "fontcolor",
    "fontname",
    "fontsize",
    "href",
    "id",
    "label",
    "nojustify",
    "style",
    "target",
    "tooltip",
]

# Attributes valid for both nodes and edges (but not graphs/clusters).
NODE_EDGE_ATTRIBUTES = COMMON_ATTRIBUTES + [
    "color",
    "comment",
    "fillcolor",
    "layer",
    "penwidth",
    "pos",
    "showboxes",
    "xlabel",
    "xlp",
]

# Attributes valid only for nodes.
NODE_ATTRIBUTES = NODE_EDGE_ATTRIBUTES + [
    "area",
    "distortion",
    "fixedsize",
    "gradientangle",
    "group",
    "height",
    "image",
    "imagepos",
    "imagescale",
    "labelloc",
    "margin",
    "ordering",
    "orientation",
    "peripheries",
    "pin",
    "rects",
    "regular",
    "root",
    "samplepoints",
    "shape",
    "shapefile",
    "sides",
    "skew",
    "sortv",
    "vertices",
    "width",
    "z",
]

# Attributes valid only for edges.
EDGE_ATTRIBUTES = NODE_EDGE_ATTRIBUTES + [
    "arrowhead",
    "arrowsize",
    "arrowtail",
    "constraint",
    "decorate",
    "dir",
    "edgeURL",
    "edgehref",
    "edgetarget",
    "edgetooltip",
    "headURL",
    "head_lp",
    "headclip",
    "headhref",
    "headlabel",
    "headport",
    "headtarget",
    "headtooltip",
    "labelURL",
    "labelangle",
    "labeldistance",
    "labelfloat",
    "labelfontcolor",
    "labelfontname",
    "labelfontsize",
    "labelhref",
    "labeltarget",
    "labeltooltip",
    "len",
    "lhead",
    "lp",
    "ltail",
    "minlen",
    "samehead",
    "sametail",
    "tailURL",
    "tail_lp",
    "tailclip",
    "tailhref",
    "taillabel",
    "tailport",
    "tailtarget",
    "tailtooltip",
    "weight",
]

# Attributes valid for both clusters and top-level graphs.
CLUSTER_GRAPH_ATTRIBUTES = COMMON_ATTRIBUTES + [
    "K",
    "bb",
    "bgcolor",
    "gradientangle",
    "labeljust",
    "labelloc",
    "lheight",
    "lp",
    "lwidth",
    "margin",
    "sortv",
]

# Attributes valid only for clusters (subgraphs whose name begins with
# ``cluster_``).
CLUSTER_ATTRIBUTES = CLUSTER_GRAPH_ATTRIBUTES + [
    "area",
    "cluster",
    "color",
    "fillcolor",
    "layer",
    "pencolor",
    "penwidth",
    "peripheries",
]

# Attributes valid only for the top-level graph.
GRAPH_ATTRIBUTES = CLUSTER_GRAPH_ATTRIBUTES + [
    "Damping",
    "TBbalance",
    "_background",
    "beautify",
    "center",
    "charset",
    "clusterrank",
    "comment",
    "compound",
    "concentrate",
    "defaultdist",
    "dim",
    "dimen",
    "diredgeconstraints",
    "dpi",
    "epsilon",
    "esep",
    "fontnames",
    "fontpath",
    "forcelabels",
    "imagepath",
    "inputscale",
    "label_scheme",
    "landscape",
    "layerlistsep",
    "layers",
    "layerselect",
    "layersep",
    "layout",
    "levels",
    "levelsgap",
    "linelength",
    "maxiter",
    "mclimit",
    "mindist",
    "mode",
    "model",
    "newrank",
    "nodesep",
    "normalize",
    "notranslate",
    "nslimit",
    "nslimit1",
    "oneblock",
    "ordering",
    "orientation",
    "outputorder",
    "overlap",
    "overlap_scaling",
    "overlap_shrink",
    "pack",
    "packmode",
    "pad",
    "page",
    "pagedir",
    "quadtree",
    "quantum",
    "rankdir",
    "ranksep",
    "ratio",
    "remincross",
    "repulsiveforce",
    "resolution",
    "root",
    "rotate",
    "rotation",
    "scale",
    "searchsize",
    "sep",
    "showboxes",
    "size",
    "smoothing",
    "splines",
    "start",
    "stylesheet",
    "truecolor",
    "viewport",
    "voro_margin",
    "xdotversion",
]


if __name__ == "__main__":
    # Helper script to verify that the attribute group boundaries are correct.
    common = (set(CLUSTER_ATTRIBUTES).intersection(GRAPH_ATTRIBUTES)) - set(COMMON_ATTRIBUTES)
    print(f"CLUSTER_GRAPH_ATTRIBUTES = COMMON_ATTRIBUTES + {repr(sorted(common- set(COMMON_ATTRIBUTES)))}")
    print(
        f"CLUSTER_ATTRIBUTES =  CLUSTER_GRAPH_ATTRIBUTES+{repr(sorted(set(CLUSTER_ATTRIBUTES)-common- set(COMMON_ATTRIBUTES)))}"
    )
    print(
        f"GRAPH_ATTRIBUTES = CLUSTER_GRAPH_ATTRIBUTES + {repr(sorted(set(GRAPH_ATTRIBUTES)-common- set(COMMON_ATTRIBUTES)))}"
    )
