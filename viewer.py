# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Interactive graph viewer for torch-flowgraph.

Run with::

    streamlit run viewer.py

You can pre-load a graph file by passing it as a CLI argument::

    streamlit run viewer.py -- path/to/graph.json
"""

import json
import os
import sys
import tempfile

import streamlit as st

from torch_flowgraph import GraphBase


def _load_graph(path: str) -> GraphBase:
    return GraphBase.from_json_file(path)


def _render_svg(graph: GraphBase) -> str:
    """Render *graph* to a temporary SVG and return the ``<svg>`` markup.

    Graphviz prefixes the file with an ``<?xml?>`` declaration and a
    ``<!DOCTYPE>``; those break HTML parsing when the SVG is embedded inline,
    so everything before the first ``<svg`` tag is stripped.
    """
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
        tmp = f.name
    try:
        graph.plot_graph(tmp)
        with open(tmp, encoding="utf-8") as fh:
            svg = fh.read()
        start = svg.find("<svg")
        return svg[start:] if start != -1 else svg
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> None:
    st.set_page_config(page_title="torch-flowgraph viewer", layout="wide")
    st.title("torch-flowgraph viewer")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Graph source")

        # Allow pre-loading a path passed on the command line after `--`
        default_path = sys.argv[1] if len(sys.argv) > 1 else ""

        source = st.radio("Load from", ["File path", "Upload JSON"])

        graph: GraphBase | None = None
        load_error: str | None = None

        if source == "File path":
            path = st.text_input("Path to JSON file", value=default_path)
            if path:
                if os.path.isfile(path):
                    try:
                        graph = _load_graph(path)
                    except Exception as exc:
                        load_error = str(exc)
                else:
                    load_error = f"File not found: {path}"
        else:
            uploaded = st.file_uploader("Graph JSON", type=["json"])
            if uploaded is not None:
                try:
                    graph = GraphBase.from_dict(json.load(uploaded))
                except Exception as exc:
                    load_error = str(exc)

        if load_error:
            st.error(load_error)

        if graph is None:
            st.info("Load a graph file to begin.")
            return

        # ── Graph stats ───────────────────────────────────────────────────────
        st.divider()
        st.subheader("Graph info")
        st.write(f"**Nodes:** {len(graph)}")
        st.write(f"**Edges:** {len(graph.edges)}")
        st.write(f"**Cluster depth:** {graph.depth}")

        # ── Detail level ──────────────────────────────────────────────────
        cluster_level = 0

        if graph.depth > 0:
            st.divider()
            cluster_level = st.slider(
                "Detail level",
                min_value=0,
                max_value=graph.depth,
                value=0,
                help="How much of the module hierarchy to expand. Level 0 "
                "collapses each top-level module to a single node "
                "(e.g. x → Model → y); each higher level opens one more layer, "
                "and the maximum shows every operation.",
            )

    # ── Main area ─────────────────────────────────────────────────────────────
    display_graph = graph.collapse_to_level(cluster_level)
    n = len(display_graph)
    st.caption(f"Level {cluster_level} — {n} node{'s' if n != 1 else ''}")

    if len(display_graph) == 0:
        st.warning("The graph is empty.")
        return

    try:
        with st.spinner("Loading graph..."):
            svg = _render_svg(display_graph)
    except FileNotFoundError:
        st.error(
            "Graphviz `dot` executable not found. "
            "Install Graphviz (e.g. `sudo apt install graphviz` or `brew install graphviz`) and restart."
        )
    except Exception as exc:
        st.error(f"Rendering failed: {exc}")
    else:
        st.image(svg, width="content")


if __name__ == "__main__":
    main()
