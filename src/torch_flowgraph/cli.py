# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Command-line interface for torch-flowgraph.

Usage::

    torch-flowgraph validate path/to/graph.json
    python -m torch_flowgraph validate path/to/graph.json
"""

from __future__ import annotations

import argparse
import sys

from .graph import GraphBase
from .validation import validate_graph


def _cmd_validate(args) -> int:
    graph = GraphBase.from_json_file(args.path)
    ok = validate_graph(graph, verbose=not args.quiet)
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-flowgraph",
        description="Inspect and validate torch-flowgraph JSON graphs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Check a graph JSON file for inconsistencies.")
    validate.add_argument("path", help="Path to a graph JSON file.")
    validate.add_argument("-q", "--quiet", action="store_true", help="Suppress the report; only set the exit code.")
    validate.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
