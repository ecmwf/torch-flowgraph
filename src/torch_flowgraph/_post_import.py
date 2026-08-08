# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Post-import hooks.

Provides :func:`on_import` to register a callback that fires the first time
a given module is imported (or immediately, if already loaded).  This lets a
package register patches that apply only when a third-party dependency is
actually pulled in by some other code path.
"""

import builtins
import sys

_pending: dict = {}
_done: set = set()
_orig_import = None
_depth = 0


def on_import(module_name, callback):
    """Run *callback(module)* the first time *module_name* is imported.

    If the module is already in :data:`sys.modules`, *callback* is invoked
    immediately.  Otherwise an import hook is installed that fires the
    callback exactly once, the first time the module appears in
    :data:`sys.modules` *after* the outermost ``import`` statement that
    caused it to load has fully completed.  (Submodule imports during a
    package's ``__init__.py`` execution don't fire callbacks for the parent
    package, which would otherwise see a half-initialised module.)

    Parameters
    ----------
    module_name:
        Fully qualified module name (e.g. ``"torch_geometric.utils._subgraph"``).
    callback:
        Callable taking the imported module as its single argument.
    """
    if module_name in sys.modules and module_name not in _pending:
        _done.add(module_name)
        callback(sys.modules[module_name])
        return

    _pending.setdefault(module_name, []).append(callback)
    _install_hook()


def _install_hook():
    global _orig_import
    if _orig_import is not None:
        return
    _orig_import = builtins.__import__

    def _wrapped_import(name, globals=None, locals=None, fromlist=(), level=0):
        global _depth
        _depth += 1
        try:
            result = _orig_import(name, globals, locals, fromlist, level)
        finally:
            _depth -= 1

        # Only fire callbacks once we have returned to the outermost
        # __import__ call.  Otherwise modules in sys.modules may still be
        # mid-execution (a package whose __init__.py is partially run).
        if _depth == 0 and _pending:
            for pending_name in list(_pending):
                if pending_name in sys.modules and pending_name not in _done:
                    _done.add(pending_name)
                    callbacks = _pending.pop(pending_name)
                    mod = sys.modules[pending_name]
                    for cb in callbacks:
                        try:
                            cb(mod)
                        except Exception:
                            pass
        return result

    builtins.__import__ = _wrapped_import
