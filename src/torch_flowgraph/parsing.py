# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

"""Clan-based graph parsing (parse tree construction).

This module implements the **clan-based graph-parsing algorithm** described in:

    C. L. McCreary, C. L. Combs, and D. H. Gill,
    "Using Graph-Grammar Parsing for Automatic Graph Drawing",
    Proc. Graph Drawing (GD 1994), LNCS 894, Springer-Verlag, 1995.
    https://link.springer.com/chapter/10.1007/3-540-58950-3_380

    C. L. McCreary, A. Chapman, and F. Shieh,
    "Using Graph Parsing for Automatic Graph Drawing",
    IEEE Transactions on Systems, Man, and Cybernetics — Part A, vol. 28,
    no. 5, September 1998.
    https://ieeexplore.ieee.org/document/709599

Algorithm summary
-----------------
A **clan** is a maximal subset X of a directed graph G such that every vertex
outside X has the same ancestor/descendant relationship to every vertex inside
X.  Three structural clan types exist:

* **Series** (here called ``LINEAR``): nodes form a directed chain.
* **Parallel** (here called ``INDEPENDENT``): nodes have no edges between them.
* **Primitive**: the internal structure has no uniform series/parallel pattern.

The algorithm works as follows:

1. **Partition** — for each vertex v, compute its *sibling* group
   (vertices sharing the same parent set) and its *mate* group (vertices
   sharing the same child set).

2. **Find candidates** — for every (Si, Mj) sibling/mate pair, compute
   ``F = D*(Si) ∩ A*(Mj)`` (descendants-star of Si intersected with
   ancestors-star of Mj).  Split F into weakly-connected components and
   check each for legality using conditions (ii) and (iii) from the paper.
   Legal components become clan candidates; if more than one legal component
   exists, their union is also added as an ``INDEPENDENT`` candidate.

3. **Build parse tree** — sort candidates by size (largest first).  The
   largest clan becomes the root; each subsequent clan is placed as a child
   of the smallest existing tree node that contains it.

4. **Simplify** — iteratively merge linearly-joined sibling pairs into
   ``LINEAR`` nodes and flatten degenerate single-child wrappers.

Key types
---------
:class:`Kind`
    Enum labelling each clan as LEAF, LINEAR, INDEPENDENT, or PRIMITIVE.
:class:`Clan`
    A node set paired with its :class:`Kind`.
:class:`ParseTree`
    A tree of :class:`Clan` objects; the result of parsing.
:class:`Parser`
    Drives the decomposition algorithm.

Known limitations
-----------------
* Primitive clans are not recursively decomposed; the simplification passes
  only detect series structure in post-processing.
"""

import subprocess
from collections import defaultdict
from enum import Enum
from functools import cached_property
from functools import reduce
from itertools import product

import networkx as nx
import tqdm
from networkx import DiGraph

from torch_flowgraph.plotting import plot_graph
from torch_flowgraph.plotting import plot_tree


class Kind(Enum):
    """Classification of a clan within the parse tree.

    Attributes
    ----------
    UNDEF:
        Initial state; not yet classified.
    LINEAR:
        Series clan — nodes form a directed chain (no bypass edges).
        Corresponds to "series" in McCreary's taxonomy.
    INDEPENDENT:
        Parallel clan — no edges exist between the member nodes.
        Corresponds to "parallel" in McCreary's taxonomy.
    PRIMITIVE:
        The clan has a complex internal structure that is neither series
        nor parallel.
    LEAF:
        Singleton clan — contains exactly one node.
    """

    UNDEF = 0
    LINEAR = 1
    INDEPENDENT = 2
    PRIMITIVE = 3
    LEAF = 4


def levels(G):
    """Assign a topological level (generation) to every node of *G*.

    Nodes in the same topological generation receive the same level index.
    Generation 0 contains all source nodes (nodes with no predecessors).

    Parameters
    ----------
    G:
        A directed acyclic :class:`~networkx.DiGraph`.

    Returns
    -------
    dict
        Mapping ``node -> level_index`` (non-negative integer).
    """
    result = {}
    for i, generation in enumerate(nx.topological_generations(G)):
        for node in generation:
            result[node] = i
    return result


def cache(f):
    """Memoisation decorator that handles graph, list, set, and tuple arguments.

    Unlike :func:`functools.lru_cache`, this decorator converts mutable
    arguments (lists, sets, :class:`~networkx.DiGraph` objects) to hashable
    equivalents before using them as cache keys, so they can be cached safely
    without freezing the inputs.

    Parameters
    ----------
    f:
        Function to wrap.  Arguments must be composed of
        :class:`~networkx.DiGraph`, ``list``, ``set``, ``tuple``, or
        other hashable types.

    Returns
    -------
    callable
        Memoised version of *f*.
    """
    _cache = {}

    def wrapped(*args):
        def _key(x):
            """Convert *x* to a hashable cache key."""
            if isinstance(x, DiGraph):
                return id(x)
            if isinstance(x, list):
                return tuple(_key(y) for y in x)
            if isinstance(x, set):
                return frozenset(x)
            if isinstance(x, tuple):
                return tuple(_key(y) for y in x)
            return x

        key = _key(args)
        if key not in _cache:
            _cache[key] = f(*args)
        return _cache[key]

    return wrapped


class Reachability:
    """Pre-computed ancestor and descendant sets for every node in a DAG.

    Computes all reachability information in two passes (one reverse-topological
    for descendants, one topological for ancestors) at O(V + E) per pass with
    set unions.  Subsequent lookups for arbitrary node groups are simple set
    unions with no further graph traversal.
    """

    def __init__(self, G):
        self._anc = {}  # node -> frozenset of strict ancestors
        self._desc = {}  # node -> frozenset of strict descendants

        # Ancestors: process in topological order (parents before children).
        for n in nx.topological_sort(G):
            a = set()
            for p in G.predecessors(n):
                a.add(p)
                a |= self._anc[p]
            self._anc[n] = frozenset(a)

        # Descendants: process in reverse topological order.
        for n in reversed(list(nx.topological_sort(G))):
            d = set()
            for c in G.successors(n):
                d.add(c)
                d |= self._desc[c]
            self._desc[n] = frozenset(d)

    def ancestors_star(self, nodes):
        """Return *nodes* union all their ancestors (A*(nodes))."""
        result = set(nodes)
        for n in nodes:
            result |= self._anc[n]
        return result

    def descendants_star(self, nodes):
        """Return *nodes* union all their descendants (D*(nodes))."""
        result = set(nodes)
        for n in nodes:
            result |= self._desc[n]
        return result

    def ancestors(self, nodes):
        """Return strict ancestors of *nodes* (excluding *nodes*)."""
        result = set()
        for n in nodes:
            result |= self._anc[n]
        return result

    def descendants(self, nodes):
        """Return strict descendants of *nodes* (excluding *nodes*)."""
        result = set()
        for n in nodes:
            result |= self._desc[n]
        return result

    def max_ancestor_level(self, nodes, lvls):
        """Return the maximum topological level among ancestors of *nodes*."""
        a = self.ancestors_star(nodes)
        return max(lvls[n] for n in a) if a else 0

    def min_descendant_level(self, nodes, lvls):
        """Return the minimum topological level among descendants of *nodes*."""
        d = self.descendants_star(nodes)
        return min(lvls[n] for n in d) if d else 0


@cache
def ancestors_star(G, nodes):
    """Return *nodes* union all their ancestors in *G*  (A*(nodes)).

    Parameters
    ----------
    G:
        Directed graph.
    nodes:
        Collection of starting node identifiers.

    Returns
    -------
    set
        ``nodes ∪ ancestors(nodes)``.
    """
    return reduce(lambda a, b: a | b, [nx.ancestors(G, n) for n in nodes], set(nodes))


@cache
def descendants_star(G, nodes):
    """Return *nodes* union all their descendants in *G*  (D*(nodes)).

    Parameters
    ----------
    G:
        Directed graph.
    nodes:
        Collection of starting node identifiers.

    Returns
    -------
    set
        ``nodes ∪ descendants(nodes)``.
    """
    return reduce(lambda a, b: a | b, [nx.descendants(G, n) for n in nodes], set(nodes))


@cache
def ancestors(G, nodes):
    """Return all strict ancestors of *nodes* in *G* (excluding *nodes*).

    Parameters
    ----------
    G:
        Directed graph.
    nodes:
        Collection of starting node identifiers.

    Returns
    -------
    set
        ``ancestors(nodes)`` (strict, does not include *nodes* themselves).
    """
    return reduce(lambda a, b: a | b, [nx.ancestors(G, n) for n in nodes], set())


@cache
def descendants(G, nodes):
    """Return all strict descendants of *nodes* in *G* (excluding *nodes*).

    Parameters
    ----------
    G:
        Directed graph.
    nodes:
        Collection of starting node identifiers.

    Returns
    -------
    set
        ``descendants(nodes)`` (strict, does not include *nodes* themselves).
    """
    return reduce(lambda a, b: a | b, [nx.descendants(G, n) for n in nodes], set())


@cache
def sources(G, nodes=None):
    """Return the set of source nodes (in-degree zero) in *G* or its subgraph.

    When *nodes* is provided, uses set-based predecessor checking instead of
    constructing a NetworkX subgraph view, which is significantly faster for
    repeated calls with different node sets.

    Parameters
    ----------
    G:
        Directed graph.
    nodes:
        When provided, restrict the search to the subgraph induced by *nodes*.

    Returns
    -------
    set
        Nodes with no incoming edges in the (sub)graph.
    """
    if nodes is not None:
        node_set = nodes if isinstance(nodes, (set, frozenset)) else set(nodes)
        return {n for n in node_set if not any(p in node_set for p in G.predecessors(n))}
    return set(n for n in G.nodes if G.in_degree(n) == 0)


@cache
def sinks(G, nodes=None):
    """Return the set of sink nodes (out-degree zero) in *G* or its subgraph.

    When *nodes* is provided, uses set-based successor checking instead of
    constructing a NetworkX subgraph view, which is significantly faster for
    repeated calls with different node sets.

    Parameters
    ----------
    G:
        Directed graph.
    nodes:
        When provided, restrict the search to the subgraph induced by *nodes*.

    Returns
    -------
    set
        Nodes with no outgoing edges in the (sub)graph.
    """
    if nodes is not None:
        node_set = nodes if isinstance(nodes, (set, frozenset)) else set(nodes)
        return {n for n in node_set if not any(c in node_set for c in G.successors(n))}
    return set(n for n in G.nodes if G.out_degree(n) == 0)


def neighbors(G, nodes):
    """Return all neighbours (predecessors and successors) of *nodes* in *G*.

    Parameters
    ----------
    G:
        A NetworkX graph (directed or undirected).
    nodes:
        Iterable of node identifiers.

    Returns
    -------
    set
        Union of all neighbour sets.
    """
    return reduce(lambda a, b: a | set(b), [nx.all_neighbors(G, n) for n in nodes], set())


def incoming_nodes(G, subG):
    """Return nodes in *G* that have edges pointing into the sources of *subG*.

    Parameters
    ----------
    G:
        The full directed graph.
    subG:
        A subgraph whose source nodes are considered.

    Returns
    -------
    set
        Predecessor nodes of the sources of *subG* within *G*.
    """
    return reduce(lambda a, b: a | set(b), [G.predecessors(n) for n in sources(subG)], set())


def outgoing_nodes(G, subG):
    """Return nodes in *G* that are successors of the sinks of *subG*.

    Parameters
    ----------
    G:
        The full directed graph.
    subG:
        A subgraph whose sink nodes are considered.

    Returns
    -------
    set
        Successor nodes of the sinks of *subG* within *G*.
    """
    return reduce(lambda a, b: a | set(b), [G.successors(n) for n in sinks(subG)], set())


@cache
def parents(G, node):
    """Return the direct predecessors of *node* in *G*.

    Parameters
    ----------
    G:
        Directed graph.
    node:
        Node identifier.

    Returns
    -------
    list
        List of predecessor node identifiers.
    """
    return list(G.predecessors(node))


@cache
def children(G, node):
    """Return the direct successors of *node* in *G*.

    Parameters
    ----------
    G:
        Directed graph.
    node:
        Node identifier.

    Returns
    -------
    list
        List of successor node identifiers.
    """
    return list(G.successors(node))


@cache
def connected1(G, nodes):
    """Return the connected components of the undirected subgraph of *nodes*.

    Parameters
    ----------
    G:
        Directed graph.
    nodes:
        Iterable of node identifiers inducing the subgraph.

    Returns
    -------
    list[set]
        Connected components of the induced undirected subgraph.
    """
    return list(nx.connected_components(G.subgraph(nodes).to_undirected()))


@cache
def connected2(G, nodes):
    """Return the weakly connected components of the directed subgraph of *nodes*.

    Parameters
    ----------
    G:
        Directed graph.
    nodes:
        Iterable of node identifiers inducing the subgraph.

    Returns
    -------
    list[set]
        Weakly connected components of the induced directed subgraph.
    """
    return list(nx.weakly_connected_components(G.subgraph(nodes)))


# Default connectivity function used by the parsing algorithm.
# Weakly connected components are used (edges treated as undirected for
# connectivity purposes), matching the McCreary paper's description.
connected = connected2


class Clan:
    """A set of graph nodes annotated with a decomposition kind.

    A :class:`Clan` represents a subset of a directed graph's nodes that
    satisfies the clan property: every external vertex z (not in the clan) has
    the same ancestor/descendant relationship to every vertex inside the clan.

    Clans are the building blocks of the :class:`ParseTree`.  Each clan
    stores a reference to the full graph *G* and a *subgraph view* restricted
    to its own nodes.

    See also: McCreary et al. 1995, section 2 "Graph Parsing".
    """

    def __init__(self, G, nodes, kind=Kind.UNDEF):
        """Initialise a Clan.

        Parameters
        ----------
        G:
            The full directed graph of which *nodes* is a subset.
        nodes:
            ``set`` of integer node identifiers forming this clan.
        kind:
            Initial :class:`Kind` classification; defaults to
            :attr:`Kind.UNDEF`.
        """
        assert isinstance(G, nx.DiGraph)
        assert isinstance(nodes, set)
        assert all(isinstance(n, int) for n in nodes)

        self._G = G.subgraph(nodes)
        self.G = self._G
        self.nodes = nodes
        self.kind = kind

    def __len__(self):
        """Return the number of nodes in this clan."""
        return len(self.nodes)

    def copy(self):
        """Return a shallow copy of this clan (same graph, copied node set).

        Returns
        -------
        Clan
        """
        return Clan(self._G, self.nodes.copy(), self.kind)

    def subclan(self, nodes, kind=None):
        """Create a new clan for a subset of this clan's nodes.

        Parameters
        ----------
        nodes:
            ``set`` of node identifiers; must be a subset of ``self.nodes``.
        kind:
            Override :class:`Kind` for the new clan.  Inherits ``self.kind``
            when ``None``.

        Returns
        -------
        Clan
        """
        return Clan(self._G, nodes, kind if kind is not None else self.kind)

    def __repr__(self) -> str:
        return f"Clan({sorted(self.nodes)},{self.kind})"


class ParseTree:
    """A node in the parse tree produced by the McCreary clan-parsing algorithm.

    Each :class:`ParseTree` node wraps a :class:`Clan` and holds an ordered
    list of child :class:`ParseTree` nodes that partition the clan's node set.
    The tree is built top-down by :class:`Parser` and then simplified by the
    :meth:`simplify` / :meth:`find_linear` / :meth:`simplify2` passes.
    """

    def __init__(self, owner, clan, children=None):
        """Initialise a ParseTree node.

        Parameters
        ----------
        owner:
            The :class:`Parser` that created this tree.
        clan:
            The :class:`Clan` associated with this tree node.
        children:
            Initial list of child :class:`ParseTree` nodes.  Defaults to
            ``[]``.
        """
        assert isinstance(owner, Parser), type(owner)
        assert isinstance(clan, Clan), type(clan)
        self.clan = clan
        self.owner = owner
        self.children = [] if children is None else children

    @property
    def nodes(self):
        """Node set of the underlying clan."""
        return self.clan.nodes

    @property
    def clusters(self):
        """Alias for :attr:`children` (for compatibility with cluster APIs)."""
        return self.children

    def __repr__(self):
        return f"ParseTree({self.clan})"

    def __len__(self):
        """Return the size of the underlying clan."""
        return len(self.clan)

    def contains(self, s):
        """Return ``True`` if this tree's node set is a superset of *s*'s nodes.

        Parameters
        ----------
        s:
            Another :class:`Clan` or :class:`ParseTree`.

        Returns
        -------
        bool
        """
        return self.clan.nodes.issuperset(s.nodes)

    def add(self, p):
        """Append child parse-tree node *p* to this node's children list.

        Parameters
        ----------
        p:
            :class:`ParseTree` node to add as a child.
        """
        self.children.append(p)

    @cached_property
    def level(self):
        """Minimum topological level among this clan's nodes.

        Computed once from :attr:`Parser.levels` and cached.

        Returns
        -------
        int
        """
        return min(self.owner.levels[n] for n in self.clan.nodes)

    def linearly_joined(self, a, b):
        """Check whether clan *a* feeds directly into clan *b* (series composition).

        Returns ``True`` when the sinks of *a* are exactly the predecessors of
        the sources of *b* — i.e. every path leaving *a* enters *b*, and every
        path entering *b* came from *a*.  This is the defining property of a
        series (LINEAR) composition with no bypass edges.

        Parameters
        ----------
        a, b:
            ``set`` of node identifiers representing the two clans to test.

        Returns
        -------
        bool
        """
        fwd = self.owner._fwd
        bwd = self.owner._bwd

        clan_a_sinks = {n for n in a if not (fwd[n] & a)}
        clan_b_sources = {n for n in b if not (bwd[n] & b)}

        # Nodes immediately after each sink of a.
        after_a = set()
        for n in clan_a_sinks:
            after_a |= fwd[n]

        # Nodes immediately before each source of b.
        before_b = set()
        for n in clan_b_sources:
            before_b |= bwd[n]

        return clan_b_sources == after_a and clan_a_sinks == before_b

    def dump(self, indent=0):
        """Print a human-readable indented representation of this subtree.

        Parameters
        ----------
        indent:
            Number of dots to prepend at each level (increases by 1 per
            level of recursion).
        """
        print("." * indent, len(self.clan), self.clan.kind)
        for c in self.children:
            c.dump(indent + 1)

    def as_dict(self):
        """Serialise this parse-tree node to a JSON-compatible dictionary.

        Children are sorted by their topological level before serialisation
        so the representation is deterministic.

        Returns
        -------
        dict
            Dictionary with ``"nodes"``, ``"name"`` (kind), and ``"children"``
            keys.
        """
        return {
            "nodes": sorted(self.clan.nodes),
            "name": self.clan.kind.name,
            "children": ([c.as_dict() for c in sorted(self.children, key=lambda x: x.level)]),
        }

    @classmethod
    def from_dict(cls, owner, d):
        """Deserialise a parse-tree node from a dictionary.

        Parameters
        ----------
        owner:
            The owning :class:`Parser` instance.
        d:
            Dictionary as produced by :meth:`as_dict`.

        Returns
        -------
        ParseTree
        """
        clan = Clan(owner.G, set(d["nodes"]), Kind[d["kind"]])
        children = [cls.from_dict(owner, _) for _ in d["children"]]
        return cls(owner, clan, children)

    def find_linear(self, depth=0):
        """Merge adjacent linearly-joined siblings into LINEAR (series) clans.

        Performs a single pass over the children of this node, looking for
        consecutive pairs that satisfy :meth:`linearly_joined`.  Matching
        pairs are merged into a new :class:`ParseTree` node with kind
        :attr:`Kind.LINEAR`.  The pass repeats via :meth:`simplify` until
        no further merges are possible.

        Parameters
        ----------
        depth:
            Current recursion depth (informational only).

        Returns
        -------
        tuple[ParseTree, bool]
            The (possibly modified) tree node and a flag indicating whether
            any merge occurred during this pass.
        """
        changed = False

        if len(self.children) == 0:
            return self, changed

        # Recurse into children first.
        kids = self.children.copy()
        self.children = []
        for c in kids:
            c, ch = c.find_linear(depth + 1)
            self.children.append(c)
            changed = changed or ch

        if len(self.children) == 1:
            return self, changed

        assert len(self.children) > 1

        # Sort siblings by topological level so we test adjacent pairs in order.
        self.children = sorted(self.children, key=lambda x: x.level)

        i = iter(self.children)
        queue = []
        queue.append(next(i))
        queue.append(next(i))

        while True:
            try:
                c1 = queue[-2]
                c2 = queue[-1]

                # Only merge when this node is not already LINEAR (to avoid
                # collapsing the top-level root into a flat chain prematurely).
                if self.clan.kind != Kind.LINEAR and self.linearly_joined(c1.clan.nodes, c2.clan.nodes):
                    merged_clan = Clan(self.owner.G, c1.clan.nodes.union(c2.clan.nodes), Kind.LINEAR)
                    queue.pop()
                    queue.pop()
                    queue.append(ParseTree(self.owner, merged_clan, children=[c1, c2]))
                    changed = True
                    queue.append(next(i))
                    continue

                queue.append(next(i))

            except StopIteration:
                break

        self.children = queue

        return self, changed

    def simplify2(self, depth=0):
        """Flatten redundant single-child wrappers and inline nested LINEAR nodes.

        Performs two simplifications:

        1. A node with exactly one child is replaced by that child directly.
        2. A ``LINEAR`` node whose children include other ``LINEAR`` nodes has
           those children inlined (one level of nesting removed).

        Parameters
        ----------
        depth:
            Current recursion depth (informational only).

        Returns
        -------
        tuple[ParseTree, bool]
            The (possibly modified) tree node and a flag indicating whether
            any change occurred.
        """
        changed = False

        if len(self.children) == 0:
            return self, changed

        kids = self.children.copy()
        self.children = []
        for c in kids:
            c, ch = c.simplify2(depth + 1)
            self.children.append(c)
            changed = changed or ch

        if len(self.children) == 1:
            return self.children[0], True

        if self.clan.kind == Kind.LINEAR:
            kids = []
            for c in self.children:
                if c.clan.kind == Kind.LINEAR:
                    changed = True
                    kids.extend(c.children)
                else:
                    kids.append(c)
            self.children = kids

        return self, changed

    def relabel(self):
        """Relabel any remaining UNDEF clans as PRIMITIVE.

        After the linear-detection and flattening passes, any clan still
        classified as :attr:`Kind.UNDEF` is promoted to
        :attr:`Kind.PRIMITIVE` — it has a complex internal structure that
        neither the series nor parallel patterns can describe.

        Returns
        -------
        ParseTree
            *self*, mutated in-place.
        """
        for c in self.children:
            c.relabel()

        if self.clan.kind == Kind.UNDEF:
            self.clan.kind = Kind.PRIMITIVE

        return self

    def rebuild_node_lists(self):
        """Recompute each clan's node set from its children, bottom-up.

        After structural merges the node sets stored in parent clans may be
        out of date.  This method recursively rebuilds them so that every
        internal node's set equals the union of its children's sets.

        Returns
        -------
        ParseTree
            *self*, mutated in-place.
        """
        if len(self.children) == 0:
            return self

        nodes = set()
        for c in self.children:
            c.rebuild_node_lists()
            nodes = nodes.union(c.clan.nodes)
        self.clan.nodes = nodes
        return self

    def simplify(self):
        """Run all simplification passes until the tree reaches a fixed point.

        Performs the following passes in order, repeating until no pass
        reports a change:

        1. :meth:`rebuild_node_lists` — ensures clan node sets are consistent.
        2. :meth:`find_linear` — merges linearly-joined siblings into LINEAR.
        3. :meth:`simplify2` — flattens single-child wrappers and nested LINEAR.
        4. :meth:`relabel` — assigns PRIMITIVE to any remaining UNDEF clans.

        Returns
        -------
        ParseTree
            Simplified tree root (may be a different object if the root was
            replaced during simplification).
        """
        top = self
        top = top.rebuild_node_lists()

        more = True
        while more:
            more = False

            while True:
                top, changed = top.find_linear()
                if not changed:
                    break
                more = True

            while True:
                top, changed = top.simplify2()
                if not changed:
                    break
                more = True

        top = top.relabel()
        return top


class Parser:
    """Clan-based graph parser following McCreary et al. 1995.

    Given a directed acyclic graph *G*, :meth:`parse` returns a
    :class:`ParseTree` that hierarchically decomposes *G* into clans.

    The algorithm is O(n² · m) in the number of vertices n and edges m due
    to the all-pairs sibling/mate computation.  For large graphs the
    ``tqdm`` progress bar in :meth:`find_candidates` provides feedback.
    """

    def __init__(self, G):
        """Initialise a Parser for graph *G*.

        Pre-computes topological levels, full reachability (ancestors and
        descendants for every node), and plain-Python adjacency dicts so that
        the candidate-finding phase can perform O(1) lookups instead of
        repeated graph traversals through NetworkX subgraph views.

        Parameters
        ----------
        G:
            The directed acyclic graph to decompose.  Should be a
            :class:`~networkx.DiGraph`.
        """
        self.G = G
        self.levels = levels(G)
        self.reach = Reachability(G)

        # Plain-Python adjacency dicts — avoids all NetworkX subgraph-view
        # overhead (coreviews filters, SubDiGraph indirection) in the hot loop.
        self._fwd = defaultdict(set)  # node -> set of successors
        self._bwd = defaultdict(set)  # node -> set of predecessors
        for u, v in G.edges():
            self._fwd[u].add(v)
            self._bwd[v].add(u)
        # Ensure every node has an entry even if it has no edges.
        for n in G.nodes():
            self._fwd.setdefault(n, set())
            self._bwd.setdefault(n, set())

    def parse(self):
        """Decompose *G* into a hierarchical :class:`ParseTree`.

        Implements the three-phase McCreary algorithm (partition → find
        candidates → build tree) followed by the simplification passes.

        Returns
        -------
        ParseTree
            Root of the clan parse tree, simplified via
            :meth:`ParseTree.simplify`.
        """
        if len(self.G) < 2:
            return ParseTree(self, Clan(self.G, set(self.G.nodes)))

        siblings, mates = self.partition()
        candidates = self.find_candidates(siblings, mates)
        clans = self.find_clans(candidates)

        # Root := largest clan from the clan list.
        tree_nodes = []
        top = ParseTree(self, clans[0])
        tree_nodes.append(top)

        # For each remaining clan F in the clan list, from largest to smallest:
        # place F as a child of the smallest existing tree node that contains F.
        for F in clans[1:]:
            parent = sorted(
                [t for t in tree_nodes if t.contains(F)],
                key=lambda x: len(x.clan.nodes),
            )[0]
            node = ParseTree(self, F)
            tree_nodes.append(node)
            parent.add(node)

        return top.simplify()

    def partition(self):
        """Partition nodes into sibling groups and mate groups.

        For each vertex *c*:

        * ``siblings[parents(c)]`` collects all vertices that share the same
          immediate parent set (the *sibling* groups Si).
        * ``mates[children(c)]`` collects all vertices that share the same
          immediate child set (the *mate* groups Mj).

        Returns
        -------
        tuple[defaultdict, defaultdict]
            ``(siblings, mates)`` — each maps a sorted tuple of co-parents /
            co-children to the set of nodes that have those co-parents /
            co-children.
        """
        siblings = defaultdict(set)
        mates = defaultdict(set)

        for c in self.G.nodes:
            pp = parents(self.G, c)
            siblings[tuple(sorted(pp))].add(c)

            cc = children(self.G, c)
            mates[tuple(sorted(cc))].add(c)

        return siblings, mates

    def find_candidates(self, siblings, mates):
        """Build the initial candidate clan list from all (Si, Mj) pairs.

        Algorithm (McCreary et al. 1995, section 3):

        For every combination of a sibling group *Si* and a mate group *Mj*:

        1. Compute ``F = D*(Si) ∩ A*(Mj)``.
        2. If F is non-empty, split it into weakly-connected components.
        3. Compute Sk (sources) and Mk (sinks) of each component *fi* and
           check legality via :meth:`legal`.  Legal components are added to
           the candidate list.
        4. If more than one legal component exists, also add their union as
           an :attr:`Kind.INDEPENDENT` candidate.

        Individual nodes (leaves) are always added at the end.

        Parameters
        ----------
        siblings:
            Mapping produced by :meth:`partition`.
        mates:
            Mapping produced by :meth:`partition`.

        Returns
        -------
        list[Clan]
            Candidate clans sorted by size (descending).
        """
        candidates = []
        reach = self.reach
        lvls = self.levels

        # Pre-compute D*(Si) and A*(Mj) using the pre-built reachability index.
        Dsi = [(si, reach.descendants_star(si)) for si in siblings.values()]
        Ami = [(mj, reach.ancestors_star(mj)) for mj in mates.values()]

        # Pre-compute level ranges for each group to enable early pruning.
        # D*(Si) spans levels [min_level(Si), max_desc_level(Si)].
        # A*(Mj) spans levels [min_anc_level(Mj), max_level(Mj)].
        Dsi_with_levels = []
        for si, Ds in Dsi:
            min_lvl = min(lvls[n] for n in Ds)
            max_lvl = max(lvls[n] for n in Ds)
            Dsi_with_levels.append((si, Ds, min_lvl, max_lvl))

        Ami_with_levels = []
        for mj, Am in Ami:
            min_lvl = min(lvls[n] for n in Am)
            max_lvl = max(lvls[n] for n in Am)
            Ami_with_levels.append((mj, Am, min_lvl, max_lvl))

        skipped = 0
        total = len(Dsi_with_levels) * len(Ami_with_levels)

        for (si, Ds, ds_min, ds_max), (mj, Am, am_min, am_max) in tqdm.tqdm(
            product(Dsi_with_levels, Ami_with_levels),
            total=total,
            desc="Finding candidates",
            leave=False,
        ):
            # Level-range pruning: if the level ranges of D*(Si) and A*(Mj)
            # don't overlap, their intersection is guaranteed empty.
            if ds_min > am_max or am_min > ds_max:
                skipped += 1
                continue

            # F = D*(Si) ∩ A*(Mj)
            # Iterate the smaller set for faster intersection.
            if len(Ds) <= len(Am):
                f = Ds & Am
            else:
                f = Am & Ds

            if not f:
                continue

            # Split F into weakly-connected components and check each for legality.
            # Uses plain-Python BFS on pre-built adjacency dicts instead of
            # nx.weakly_connected_components which builds expensive subgraph views.
            components = self._fast_wcc(f)
            legal_components = []

            for fi in components:
                # Per the paper (section 3.1, formulas ii and iii): legality
                # must be checked using the sources Sk and sinks Mk of this
                # specific connected component Fk, not the full Si / Mj groups.
                sk = self._fast_sources(fi)
                mk = self._fast_sinks(fi)
                if self.legal(fi, sk, mk):
                    candidates.append(Clan(self.G, fi))
                    legal_components.append(fi)

            # If more than one legal component exists, their union is an
            # INDEPENDENT (parallel) clan — no edges between the components.
            if len(legal_components) > 1:
                union = set().union(*legal_components)
                candidates.append(Clan(self.G, union, Kind.INDEPENDENT))

        if skipped:
            print(f"  Level pruning skipped {skipped}/{total} pairs ({100*skipped/total:.0f}%)")

        # Every individual node is a trivial (leaf) clan.
        for n in self.G.nodes:
            candidates.append(Clan(self.G, {n}, Kind.LEAF))

        return sorted(candidates, key=lambda x: len(x), reverse=True)

    def find_clans(self, candidates):
        """Build the final clan list from the candidate list.

        Implements step 4 of the McCreary algorithm (section 3):

        For each candidate F (processed largest-first):

        a. If F is an exact duplicate of an existing clan Fi, propagate F's
           kind to Fi and discard F.
        b. If F overlaps with an existing clan Fi (partial intersection —
           neither contains the other), merge them: Fi grows to Fi ∪ F and is
           relabelled :attr:`Kind.LINEAR`.  The merged Fi may now overlap with
           other clans, so all remaining clans are re-checked against it.
        c. If F is properly contained in, or properly contains, an existing
           clan (no merge needed), leave Fi alone.
        d. If no existing clan was a duplicate of F, add F to the clan list
           (labelling it :attr:`Kind.PRIMITIVE` if still :attr:`Kind.UNDEF`).

        The invariant after processing all candidates is that no two clans in
        the list partially overlap — every pair is either disjoint or one
        properly contains the other — which is the precondition required by
        the tree-building step in :meth:`parse`.

        Parameters
        ----------
        candidates:
            List of :class:`Clan` instances as produced by
            :meth:`find_candidates`, sorted by size (descending).

        Returns
        -------
        list[Clan]
            Deduplicated, non-overlapping clans sorted by size (descending).
        """
        clans = []

        for F in candidates:  # largest to smallest
            F_nodes = F.nodes
            duplicate = False

            i = 0
            while i < len(clans):
                Fi = clans[i]
                if not F_nodes.intersection(Fi.nodes):
                    i += 1
                    continue

                Fi_in_F = F_nodes.issuperset(Fi.nodes)
                F_in_Fi = Fi.nodes.issuperset(F_nodes)

                if Fi_in_F and F_in_Fi:
                    # Exact duplicate: propagate kind if F carries one, discard F.
                    if F.kind != Kind.UNDEF:
                        Fi.kind = F.kind
                    duplicate = True
                    break

                if not Fi_in_F and not F_in_Fi:
                    # Partial overlap: merge F into Fi in-place, relabel as LINEAR.
                    # Stay at the same index (clans[i] is still Fi, now enlarged)
                    # and re-check it against the remaining candidates.
                    Fi.nodes = Fi.nodes.union(F_nodes)
                    Fi._G = self.G.subgraph(Fi.nodes)
                    Fi.G = Fi._G
                    Fi.kind = Kind.LINEAR
                    F_nodes = Fi.nodes  # F now refers to the merged clan
                    duplicate = True  # Fi already in clans, nothing to re-add
                    break

                # Proper containment (F ⊂ Fi or Fi ⊂ F): no structural change.
                i += 1

            if not duplicate:
                if F.kind == Kind.UNDEF:
                    F.kind = Kind.PRIMITIVE
                clans.append(Clan(self.G, F_nodes, F.kind))

        return sorted(clans, key=lambda x: len(x), reverse=True)

    # ------------------------------------------------------------------
    # Fast graph primitives using plain adjacency dicts
    # ------------------------------------------------------------------

    def _fast_sources(self, node_set):
        """Return nodes in *node_set* with no predecessors inside *node_set*."""
        bwd = self._bwd
        return {n for n in node_set if not (bwd[n] & node_set)}

    def _fast_sinks(self, node_set):
        """Return nodes in *node_set* with no successors inside *node_set*."""
        fwd = self._fwd
        return {n for n in node_set if not (fwd[n] & node_set)}

    def _fast_wcc(self, node_set):
        """Return weakly connected components of the subgraph induced by *node_set*.

        Uses plain-Python BFS over pre-built adjacency dicts, avoiding all
        NetworkX subgraph-view overhead.
        """
        fwd = self._fwd
        bwd = self._bwd
        remaining = set(node_set)
        components = []
        while remaining:
            seed = next(iter(remaining))
            component = set()
            queue = [seed]
            remaining.discard(seed)
            while queue:
                n = queue.pop()
                component.add(n)
                # Neighbours = successors ∪ predecessors, restricted to node_set.
                for nb in fwd[n]:
                    if nb in remaining:
                        remaining.discard(nb)
                        queue.append(nb)
                for nb in bwd[n]:
                    if nb in remaining:
                        remaining.discard(nb)
                        queue.append(nb)
            components.append(component)
        return components

    def legal(self, nodes, s, m):
        """Check whether the connected component *nodes* is a legal clan candidate.

        Applies the two legality conditions from McCreary et al. 1995,
        section 3.1, formulas (ii) and (iii).  *s* is the set of **sources**
        of component *nodes* (nodes with no predecessors within *nodes*) and
        *m* is the set of **sinks** (nodes with no successors within *nodes*):

        * **(ii)** D*(s) − (D*(m) ∪ A*(m)) = ∅
          Every node reachable from *s* must lie within the "window" between
          *s* and *m* — i.e. it must be a descendant or ancestor of *m*.
          This rules out "illegal entries" into the component from outside.

        * **(iii)** A*(m) − (D*(s) ∪ A*(s)) = ∅
          Every node that can reach *m* must lie within the window anchored
          at *s* — i.e. it must be a descendant or ancestor of *s*.
          This rules out "illegal exits" from the component to outside.

        Parameters
        ----------
        nodes:
            ``set`` of node identifiers forming the connected component Fk.
        s:
            Sources of *nodes* (Sk): nodes in *nodes* with no predecessors
            within *nodes*, computed by :meth:`_fast_sources`.
        m:
            Sinks of *nodes* (Mk): nodes in *nodes* with no successors within
            *nodes*, computed by :meth:`_fast_sinks`.

        Returns
        -------
        bool
            ``True`` if both conditions (ii) and (iii) are satisfied.
        """
        assert isinstance(nodes, set)
        reach = self.reach

        Am = reach.ancestors_star(m)
        Ds = reach.descendants_star(s)
        As = reach.ancestors_star(s)
        Dm = reach.descendants_star(m)

        # Condition (ii): D*(s) must be covered by D*(m) ∪ A*(m).
        if Ds - (Dm | Am):
            return False

        # Condition (iii): A*(m) must be covered by D*(s) ∪ A*(s).
        if Am - (Ds | As):
            return False

        return True


# ---------------------------------------------------------------------------
# Demo graphs for interactive testing (run this module as a script)
# ---------------------------------------------------------------------------

DEMO1 = """
1 > 2
2 > 3
3 > 4
4 > 17
4 > 16
4 > 13
5 > 11
5 > 9
5 > 7
6 > 11
6 > 9
6 > 7
7 > 8
8 > 13
8 > 16
8 > 17
9 > 10
10 > 17
10 > 16
10 > 13
11 > 12
12 > 13
12 > 16
12 > 17
13 > 14
14 > 15
15 > 19
15 > 25
16 > 18
17 > 18
18 > 19
18 > 25
19 > 20
19 > 22
20 > 21
21 > 24
22 > 23
23 > 24
24 > 28
24 > 34
25 > 26
26 > 27
27 > 28
27 > 34
28 > 29
28 > 31
29 > 30
30 > 33
31 > 32
32 > 33
34 > 35
"""

DEMO2 = """
3 > 6
4 > 8
4 > 5
6 > 8
5 > 10
"""

DEMO3 = """
0>3
1>3
2>3
0>4
1>4
2>4
3>5
4>5
4>6
5>7
5>8
6>8
6>7
7>9
7>10
"""

DEMO4 = """
0>1
1>2
2>3
3>4
4>5
5>6
0>4
"""

DEMO5 = """
0>1
0>2
1>3
2>4
3>5
4>6
5>7
6>7
"""

DEMO6 = """
0>1
0>2
3>2
"""

DEMO = DEMO6

if __name__ == "__main__":
    nodes = set()
    edges = []
    for line in DEMO.strip().split("\n"):
        a, b = map(int, line.split(">"))
        nodes.add(a)
        nodes.add(b)
        edges.append((a, b))

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    H = nx.transitive_reduction(G)
    nx.freeze(H)

    p = Parser(H)
    t = p.parse()

    plot_tree("tree.dot", t)
    subprocess.run(["dot", "-Tpdf", "tree.dot", "-o", "tree.pdf"])

    # Build a labelled copy of H for DOT export (H itself is frozen).
    G_display = nx.DiGraph()
    G_display.add_nodes_from((n, {"label": str(n)}) for n in H.nodes)
    G_display.add_edges_from(H.edges)
    plot_graph("graph.dot", G_display)
    subprocess.run(["dot", "-Tpdf", "graph.dot", "-o", "graph.pdf"])
