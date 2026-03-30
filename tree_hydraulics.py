"""
tree_hydraulics — exact O(N) hydraulic flow computation on rooted tree networks.

Two exact solvers for the quadratic head-loss law (dP = R * Q^2):

  1. exact_tree_flows_from_demands  — prescribed leaf/nodal outflows.
     Edge flow = sum of downstream demands. Exact, non-iterative.

  2. equivalent_resistance_dp / split_by_equivalent_resistance
     — common-downstream-pressure boundary condition.
     Exact O(N) dynamic program via equivalent-resistance recursion.

Both run in O(|V|) time and memory. For looped networks, use an
iterative method (Hardy-Cross, Newton on the loop equations, etc.).

Example
-------
>>> import numpy as np
>>> from tree_hydraulics import exact_tree_flows_from_demands
>>> children = [[1, 2], [3], [], []]
>>> edge_R = {(0, 1): 1.0, (0, 2): 2.0, (1, 3): 0.5}
>>> demands = np.array([0.0, 0.0, 0.3, 0.7])
>>> flows, pressures, subtree = exact_tree_flows_from_demands(children, edge_R, demands)
>>> flows
{(0, 1): 0.7, (0, 2): 0.3, (1, 3): 0.7}
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _postorder(children: list[list[int]], root: int = 0) -> list[int]:
    """Return nodes in post-order (children before parents)."""
    order: list[int] = []
    stack = [(root, False)]
    while stack:
        u, visited = stack.pop()
        if visited:
            order.append(u)
        else:
            stack.append((u, True))
            for v in children[u]:
                stack.append((v, False))
    return order


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def subtree_demands(
    children: list[list[int]],
    demands: ArrayLike,
    root: int = 0,
) -> np.ndarray:
    """Compute the total downstream demand (including self) at every node.

    Parameters
    ----------
    children : list[list[int]]
        Adjacency list. ``children[u]`` is the list of child nodes of *u*.
    demands : array-like, shape (n,)
        Nodal outflow demand. Leaves typically carry the entire demand;
        internal nodes have demand 0 unless there is an intermediate take-off.
    root : int
        Index of the root node (default 0).

    Returns
    -------
    sub : ndarray, shape (n,)
        ``sub[u]`` = sum of ``demands[w]`` for all *w* in the subtree rooted at *u*.
    """
    demands = np.asarray(demands, dtype=float)
    n = len(children)
    sub = demands.copy()
    for u in _postorder(children, root):
        for v in children[u]:
            sub[u] += sub[v]
    return sub


def exact_tree_flows_from_demands(
    children: list[list[int]],
    edge_R: dict[tuple[int, int], float],
    demands: ArrayLike,
    root: int = 0,
) -> tuple[dict[tuple[int, int], float], np.ndarray, np.ndarray]:
    """Exact edge flows for a tree with prescribed nodal/leaf demands.

    For a rooted tree with quadratic head-loss law dP = R * Q^2, when
    nodal outflows are prescribed the flow on each edge is exactly the
    sum of all downstream demands::

        Q_{u -> v} = sum_{w in subtree(v)} demands[w]

    This follows from conservation on a tree: every unit of demand in
    subtree *v* must cross the unique edge (u, v).

    Parameters
    ----------
    children : list[list[int]]
        Adjacency list.
    edge_R : dict[(u, v) -> float]
        Hydraulic resistance of each directed edge (parent, child).
    demands : array-like, shape (n,)
        Nodal outflow demands. Typically non-zero only at leaves.
    root : int
        Root node index (default 0).

    Returns
    -------
    flows : dict[(u, v) -> float]
        Exact flow on every edge.
    pressures : ndarray, shape (n,)
        Node pressures (root pressure set to 0).
    sub : ndarray, shape (n,)
        Subtree demand at each node (useful for verification).
    """
    demands = np.asarray(demands, dtype=float)
    sub = subtree_demands(children, demands, root)

    flows: dict[tuple[int, int], float] = {}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in children[u]:
            flows[(u, v)] = float(sub[v])
            stack.append(v)

    pressures = np.zeros(len(children), dtype=float)
    stack = [root]
    while stack:
        u = stack.pop()
        for v in children[u]:
            pressures[v] = pressures[u] - edge_R[(u, v)] * flows[(u, v)] ** 2
            stack.append(v)

    return flows, pressures, sub


def equivalent_resistance_dp(
    children: list[list[int]],
    edge_R: dict[tuple[int, int], float],
    root: int = 0,
) -> np.ndarray:
    """Equivalent resistance at each node via post-order dynamic programming.

    Under the quadratic head-loss law, when all child subtrees of a node
    share the same downstream pressure, the equivalent resistance seen
    looking into the subtree at node *u* satisfies::

        R_eq(u) = 1 / (sum_{v in ch(u)} 1/sqrt(R_{u,v} + R_eq(v)))^2

    with boundary condition R_eq(leaf) = 0.

    Parameters
    ----------
    children, edge_R, root
        Same as :func:`exact_tree_flows_from_demands`.

    Returns
    -------
    Req : ndarray, shape (n,)
        Equivalent resistance at each node.
    """
    n = len(children)
    Req = np.zeros(n, dtype=float)
    for u in _postorder(children, root):
        ch = children[u]
        if ch:
            s = sum(1.0 / np.sqrt(edge_R[(u, v)] + Req[v]) for v in ch)
            Req[u] = 1.0 / (s * s)
    return Req


def split_by_equivalent_resistance(
    children: list[list[int]],
    edge_R: dict[tuple[int, int], float],
    Req: np.ndarray,
    Q_total: float,
    root: int = 0,
) -> tuple[dict[tuple[int, int], float], np.ndarray]:
    """Propagate total inflow through the tree using the equivalent-resistance split.

    Valid when all child subtrees at each branching node share a common
    downstream pressure (the *common-pressure* boundary condition).
    The exact split weight for child *v* of node *u* is proportional to
    ``1 / sqrt(R_{u,v} + R_eq(v))``.

    Parameters
    ----------
    children, edge_R, root
        Same as :func:`exact_tree_flows_from_demands`.
    Req : ndarray
        Output of :func:`equivalent_resistance_dp`.
    Q_total : float
        Total inflow at the root.

    Returns
    -------
    flows : dict[(u, v) -> float]
        Flow on every edge.
    inflow : ndarray, shape (n,)
        Total flow arriving at each node.
    """
    n = len(children)
    inflow = np.zeros(n, dtype=float)
    inflow[root] = Q_total
    flows: dict[tuple[int, int], float] = {}

    stack = [root]
    while stack:
        u = stack.pop()
        ch = children[u]
        if not ch:
            continue
        weights = np.array([1.0 / np.sqrt(edge_R[(u, v)] + Req[v]) for v in ch])
        weights /= weights.sum()
        for v, w in zip(ch, weights):
            q = float(inflow[u] * w)
            flows[(u, v)] = q
            inflow[v] = q
            stack.append(v)

    return flows, inflow
