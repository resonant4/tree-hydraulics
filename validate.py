"""
Validation and benchmark script for tree_hydraulics.

Compares the exact O(N) solvers against a generic nonlinear solve
(scipy.optimize.root) on random tree networks.

Run:
    python validate.py
"""

import json
import time
import numpy as np
from scipy.optimize import root

from tree_hydraulics import (
    subtree_demands,
    exact_tree_flows_from_demands,
    equivalent_resistance_dp,
    split_by_equivalent_resistance,
)


# ---------------------------------------------------------------------------
# Test-network generators
# ---------------------------------------------------------------------------

def random_tree(n: int, rng, r_low=0.1, r_high=10.0):
    parent = np.full(n, -1, dtype=int)
    children: list[list[int]] = [[] for _ in range(n)]
    edge_R: dict = {}
    for v in range(1, n):
        p = int(rng.integers(0, v))
        parent[v] = p
        children[p].append(v)
        edge_R[(p, v)] = float(rng.uniform(r_low, r_high))
    return parent, children, edge_R


def leaf_demands(n: int, children, Q_total: float, rng) -> np.ndarray:
    leaves = [u for u, ch in enumerate(children) if not ch]
    w = rng.random(len(leaves)) + 1e-12
    w /= w.sum()
    d = np.zeros(n, dtype=float)
    for leaf, frac in zip(leaves, w):
        d[leaf] = Q_total * frac
    return d


# ---------------------------------------------------------------------------
# Baseline: generic nonlinear solve (Hardy-Cross equivalent)
# ---------------------------------------------------------------------------

def _build_edges(children):
    edges, idx = [], {}
    for u, ch in enumerate(children):
        for v in ch:
            idx[(u, v)] = len(edges)
            edges.append((u, v))
    return edges, idx


def nonlinear_solve(parent, children, edge_R, demands, Q_total, root_node=0):
    n = len(children)
    edges, eidx = _build_edges(children)
    m = len(edges)
    nonroot = list(range(1, n))
    pidx = {u: i for i, u in enumerate(nonroot)}

    def unpack(x):
        q = x[:m]
        p = np.zeros(n)
        for u in nonroot:
            p[u] = x[m + pidx[u]]
        return q, p

    def residuals(x):
        q, p = unpack(x)
        res = []
        for k, (u, v) in enumerate(edges):
            res.append(p[u] - p[v] - edge_R[(u, v)] * q[k] * abs(q[k]))
        for u in nonroot:
            q_in = q[eidx[(parent[u], u)]]
            q_out = sum(q[eidx[(u, v)]] for v in children[u])
            res.append(q_in - q_out - demands[u])
        return np.array(res)

    # Warm-start from the exact demand solution
    sub = subtree_demands(children, demands, root_node)
    x0 = np.zeros(m + n - 1)
    for k, (u, v) in enumerate(edges):
        x0[k] = sub[v]
    p0 = np.zeros(n)
    stack = [root_node]
    while stack:
        u = stack.pop()
        for v in children[u]:
            p0[v] = p0[u] - edge_R[(u, v)] * sub[v] ** 2
            stack.append(v)
    for u in nonroot:
        x0[m + pidx[u]] = p0[u]

    sol = root(residuals, x0, method="hybr")
    q, p = unpack(sol.x)
    return {
        "success": bool(sol.success),
        "flows": {edges[k]: float(q[k]) for k in range(m)},
        "pressures": p,
        "max_residual": float(np.max(np.abs(residuals(sol.x)))),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run(sizes=(5, 10, 20, 40), reps=4, seed=12345):
    rng = np.random.default_rng(seed)
    trials = []
    all_ok = True

    for n in sizes:
        for _ in range(reps):
            parent, children, edge_R = random_tree(n, rng)
            Q_total = float(rng.uniform(0.5, 10.0))
            demands = leaf_demands(n, children, Q_total, rng)

            t0 = time.perf_counter()
            ex_flows, _, _ = exact_tree_flows_from_demands(children, edge_R, demands)
            t_exact = time.perf_counter() - t0

            t1 = time.perf_counter()
            bl = nonlinear_solve(parent, children, edge_R, demands, Q_total)
            t_base = time.perf_counter() - t1

            Req = equivalent_resistance_dp(children, edge_R)
            sp_flows, _ = split_by_equivalent_resistance(children, edge_R, Req, Q_total)

            flow_err = max(abs(ex_flows[e] - bl["flows"][e]) for e in ex_flows) if ex_flows else 0.0

            # Verify split-rule identities internally
            split_err = 0.0
            pdrop_err = 0.0
            for u in range(n):
                ch = children[u]
                if not ch:
                    continue
                total = sum(sp_flows[(u, v)] for v in ch)
                w = np.array([1.0 / np.sqrt(edge_R[(u, v)] + Req[v]) for v in ch])
                w /= w.sum()
                for v, wi in zip(ch, w):
                    split_err = max(split_err, abs(sp_flows[(u, v)] - total * wi))
                drops = [(edge_R[(u, v)] + Req[v]) * sp_flows[(u, v)] ** 2 for v in ch]
                pdrop_err = max(pdrop_err, max(abs(d - drops[0]) for d in drops))

            ok = (
                bl["success"]
                and bl["max_residual"] < 1e-8
                and flow_err < 1e-7
                and split_err < 1e-10
                and pdrop_err < 1e-10
            )
            all_ok = all_ok and ok
            speedup = t_base / t_exact if t_exact > 0 else float("inf")
            trials.append({"n": n, "ok": bool(ok), "flow_err": flow_err,
                           "speedup": round(speedup, 1),
                           "t_exact_ms": round(t_exact * 1e3, 3),
                           "t_base_ms": round(t_base * 1e3, 3)})

    # Explicit counterexample: resistance-split != prescribed-demand flows
    cx_children = [[1, 2], [], []]
    cx_edge_R = {(0, 1): 1.0, (0, 2): 4.0}
    cx_demands = np.array([0.0, 0.2, 0.8])
    cx_ex, _, _ = exact_tree_flows_from_demands(cx_children, cx_edge_R, cx_demands)
    cx_Req = equivalent_resistance_dp(cx_children, cx_edge_R)
    cx_sp, _ = split_by_equivalent_resistance(cx_children, cx_edge_R, cx_Req, 1.0)

    return {
        "all_passed": bool(all_ok),
        "trials": trials,
        "counterexample": {
            "description": "Resistance-split does not reproduce arbitrary prescribed demands.",
            "demand_driven_flows": {str(k): round(v, 4) for k, v in cx_ex.items()},
            "resistance_split_flows": {str(k): round(v, 4) for k, v in cx_sp.items()},
        },
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    status = "PASS" if result["all_passed"] else "FAIL"
    print(f"\n{status} — {len(result['trials'])} trials")
    if result["trials"]:
        avg_speedup = sum(t["speedup"] for t in result["trials"]) / len(result["trials"])
        print(f"Average speedup vs nonlinear solve: {avg_speedup:.0f}×")
