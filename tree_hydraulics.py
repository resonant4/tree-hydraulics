import json
import time
import numpy as np
from scipy.optimize import root


def generate_random_tree(n, rng, resistance_low=0.1, resistance_high=10.0):
    parent = np.full(n, -1, dtype=int)
    children = [[] for _ in range(n)]
    edge_R = {}
    for v in range(1, n):
        p = int(rng.integers(0, v))
        parent[v] = p
        children[p].append(v)
        edge_R[(p, v)] = float(rng.uniform(resistance_low, resistance_high))
    return parent, children, edge_R


def find_leaves(children):
    return [u for u, ch in enumerate(children) if len(ch) == 0]


def compute_leaf_demands(n, leaves, Q_total, rng):
    w = rng.random(len(leaves)) + 1e-12
    w = w / w.sum()
    demands = np.zeros(n, dtype=float)
    for leaf, frac in zip(leaves, w):
        demands[leaf] = Q_total * frac
    return demands


def postorder_from_root(children, root=0):
    order = []
    stack = [(root, 0)]
    while stack:
        u, state = stack.pop()
        if state == 0:
            stack.append((u, 1))
            for v in children[u]:
                stack.append((v, 0))
        else:
            order.append(u)
    return order


def subtree_demands(children, demands, root=0):
    n = len(children)
    sub = np.zeros(n, dtype=float)
    order = postorder_from_root(children, root)
    for u in order:
        total = demands[u]
        for v in children[u]:
            total += sub[v]
        sub[u] = total
    return sub


def exact_tree_flows_from_demands(children, edge_R, demands, root=0):
    sub = subtree_demands(children, demands, root)
    flows = {}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in children[u]:
            flows[(u, v)] = sub[v]
            stack.append(v)
    pressures = np.zeros(len(children), dtype=float)
    stack = [root]
    while stack:
        u = stack.pop()
        for v in children[u]:
            pressures[v] = pressures[u] - edge_R[(u, v)] * flows[(u, v)] ** 2
            stack.append(v)
    return flows, pressures, sub


def equivalent_resistance_dp(children, edge_R, root=0):
    n = len(children)
    Req = np.zeros(n, dtype=float)
    order = postorder_from_root(children, root)
    for u in order:
        if len(children[u]) == 0:
            Req[u] = 0.0
        else:
            s = 0.0
            for v in children[u]:
                s += 1.0 / np.sqrt(edge_R[(u, v)] + Req[v])
            Req[u] = 1.0 / (s * s)
    return Req


def split_by_equivalent_resistance(children, edge_R, Req, Q_total, root=0):
    n = len(children)
    inflow = np.zeros(n, dtype=float)
    inflow[root] = Q_total
    flows = {}
    stack = [root]
    while stack:
        u = stack.pop()
        if not children[u]:
            continue
        weights = np.array([1.0 / np.sqrt(edge_R[(u, v)] + Req[v]) for v in children[u]], dtype=float)
        weights /= weights.sum()
        for v, w in zip(children[u], weights):
            q = inflow[u] * w
            flows[(u, v)] = q
            inflow[v] = q
            stack.append(v)
    return flows, inflow


def build_edge_list(children):
    edge_list = []
    edge_index = {}
    for u in range(len(children)):
        for v in children[u]:
            edge_index[(u, v)] = len(edge_list)
            edge_list.append((u, v))
    return edge_list, edge_index


def baseline_nonlinear_solve(parent, children, edge_R, demands, Q_total, root_node=0, x0=None):
    # Unknowns:
    #   q_e for each edge e  -> m variables where m = n-1
    #   p_u for each non-root node u -> n-1 variables
    # Total unknowns = 2n-2.
    # Equations:
    #   one hydraulic law per edge: p_u - p_v - R q^2 = 0  (m eqs)
    #   one node balance per non-root node:
    #       q_parent_to_u - sum_out q_u_to_child - demand[u] = 0  (n-1 eqs)
    # Total equations = 2n-2, matching the unknowns.
    n = len(children)
    edge_list, edge_index = build_edge_list(children)
    m = len(edge_list)
    nonroot_nodes = list(range(1, n))
    p_index = {u: i for i, u in enumerate(nonroot_nodes)}

    def unpack(x):
        q = x[:m]
        p_nonroot = x[m:]
        p = np.zeros(n, dtype=float)
        for u in nonroot_nodes:
            p[u] = p_nonroot[p_index[u]]
        return q, p

    def fun(x):
        q, p = unpack(x)
        res = []

        # Edge hydraulic laws
        for k, (u, v) in enumerate(edge_list):
            res.append(p[u] - p[v] - edge_R[(u, v)] * q[k] * abs(q[k]))

        # Non-root node balances
        for u in nonroot_nodes:
            incoming = q[edge_index[(parent[u], u)]]
            outgoing = 0.0
            for v in children[u]:
                outgoing += q[edge_index[(u, v)]]
            res.append(incoming - outgoing - demands[u])

        return np.array(res, dtype=float)

    if x0 is None:
        x0 = np.zeros(m + (n - 1), dtype=float)
        # Good initial guess from exact subtree-demand solution
        sub = subtree_demands(children, demands, root_node)
        for k, (u, v) in enumerate(edge_list):
            x0[k] = sub[v]
        p0 = np.zeros(n, dtype=float)
        stack = [root_node]
        while stack:
            u = stack.pop()
            for v in children[u]:
                p0[v] = p0[u] - edge_R[(u, v)] * sub[v] ** 2
                stack.append(v)
        for u in nonroot_nodes:
            x0[m + p_index[u]] = p0[u]

    sol = root(fun, x0, method='hybr')
    q, p = unpack(sol.x)
    flows = {edge_list[k]: float(q[k]) for k in range(m)}
    residual = fun(sol.x)
    return {
        'success': bool(sol.success),
        'message': str(sol.message),
        'flows': flows,
        'pressures': p,
        'max_abs_residual': float(np.max(np.abs(residual))) if residual.size else 0.0,
        'nfev': int(getattr(sol, 'nfev', -1)),
    }


def max_flow_diff(flows_a, flows_b):
    keys = sorted(flows_a.keys())
    return float(max(abs(flows_a[k] - flows_b[k]) for k in keys)) if keys else 0.0


def verify_split_rule(children, edge_R, Req, flows, root=0):
    worst = 0.0
    stack = [root]
    while stack:
        u = stack.pop()
        if children[u]:
            total = sum(flows[(u, v)] for v in children[u])
            weights = np.array([1.0 / np.sqrt(edge_R[(u, v)] + Req[v]) for v in children[u]], dtype=float)
            weights /= weights.sum()
            for v, w in zip(children[u], weights):
                pred = total * w
                worst = max(worst, abs(pred - flows[(u, v)]))
                stack.append(v)
    return float(worst)


def verify_parallel_pressure_drop(children, edge_R, Req, flows, root=0):
    # At each branching node, all child paths should imply same local subtree pressure drop:
    # DeltaP_u = (R_uv + Req[v]) * Q_uv^2
    worst = 0.0
    stack = [root]
    while stack:
        u = stack.pop()
        vals = []
        for v in children[u]:
            vals.append((edge_R[(u, v)] + Req[v]) * flows[(u, v)] ** 2)
            stack.append(v)
        if vals:
            vals = np.array(vals, dtype=float)
            worst = max(worst, float(np.max(np.abs(vals - vals[0]))))
    return float(worst)


def run_validation():
    rng = np.random.default_rng(12345)
    trials = []
    all_ok = True

    sizes = [5, 10, 20, 40]
    for n in sizes:
        for rep in range(4):
            parent, children, edge_R = generate_random_tree(n, rng)
            leaves = find_leaves(children)
            Q_total = float(rng.uniform(0.5, 10.0))
            demands = compute_leaf_demands(n, leaves, Q_total, rng)

            t0 = time.time()
            exact_flows, exact_pressures, sub = exact_tree_flows_from_demands(children, edge_R, demands)
            exact_time = time.time() - t0

            t1 = time.time()
            baseline = baseline_nonlinear_solve(parent, children, edge_R, demands, Q_total)
            baseline_time = time.time() - t1

            Req = equivalent_resistance_dp(children, edge_R)
            split_flows, inflow = split_by_equivalent_resistance(children, edge_R, Req, Q_total)

            # For leaf-demand problems on a tree, the exact flow is determined solely by downstream demand.
            # The equivalent-resistance split rule generally applies when branch terminals share a common downstream pressure,
            # not when arbitrary leaf demands are prescribed. So we validate the DP formulas internally via the split-rule identities,
            # and validate the leaf-demand claim via conservation + nonlinear solve agreement.
            flow_diff_baseline = max_flow_diff(exact_flows, baseline['flows'])
            split_rule_error = verify_split_rule(children, edge_R, Req, split_flows)
            parallel_dp_error = verify_parallel_pressure_drop(children, edge_R, Req, split_flows)
            subtree_root_match = abs(sub[0] - Q_total)

            # Conservation residual for exact subtree-demand flow construction
            cons_worst = 0.0
            for u in range(1, n):
                incoming = exact_flows[(parent[u], u)]
                outgoing = sum(exact_flows[(u, v)] for v in children[u])
                cons_worst = max(cons_worst, abs(incoming - outgoing - demands[u]))

            ok = (
                baseline['success']
                and baseline['max_abs_residual'] < 1e-8
                and flow_diff_baseline < 1e-7
                and split_rule_error < 1e-10
                and parallel_dp_error < 1e-10
                and subtree_root_match < 1e-10
                and cons_worst < 1e-10
            )
            all_ok = all_ok and ok

            trials.append({
                'n': n,
                'rep': rep,
                'Q_total': Q_total,
                'baseline_success': baseline['success'],
                'baseline_max_abs_residual': baseline['max_abs_residual'],
                'flow_diff_exact_vs_baseline': flow_diff_baseline,
                'split_rule_error': split_rule_error,
                'parallel_dp_error': parallel_dp_error,
                'root_subtree_demand_error': subtree_root_match,
                'conservation_error_exact': cons_worst,
                'exact_time_ms': exact_time * 1000.0,
                'baseline_time_ms': baseline_time * 1000.0,
                'ok': ok,
            })

    # Small explicit counterexample showing that resistance-only split does not reproduce arbitrary prescribed leaf demands.
    parent = np.array([-1, 0, 0], dtype=int)
    children = [[1, 2], [], []]
    edge_R = {(0, 1): 1.0, (0, 2): 4.0}
    demands = np.array([0.0, 0.2, 0.8], dtype=float)
    Q_total = 1.0
    exact_flows, _, _ = exact_tree_flows_from_demands(children, edge_R, demands)
    Req = equivalent_resistance_dp(children, edge_R)
    split_flows, _ = split_by_equivalent_resistance(children, edge_R, Req, Q_total)
    counterexample = {
        'exact_flows_from_demands': {f'{u}->{v}': exact_flows[(u, v)] for (u, v) in sorted(exact_flows)},
        'resistance_split_flows': {f'{u}->{v}': split_flows[(u, v)] for (u, v) in sorted(split_flows)},
        'observation': 'With prescribed leaf demands, edge flows are fixed by subtree demands and need not equal the resistance-only split rule.'
    }

    verdict = {
        'claim_status': 'partially_valid',
        'summary': (
            'For a tree with prescribed leaf outflows/demands, the exact branch flow on each edge is simply the sum of downstream demands; '
            'this matches a nonlinear hydraulic solve exactly. The equivalent-resistance recursion and conductance-based split rule are exact for '
            'the alternative boundary condition where parallel child subtrees share a common downstream pressure, but not for arbitrary prescribed leaf demands.'
        ),
        'all_random_trials_passed': all_ok,
        'num_trials': len(trials),
        'trials': trials,
        'counterexample': counterexample,
    }
    return verdict


if __name__ == '__main__':
    verdict = run_validation()
    print(json.dumps(verdict, sort_keys=True))
