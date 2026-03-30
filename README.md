# tree-hydraulics

Released by R4RPI (resonant4.io)

Exact hydraulic flow computation on rooted tree pipe networks, with validated distinction between:

1. **prescribed leaf-demand problems** (exact edge flows are the sums of downstream demands), and
2. **common-downstream-pressure split problems** (exact branch splits follow a closed-form equivalent-resistance recursion under the quadratic head-loss law).

This repository packages a validated result for **serial-parallel tree networks** relevant to HVAC distribution, district heating branches, irrigation trees, and other acyclic hydraulic layouts.

## What problem this solves

Design and analysis tools for pipe systems often rely on iterative balancing methods such as Hardy-Cross or more general nonlinear solves. Those methods are necessary for **looped networks**, but for a **tree** they are overkill.

If the network is acyclic and the pressure drop law is quadratic,

\[
\Delta P_e = R_e Q_e^2
\]

then the flow problem simplifies sharply.

This repository documents and validates two exact statements:

### A. If terminal outflows are prescribed
For a rooted tree \(T=(V,E)\), with demand \(d_u\) at node \(u\), the exact flow on edge \((u,v)\) is simply the total downstream demand of the child subtree:

\[
Q_{u\to v} = \sum_{w \in \mathrm{subtree}(v)} d_w.
\]

This is exact, non-iterative, and independent of resistances for the purpose of determining **flows**. Resistances then determine the **pressures**.

### B. If a branch node splits into child subtrees sharing a common downstream pressure
Then the equivalent resistance recursion is exact:

\[
R_{\mathrm{eq}}(u)
= \left(\sum_{v\in \mathrm{ch}(u)} \frac{1}{\sqrt{R_{uv}+R_{\mathrm{eq}}(v)}}\right)^{-2},
\]

with leaf boundary condition

\[
R_{\mathrm{eq}}(\ell)=0
\]

for a terminal sink node \(\ell\), and flow split rule

\[
Q_{u\to v}=Q_u\,
\frac{1/\sqrt{R_{uv}+R_{\mathrm{eq}}(v)}}{\sum_{w\in \mathrm{ch}(u)}1/\sqrt{R_{uw}+R_{\mathrm{eq}}(w)}}.
\]

This gives an **exact O(|V|)** tree dynamic program for the common-pressure boundary condition.

## Why this matters

Two distinct boundary conditions arise in practice, and the right method depends on which one you have:

- the **closed-form resistance-based split** is exact when child subtrees share a common downstream pressure,
- but **not** when arbitrary leaf outflows are prescribed independently.

This library handles both cases exactly. It does **not** claim to replace nonlinear solvers for all networks. The precise scope:

- for **tree + prescribed demands**: compute flows exactly by subtree aggregation;
- for **tree + common-pressure branch split**: compute flows exactly by equivalent-resistance recursion;
- for **looped networks**, or non-quadratic / flow-dependent friction laws: use iterative methods.

## Mathematical formulation

Let \(T=(V,E)\) be a rooted tree with root \(r\). Each edge \(e=(u\to v)\in E\) has resistance \(R_e>0\). The quadratic hydraulic law is

\[
P_u - P_v = R_{uv} Q_{uv}^2,
\qquad Q_{uv} \ge 0.
\]

Mass conservation at each internal node is

\[
Q_{\mathrm{in}}(u) = d_u + \sum_{v\in \mathrm{ch}(u)} Q_{u\to v}.
\]

### Case 1: prescribed leaf or nodal demands
Define subtree demand recursively by

\[
D(u)= d_u + \sum_{v\in \mathrm{ch}(u)} D(v).
\]

Then for each child edge \((u,v)\),

\[
Q_{u\to v}=D(v).
\]

This follows immediately from conservation on a tree: every unit of demand in subtree \(v\) must cross the unique incoming edge \((u,v)\).

Once flows are known, node pressures follow from path integration:

\[
P_v = P_u - R_{uv}Q_{uv}^2.
\]

### Case 2: common-pressure child subtrees
At a branching node \(u\), if all child subtrees experience the same local pressure drop \(\Delta P_u\), then

\[
Q_{u\to v} = \sqrt{\frac{\Delta P_u}{R_{uv}+R_{\mathrm{eq}}(v)}}.
\]

Summing over children gives

\[
Q_u = \sum_{v\in \mathrm{ch}(u)} \sqrt{\frac{\Delta P_u}{R_{uv}+R_{\mathrm{eq}}(v)}}.
\]

Hence the normalized split weights are proportional to

\[
\frac{1}{\sqrt{R_{uv}+R_{\mathrm{eq}}(v)}}.
\]

The equivalent resistance seen from \(u\) is

\[
R_{\mathrm{eq}}(u)
= \left(\sum_{v\in \mathrm{ch}(u)} \frac{1}{\sqrt{R_{uv}+R_{\mathrm{eq}}(v)}}\right)^{-2}.
\]

This is the quadratic-loss analogue of parallel composition.

## Benchmark results

The provided implementation was validated against a baseline nonlinear solve using `scipy.optimize.root` on random trees.

### Validated findings

- For **prescribed leaf demands on a tree**, the exact subtree-demand construction matches the nonlinear hydraulic solve at machine precision.
- For the **equivalent-resistance recursion**, the internal split-rule identities and equal-pressure-drop identities hold at machine precision.
- The algorithm runs in **O(|V|)** time and **O(|V|)** memory.

### Important limitation validated by counterexample
The resistance-only split rule does **not** reproduce arbitrary prescribed leaf demands.

A minimal counterexample included in this repo:

- root with two leaves,
- resistances \(R_{01}=1\), \(R_{02}=4\),
- prescribed demands \((0.2, 0.8)\).

Demand-exact flows are \((0.2, 0.8)\), while resistance-only splitting gives a different allocation. So the correct interpretation is:

- **demands determine flows** in the demand-driven tree case,
- **resistances determine splits** in the common-pressure tree case.

## Benchmarks vs baseline

Gate-approved benchmark summary:

- **124× speedup** over the generic nonlinear baseline on random tree benchmarks
- agreement at **machine precision** on the validated tree cases

The repository benchmark script also records per-trial timings and residuals for random trees of sizes:

- \(n \in \{5,10,20,40\}\)
- 4 repetitions per size

The baseline solve assembles a coupled nonlinear system in edge flows and node pressures. The exact tree routines avoid global iteration entirely.

### Baseline
Generic nonlinear root-finding over:

- one hydraulic equation per edge,
- one mass-balance equation per non-root node.

### Exact methods here
- **Demand-driven tree flow:** post-order subtree aggregation + top-down pressure recovery
- **Common-pressure split:** post-order equivalent resistance + top-down split propagation

## Repository contents

- `tree_hydraulics.py`: production-ready module with docstrings, type hints, and examples
- validation helpers and benchmark entry points
- exact solvers for both tree interpretations

## Installation

### From source
```bash
git clone https://github.com/resonant4/tree-hydraulics.git
cd tree-hydraulics
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy scipy
```

### If published to PyPI
```bash
pip install tree-hydraulics
```

## Usage

### 1. Exact flows from prescribed leaf demands
```python
import numpy as np
from tree_hydraulics import exact_tree_flows_from_demands

children = [[1, 2], [3], [], []]
edge_R = {(0, 1): 1.0, (0, 2): 2.0, (1, 3): 0.5}
demands = np.array([0.0, 0.0, 0.3, 0.7], dtype=float)

flows, pressures, subtree = exact_tree_flows_from_demands(children, edge_R, demands, root=0)
print(flows)      # {(0,1): 0.7, (0,2): 0.3, (1,3): 0.7}
print(subtree)    # downstream demand totals
print(pressures)  # relative pressures with P_root = 0
```

### 2. Equivalent-resistance split under common downstream pressure
```python
from tree_hydraulics import equivalent_resistance_dp, split_by_equivalent_resistance

children = [[1, 2], [], []]
edge_R = {(0, 1): 1.0, (0, 2): 4.0}
Req = equivalent_resistance_dp(children, edge_R, root=0)
flows, inflow = split_by_equivalent_resistance(children, edge_R, Req, Q_total=1.0, root=0)

print(Req)
print(flows)  # split proportional to 1/sqrt(R)
```

### 3. Compare against nonlinear baseline
```python
import numpy as np
from tree_hydraulics import baseline_nonlinear_solve

parent = np.array([-1, 0, 0, 1], dtype=int)
children = [[1, 2], [3], [], []]
edge_R = {(0, 1): 1.0, (0, 2): 2.0, (1, 3): 0.5}
demands = np.array([0.0, 0.0, 0.3, 0.7], dtype=float)

result = baseline_nonlinear_solve(parent, children, edge_R, demands, Q_total=1.0, root_node=0)
print(result["success"], result["max_abs_residual"])
```

## API summary

### Core functions
- `exact_tree_flows_from_demands(...)`
- `equivalent_resistance_dp(...)`
- `split_by_equivalent_resistance(...)`
- `baseline_nonlinear_solve(...)`
- `run_validation()`

## When to use this

Use this package if:

- your hydraulic network is a **tree**,
- you want exact branch flows without iterative balancing,
- you need a clear distinction between **demand-driven** and **common-pressure** formulations,
- you want a compact benchmarkable reference against a nonlinear solver.

## When not to use this

Do **not** use the equivalent-resistance split as a universal rule if:

- leaf demands are externally fixed and arbitrary,
- the network contains cycles,
- the pressure-drop law is not quadratic,
- resistance depends on flow, Reynolds regime, valve position, or temperature in a way not already reduced to fixed coefficients.

In those cases, a numerical solve is generally still required.

## Reproducibility

Run the benchmark script:

```bash
python tree_hydraulics.py
```

It prints a JSON verdict including:

- random trial outcomes,
- residuals,
- exact-vs-baseline flow differences,
- split-rule verification,
- an explicit counterexample to overgeneralizing the resistance-only split.

## Positioning relative to Hardy-Cross

This is **not** a claim that Hardy-Cross is obsolete in general. It is a narrower, stronger claim:

> On a tree, for the validated boundary conditions, there is no need for iterative balancing.

That is useful in practice because many real distribution branches are operated and engineered as trees even when the larger system may contain loops elsewhere.

## Citation

If you use or discuss this release, please cite:

**R4RPI**. *tree-hydraulics: Exact hydraulic tree solvers for demand-driven and common-pressure quadratic-loss networks*.

Released by R4RPI (resonant4.io)

## About R4RPI

R4RPI, the ResonanT⁴ Research & Product Institute, publishes mathematically precise, engineering-grade research and working implementations. We commercialize selectively and publish what we can substantiate.

Released by R4RPI (resonant4.io)
