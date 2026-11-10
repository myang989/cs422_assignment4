"""
CS 422 Assignment 4 - Maximum Concurrent Flow and Network Design.

Part 2 - given a traffic matrix T (hose model), find the d-regular directed
topology that maximizes the concurrent flow theta. We use a mixed-integer 
linear program (MILP) to solve it.

Part 1 helpers: build the circulant topology for the uniform traffic matrix,
compute its concurrent flow (LP), and compare with the upper bounds.

Usage:
    python hw4_topology.py  #runs the tests
"""
import itertools
import random

import gurobipy as gp
import networkx as nx
import numpy as np
from gurobipy import GRB


# ---------------------------------------------------------------------------
# Traffic matrices
# ---------------------------------------------------------------------------
def uniform_tm(n, d):
    """T[i][j] = d/(n-1) for i != j (Part 1)."""
    T = np.full((n, n), d / (n - 1))
    np.fill_diagonal(T, 0.0)
    return T


def random_hose_tm(n, d, seed=0, fractional=True):
    """A random traffic matrix with every row sum and column sum equal to d
    (the hardest case of the hose model: every node uses all of its volume).
    Built as a convex combination of random derangements, scaled by d."""
    rng = random.Random(seed)
    T = np.zeros((n, n))
    k = 6 if fractional else d
    weights = [rng.random() for _ in range(k)] if fractional else [1.0] * k
    s = sum(weights)
    for w in weights:
        while True:
            p = list(range(n))
            rng.shuffle(p)
            if all(p[i] != i for i in range(n)):
                break
        for i in range(n):
            T[i, p[i]] += d * w / s
    return T


def is_hose(T, d, tol=1e-9):
    n = T.shape[0]
    return (np.all(T >= -tol) and np.all(np.abs(np.diag(T)) <= tol)
            and np.all(T.sum(axis=1) <= d + tol)
            and np.all(T.sum(axis=0) <= d + tol))


# ---------------------------------------------------------------------------
# Part 2: best topology for a given traffic matrix (MILP)
# ---------------------------------------------------------------------------
def best_topology(T, d, allow_parallel=True, time_limit=None, verbose=False):
    """
    Decision variables
        x[i,j]   integer, number of unit-capacity links i -> j  (topology)
        f[s,i,j] >= 0, flow of traffic that originates at source s on link (i,j)
        theta    >= 0, the common scaling factor of all demands
    Objective
        maximize theta
    Constraints
        (1) out-degree:   sum_j x[i,j] = d                       for every i
        (2) in-degree:    sum_i x[i,j] = d                       for every j
        (3) capacity:     sum_s f[s,i,j] <= x[i,j]               for every i != j
        (4) conservation: for every source s and node v
              sum_j f[s,v,j] - sum_j f[s,j,v] =  theta * sum_t T[s,t]   if v == s
                                              = -theta * T[s,v]          if v != s
    Returns (theta, links) where links[(i,j)] = number of links i -> j.
    """
    n = T.shape[0]
    N = range(n)
    arcs = [(i, j) for i in N for j in N if i != j]
    sources = [s for s in N if T[s].sum() > 0]

    m = gp.Model("best_topology")
    m.Params.OutputFlag = 1 if verbose else 0
    if time_limit:
        m.Params.TimeLimit = time_limit

    ub = d if allow_parallel else 1
    x = m.addVars(arcs, vtype=GRB.INTEGER, lb=0, ub=ub, name="x")
    f = m.addVars(sources, arcs, lb=0.0, name="f")
    theta = m.addVar(lb=0.0, name="theta")

    m.setObjective(theta, GRB.MAXIMIZE)

    m.addConstrs((x.sum(i, "*") == d for i in N), name="outdeg")
    m.addConstrs((x.sum("*", j) == d for j in N), name="indeg")
    m.addConstrs((gp.quicksum(f[s, i, j] for s in sources) <= x[i, j]
                  for (i, j) in arcs), name="cap")
    for s in sources:
        for v in N:
            rhs = theta * T[s].sum() if v == s else -theta * T[s, v]
            m.addConstr(f.sum(s, v, "*") - f.sum(s, "*", v) == rhs,
                        name=f"cons[{s},{v}]")

    m.optimize()
    links = {(i, j): int(round(x[i, j].X)) for (i, j) in arcs
             if x[i, j].X > 0.5}
    return theta.X, links, m


# ---------------------------------------------------------------------------
# Concurrent flow of a FIXED topology (LP) - used to check Part 1 designs
# ---------------------------------------------------------------------------
def concurrent_flow(n, links, T):
    """links: dict (i,j) -> capacity (number of unit links). Returns theta*."""
    N = range(n)
    arcs = list(links)
    sources = [s for s in N if T[s].sum() > 0]
    m = gp.Model("mcf")
    m.Params.OutputFlag = 0
    f = m.addVars(sources, arcs, lb=0.0)
    theta = m.addVar(lb=0.0)
    m.setObjective(theta, GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum(f[s, i, j] for s in sources) <= links[i, j]
                  for (i, j) in arcs))
    for s in sources:
        for v in N:
            out_v = gp.quicksum(f[s, v, j] for (a, j) in arcs if a == v)
            in_v = gp.quicksum(f[s, i, v] for (i, b) in arcs if b == v)
            rhs = theta * T[s].sum() if v == s else -theta * T[s, v]
            m.addConstr(out_v - in_v == rhs)
    m.optimize()
    return theta.X


# ---------------------------------------------------------------------------
# Part 1 helpers
# ---------------------------------------------------------------------------
def circulant(n, jumps):
    """Node i has one link to (i + j) mod n for every jump j."""
    links = {}
    for i in range(n):
        for j in jumps:
            links[(i, (i + j) % n)] = links.get((i, (i + j) % n), 0) + 1
    return links


def avg_path_length(n, links):
    G = nx.DiGraph(list(links))
    G.add_nodes_from(range(n))
    total = 0
    for _, dist in nx.all_pairs_shortest_path_length(G):
        if len(dist) < n:
            return float("inf")
        total += sum(dist.values())
    return total / (n * (n - 1))


def moore_bound(n, d):
    """Upper bound on theta for uniform traffic: (n-1) / S, where S is the
    smallest possible total distance from one node (d^k nodes at k hops)."""
    remaining, k, S = n - 1, 1, 0
    while remaining > 0:
        c = min(d ** k, remaining)
        S += k * c
        remaining -= c
        k += 1
    return (n - 1) / S


def distance_profile(n, jumps):
    """Number of nodes at 1, 2, 3, ... hops from node 0 in the circulant
    (the same for every node because the graph is vertex-symmetric)."""
    dist = nx.single_source_shortest_path_length(
        nx.DiGraph(list(circulant(n, jumps))), 0)
    if len(dist) < n:
        return None
    prof = [0] * (max(dist.values()) + 1)
    for k in dist.values():
        prof[k] += 1
    return prof[1:]


def moore_profile(n, d):
    remaining, k, prof = n - 1, 1, []
    while remaining > 0:
        prof.append(min(d ** k, remaining))
        remaining -= prof[-1]
        k += 1
    return prof


def part1_construction(n, d):
    """Part 1 construction (circulant search).
    1. Candidate topologies: circulants, node i -> (i + j) mod n, j in J, |J| = d.
    2. Keep jump sets J whose distance profile equals the Moore profile
       (d nodes at 1 hop, d^2 at 2 hops, ... as far as n allows).
    3. Among those, return the first J whose links can be loaded equally
       (concurrent-flow LP reaches the Moore bound) -> provably optimal.
    4. If none reaches the bound, return the best circulant found
       (use best_topology(uniform_tm(n, d), d) for the exact optimum)."""
    target, mp = moore_bound(n, d), moore_profile(n, d)
    T = uniform_tm(n, d)
    jump_sets = list(itertools.combinations(range(1, n), d))
    # Steps 2-3: only Moore-profile circulants can meet the bound.
    for J in jump_sets:
        if distance_profile(n, J) == mp:
            th = concurrent_flow(n, circulant(n, J), T)
            if th >= target - 1e-6:
                return th, J, True       # meets the upper bound: optimal
    # Step 4: no circulant meets the bound; return the best circulant.
    best = (0.0, None)
    for J in jump_sets:
        L = avg_path_length(n, circulant(n, J))
        if L == float("inf") or 1 / L <= best[0] + 1e-9:
            continue                     # 1/avg-path bound: cannot win
        th = concurrent_flow(n, circulant(n, J), T)
        if th > best[0] + 1e-9:
            best = (th, J)
    return best[0], best[1], False


def example_n9_d4():
    """Part 1 worked example: n = 9, d = 4, J = {1, 2, 3, 6}.
    Prints the bound, the hop profile from node 0 (C1), the jump usage (C2),
    the per-link load, the LP throughput, and the J = {1, 2, 3, 4} contrast."""
    n, d = 9, 4
    k = (n - 1) // d
    print(f"n={n}, d={d}, k={k}, upper bound k/(2k-1) = {k}/{2*k-1} = {k/(2*k-1):.4f}")
    for J in [(1, 2, 3, 6), (1, 2, 3, 4)]:
        G = nx.DiGraph(list(circulant(n, J)))
        paths = nx.single_source_shortest_path(G, 0)        # one shortest route per offset
        hops = {r: len(p) - 1 for r, p in paths.items() if r != 0}
        usage = {j: 0 for j in J}
        for r, p in paths.items():
            for a, b in zip(p, p[1:]):
                usage[(b - a) % n] += 1
        H = sum(hops.values())
        print(f"\nJ = {set(J)}")
        print(f"  (C1) 1-hop offsets: {sorted(r for r, h in hops.items() if h == 1)}, "
              f"2-hop offsets: {sorted(r for r, h in hops.items() if h == 2)}, H = {H}")
        for r in sorted(hops):
            steps = [(b - a) % n for a, b in zip(paths[r], paths[r][1:])]
            print(f"     offset {r}: " + " + ".join(map(str, steps)))
        print(f"  (C2) jump usage: {usage}")
        print(f"  heaviest link load with these routes = {max(usage.values())} x 0.5*theta "
              f"= {0.5*max(usage.values())}*theta")
        print(f"  LP theta (best routing) = {concurrent_flow(n, circulant(n, J), uniform_tm(n, d)):.4f}")


if __name__ == "__main__":
    #part 1 exampl
    example_n9_d4()

    # ---------------- Part 1: uniform traffic ----------------
    print("Part 1 (uniform TM): construction vs. upper bound")
    for n, d in [(5, 2), (7, 3), (9, 4), (11, 5), (13, 4), (16, 5), (7, 2), (10, 3)]:
        th, J, opt = part1_construction(n, d)
        ring = concurrent_flow(n, circulant(n, range(1, d + 1)), uniform_tm(n, d))
        print(f"  n={n:2d} d={d}: J={J}  theta={th:.4f}  bound={moore_bound(n, d):.4f}"
              f"  {'OPTIMAL' if opt else 'below bound'}  (ring 1..d: {ring:.4f})")

    # ---------------- Part 2: hose model, n = 8, d = 4 ----------------
    n, d = 8, 4
    print("Part 2 (hose model, n=8, d=4): MILP optimum vs. a fixed topology")
    for seed in range(8):
        T = random_hose_tm(n, d, seed=seed)
        assert is_hose(T, d)
        th, links, _ = best_topology(T, d)
        th_simple, _, _ = best_topology(T, d, allow_parallel=False)
        fixed = concurrent_flow(n, circulant(n, (1, 2, 3, 6)), T)
        print(f"  seed={seed}: theta*={th:.4f}  (no parallel links: {th_simple:.4f})"
              f"  fixed circulant: {fixed:.4f}")
    th, links, _ = best_topology(random_hose_tm(n, d, seed=3, fractional=False), d)
    print(f"  integral TM (sum of 4 permutations): theta* = {th:.4f}")
    th, links, _ = best_topology(uniform_tm(n, d), d)
    print(f"  uniform TM: theta* = {th:.4f}")
