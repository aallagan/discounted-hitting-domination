"""
Reproducibility code for
"Discounted Hitting Domination on Graphs with Submodularity, Complexity and Exact Algorithms"

This script:
1. reproduces the source-placement study on Zachary's weighted karate-club
   network and on BA(120,2) with seed 7;
2. reports the HD, degree, closeness, weighted-strength, and weighted-closeness
   comparisons used in the manuscript;
3. records the source-selection curves used in the computational comparisons;
4. certifies the discounted-hitting optimum delta and the classical domination
   number gamma by mixed-integer linear programming (SciPy/HiGHS);
5. independently checks the weighted-karate optimum and related enumeration
   claims by exhaustive search;
6. checks the distance-r recovery identity delta_{lambda,tau}=gamma_r on
   selected small graphs by exhaustive search;
7. checks the exact spider formula delta = min{D_c, D_cbar} of the spider
   theorem against exhaustive subset minimization on small spiders; and
8. checks the complete-graph formula on a grid of small instances using exact
   rational arithmetic.

Numerical equilibrium computations use double-precision floating-point
arithmetic through numpy.linalg.solve. The mixed-integer programs use
scipy.optimize.milp with the bundled HiGHS solver and require zero reported
MIP gap; each returned source set is re-evaluated by the pinned linear system.
The transfer sequences p_k, q_k, the spider formula, and the complete-graph
checks use exact rational arithmetic through fractions.Fraction.

Requirements:
    numpy
    networkx
    scipy
"""

from __future__ import annotations

import itertools
import sys
from fractions import Fraction
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, lil_matrix

TOL = 1e-12


# ---------------------------------------------------------------------------
# Equilibrium h^S via the pinned linear system
# ---------------------------------------------------------------------------

def equilibrium(G, nodes, lam, sources, weighted=False):
    """Return the equilibrium support vector h^S (float)."""
    adjacency = nx.to_numpy_array(
        G, nodelist=list(nodes),
        weight=("weight" if weighted else None), dtype=float)
    degrees = adjacency.sum(axis=1)
    if np.any(degrees <= 0):
        raise ValueError("The graph must have no isolated vertices.")
    W = adjacency / degrees[:, None]
    source_set = set(sources)
    n = len(nodes)
    if not source_set:
        return np.zeros(n, dtype=float)
    U = [i for i in range(n) if i not in source_set]
    h = np.zeros(n, dtype=float)
    if not U:
        return np.ones(n, dtype=float)
    Ui = np.array(U, dtype=int)
    Si = np.array(sorted(source_set), dtype=int)
    matrix = np.eye(len(U)) - lam * W[np.ix_(Ui, Ui)]
    rhs = lam * W[np.ix_(Ui, Si)].sum(axis=1)
    h[Si] = 1.0
    h[Ui] = np.linalg.solve(matrix, rhs)
    return h


def worst_support(G, nodes, lam, sources, weighted=False):
    sources = list(sources)
    if not sources:
        return 0.0
    return float(equilibrium(G, nodes, lam, sources, weighted).min())


def q_tau(G, nodes, lam, tau, sources, weighted=False):
    sources = list(sources)
    if not sources:
        return 0.0
    h = equilibrium(G, nodes, lam, sources, weighted)
    return float(np.minimum(h, tau).sum())


# ---------------------------------------------------------------------------
# Source-placement study
# ---------------------------------------------------------------------------

def greedy_hd(G, nodes, lam, tau, weighted):
    """Greedy submodular-cover placement, ties broken by smallest node index."""
    n = len(nodes)
    sources: list[int] = []
    remaining = list(range(n))
    curve: list[tuple[int, float]] = []
    current_q = 0.0
    while worst_support(G, nodes, lam, sources, weighted) < tau - TOL:
        best_vertex, best_gain = None, -1.0
        for v in sorted(remaining):
            gain = q_tau(G, nodes, lam, tau, sources + [v], weighted) - current_q
            if gain > best_gain + 1e-15:
                best_gain, best_vertex = gain, v
        if best_vertex is None:
            raise RuntimeError("Greedy placement failed to select a vertex.")
        sources.append(best_vertex)
        remaining.remove(best_vertex)
        current_q = q_tau(G, nodes, lam, tau, sources, weighted)
        curve.append((len(sources), worst_support(G, nodes, lam, sources, weighted)))
        if len(sources) == n:
            break
    return sources, curve


def add_until(G, nodes, lam, tau, ranking, weighted):
    """Add vertices in a fixed ranking until the floor is met.

    Returns the selected sources and the curve [(k, worst_support), ...].
    """
    sources: list[int] = []
    curve: list[tuple[int, float]] = []
    for v in ranking:
        sources.append(v)
        w = worst_support(G, nodes, lam, sources, weighted)
        curve.append((len(sources), w))
        if w >= tau - TOL:
            break
    return sources, curve


def greedy_dominating_set(G, nodes):
    index = {u: i for i, u in enumerate(nodes)}
    closed = {index[u]: {index[u]} | {index[w] for w in G.neighbors(u)} for u in nodes}
    uncovered, sources = set(range(len(nodes))), []
    while uncovered:
        v = max(range(len(nodes)), key=lambda x: (len(closed[x] & uncovered), -x))
        sources.append(v)
        uncovered -= closed[v]
    return sources


def is_classical_dominating_set(G, nodes, sources):
    covered = set(sources)
    for i in set(sources):
        covered |= {nodes.index(w) for w in G.neighbors(nodes[i])}
    return len(covered) == len(nodes)


def jaccard(a, b):
    A, B = set(a), set(b)
    return 1.0 if not A and not B else len(A & B) / len(A | B)


def certify_optimum_below_feasible_upper_bound(G, nodes, lam, tau, weighted, feasible):
    upper_bound = len(feasible)
    if worst_support(G, nodes, lam, feasible, weighted) < tau - TOL:
        raise ValueError("The supplied upper-bound source set is not feasible.")
    for k in range(1, upper_bound):
        for S in itertools.combinations(range(len(nodes)), k):
            if worst_support(G, nodes, lam, S, weighted) >= tau - TOL:
                return k
    return upper_bound


# ---------------------------------------------------------------------------
# Exact mixed-integer formulations used in the manuscript
# ---------------------------------------------------------------------------

def transition_matrix(G, nodes, weighted=False):
    adjacency = nx.to_numpy_array(
        G, nodelist=list(nodes),
        weight=("weight" if weighted else None), dtype=float)
    degrees = adjacency.sum(axis=1)
    if np.any(degrees <= 0):
        raise ValueError("The graph must have no isolated vertices.")
    return adjacency / degrees[:, None]


def exact_dhd_milp(G, nodes, lam, tau, weighted=False):
    """Solve the exact DHD MILP from the manuscript.

    Variables are ordered as (y_1,...,y_n,h_1,...,h_n), where y_i is binary.
    The returned source set is re-evaluated by the pinned linear system.
    """
    W = transition_matrix(G, nodes, weighted)
    n = len(nodes)

    objective = np.r_[np.ones(n), np.zeros(n)]
    integrality = np.r_[np.ones(n), np.zeros(n)]
    lower_bounds = np.r_[np.zeros(n), np.full(n, tau)]
    upper_bounds = np.ones(2 * n)

    A = lil_matrix((3 * n, 2 * n), dtype=float)
    for i in range(n):
        # y_i <= h_i
        A[i, i] = 1.0
        A[i, n + i] = -1.0

        # h_i - lambda * sum_j W_ij h_j >= 0
        A[n + i, n + i] = 1.0
        A[n + i, n:] -= lam * W[i, :]

        # h_i - lambda * sum_j W_ij h_j <= y_i
        A[2 * n + i, i] = -1.0
        A[2 * n + i, n + i] = 1.0
        A[2 * n + i, n:] -= lam * W[i, :]

    constraint_lb = np.r_[
        np.full(n, -np.inf),
        np.zeros(n),
        np.full(n, -np.inf),
    ]
    constraint_ub = np.r_[
        np.zeros(n),
        np.full(n, np.inf),
        np.zeros(n),
    ]

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(csr_matrix(A), constraint_lb, constraint_ub),
        options={"disp": False, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"DHD MILP failed: {result.message}")

    sources = np.flatnonzero(result.x[:n] > 0.5).tolist()
    support = worst_support(G, nodes, lam, sources, weighted)
    return int(round(result.fun)), sources, support, result


def exact_domination_milp(G, nodes):
    """Compute the classical domination number by a binary MILP."""
    n = len(nodes)
    index = {u: i for i, u in enumerate(nodes)}
    A = lil_matrix((n, n), dtype=float)
    for i, u in enumerate(nodes):
        A[i, i] = 1.0
        for v in G.neighbors(u):
            A[i, index[v]] = 1.0

    result = milp(
        np.ones(n),
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(
            csr_matrix(A), np.ones(n), np.full(n, np.inf)),
        options={"disp": False, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"domination MILP failed: {result.message}")

    sources = np.flatnonzero(result.x > 0.5).tolist()
    return int(round(result.fun)), sources, result


def karate_exhaustive_summary(G, nodes, lam, tau):
    """Verify the finite enumeration claims stated for weighted Karate."""
    W = transition_matrix(G, nodes, weighted=True)
    n = len(nodes)
    index = {u: i for i, u in enumerate(nodes)}
    closed = {
        i: {i} | {index[v] for v in G.neighbors(u)}
        for i, u in enumerate(nodes)
    }

    def support(S):
        source_set = set(S)
        U = [i for i in range(n) if i not in source_set]
        if not U:
            return 1.0
        Ui = np.array(U, dtype=int)
        Si = np.array(sorted(source_set), dtype=int)
        matrix = np.eye(len(U)) - lam * W[np.ix_(Ui, Ui)]
        rhs = lam * W[np.ix_(Ui, Si)].sum(axis=1)
        hU = np.linalg.solve(matrix, rhs)
        return float(min(1.0, hU.min()))

    def dominates(S):
        covered = set()
        for i in S:
            covered |= closed[i]
        return len(covered) == n

    best_four = -1.0
    min_dom_count = 0
    best_min_dom_support = -1.0
    for S in itertools.combinations(range(n), 4):
        w = support(S)
        best_four = max(best_four, w)
        if dominates(S):
            min_dom_count += 1
            best_min_dom_support = max(best_min_dom_support, w)

    feasible_five = 0
    feasible_five_dominating = 0
    for S in itertools.combinations(range(n), 5):
        w = support(S)
        if w >= tau - TOL:
            feasible_five += 1
            if dominates(S):
                feasible_five_dominating += 1

    return {
        "best_four_support": best_four,
        "minimum_dominating_sets": min_dom_count,
        "best_minimum_dominating_support": best_min_dom_support,
        "feasible_five_sets": feasible_five,
        "feasible_five_dominating": feasible_five_dominating,
    }


def weighted_karate_baselines(G, lam, tau):
    nodes = list(G.nodes())
    strength = {u: G.degree(u, weight="weight") for u in nodes}
    strength_order = sorted(range(len(nodes)), key=lambda i: (-strength[nodes[i]], i))
    H = G.copy()
    for _, _, data in H.edges(data=True):
        data["inverse_weight"] = 1.0 / data["weight"]
    closeness = nx.closeness_centrality(H, distance="inverse_weight")
    closeness_order = sorted(range(len(nodes)), key=lambda i: (-closeness[nodes[i]], i))
    strength_sources, _ = add_until(G, nodes, lam, tau, strength_order, True)
    closeness_sources, _ = add_until(G, nodes, lam, tau, closeness_order, True)
    return len(strength_sources), len(closeness_sources)


def placement_study(G, name, lam, tau, weighted, exhaustive_check=False):
    nodes = list(G.nodes())
    degrees = [G.degree(u) for u in nodes]

    degree_order = sorted(
        range(len(nodes)),
        key=lambda i: (-degrees[i], i),
    )

    cv = nx.closeness_centrality(G)
    closeness_order = sorted(
        range(len(nodes)),
        key=lambda i: (-cv[nodes[i]], i),
    )

    hd_sources, hd_curve = greedy_hd(G, nodes, lam, tau, weighted)
    degree_sources, degree_curve = add_until(
        G, nodes, lam, tau, degree_order, weighted
    )
    closeness_sources, closeness_curve = add_until(
        G, nodes, lam, tau, closeness_order, weighted
    )

    D = greedy_dominating_set(G, nodes)
    dworst = worst_support(G, nodes, lam, D, weighted)

    exact_delta, milp_sources, milp_worst, delta_result = exact_dhd_milp(
        G, nodes, lam, tau, weighted
    )
    exact_gamma, gamma_sources, gamma_result = exact_domination_milp(G, nodes)

    exhaustive_delta = None
    if exhaustive_check:
        exhaustive_delta = certify_optimum_below_feasible_upper_bound(
            G, nodes, lam, tau, weighted, hd_sources
        )
        assert exhaustive_delta == exact_delta

    print(
        f"\n{name}: n={len(nodes)} m={G.number_of_edges()} "
        f"Delta={max(degrees)} "
        f"walk={'weighted' if weighted else 'simple random walk'} "
        f"lambda={lam} tau={tau}"
    )
    print(f"  HD source set (selection order): {hd_sources}")
    print(f"  HD sources={len(hd_sources)}")
    print(
        f"  exact DHD optimum delta={exact_delta}; MILP source set={milp_sources}; "
        f"re-evaluated worst={milp_worst:.6f}; "
        f"MIP gap={getattr(delta_result, 'mip_gap', float('nan')):.3g}"
    )
    if exhaustive_delta is not None:
        print(f"  exhaustive Karate check: delta={exhaustive_delta}")
    print(
        f"  degree sources={len(degree_sources)}  "
        f"closeness sources={len(closeness_sources)}"
    )
    print(
        f"  exact domination number gamma={exact_gamma}; "
        f"MILP dominating set={gamma_sources}; "
        f"MIP gap={getattr(gamma_result, 'mip_gap', float('nan')):.3g}"
    )
    print(
        f"  greedy dominating set |D|={len(D)}  "
        f"worst={dworst:.6f}  meets tau={dworst >= tau - 1e-9}"
    )
    print(
        f"  Jaccard(HD,degree)={jaccard(hd_sources, degree_sources):.2f}  "
        f"Jaccard(HD,closeness)={jaccard(hd_sources, closeness_sources):.2f}"
    )
    print(
        "  HD set is a classical dominating set? "
        f"{is_classical_dominating_set(G, nodes, hd_sources)}"
    )
    print(f"  HD curve: {[(k, round(v, 6)) for k, v in hd_curve]}")
    print(f"  degree curve: {[(k, round(v, 6)) for k, v in degree_curve]}")
    print(f"  closeness curve: {[(k, round(v, 6)) for k, v in closeness_curve]}")

    return {
        "hd_sources": hd_sources,
        "hd_curve": hd_curve,
        "degree_sources": degree_sources,
        "degree_curve": degree_curve,
        "closeness_sources": closeness_sources,
        "closeness_curve": closeness_curve,
        "delta": exact_delta,
        "gamma": exact_gamma,
        "milp_sources": milp_sources,
        "milp_worst": milp_worst,
    }


# ---------------------------------------------------------------------------
# Distance-r recovery
# ---------------------------------------------------------------------------

def gamma_r_bruteforce(G, r):
    dists = dict(nx.all_pairs_shortest_path_length(G))
    V = list(G.nodes())
    for k in range(len(V) + 1):
        for S in itertools.combinations(V, k):
            if all(any(dists[v].get(s, 10**9) <= r for s in S) for v in V):
                return k
    raise RuntimeError("no distance-r dominating set")


def delta_bruteforce(G, lam, tau):
    nodes = list(G.nodes())
    for k in range(1, len(nodes) + 1):
        for S in itertools.combinations(range(len(nodes)), k):
            if worst_support(G, nodes, lam, S) >= tau - TOL:
                return k
    raise RuntimeError("no feasible source set")


def check_distance_recovery_examples():
    examples = [("P7", nx.path_graph(7), 1), ("C8", nx.cycle_graph(8), 1),
                ("Petersen", nx.petersen_graph(), 1), ("P9", nx.path_graph(9), 2),
                ("3x3 grid", nx.grid_2d_graph(3, 3), 1)]
    print("\nDistance-r recovery checks")
    all_match = True
    for name, G, r in examples:
        G = nx.convert_node_labels_to_integers(G)
        Delta = max(dict(G.degree()).values())
        lam = 0.9 / (Delta**r + 0.5)
        lower, upper = lam ** (r + 1), (lam / Delta) ** r
        tau = (lower + upper) / 2.0
        g, d = gamma_r_bruteforce(G, r), delta_bruteforce(G, lam, tau)
        all_match &= (g == d)
        print(f"  {name}: Delta={Delta}, r={r}, lambda={lam:.6f}, "
              f"tau in ({lower:.6f},{upper:.6f}], gamma_r={g}, delta={d}, match={g == d}")
    assert all_match, "a distance-r recovery check failed"
    print("  all recovery checks passed:", all_match)


# ---------------------------------------------------------------------------
# Exact spider formula versus brute force
# ---------------------------------------------------------------------------

def transfer_sequences(kmax, lam):
    """Exact p_k, q_k for k=0..kmax via y_{k+1}=(2/lam)y_k - y_{k-1}."""
    two = 2 / lam
    p, q = [Fraction(0), Fraction(1)], [Fraction(1), 1 / lam]
    for _ in range(1, kmax):
        p.append(two * p[-1] - p[-2])
        q.append(two * q[-1] - q[-2])
    return p, q


def scales_LB(lam, tau, p, q):
    B = 0
    while B + 1 < len(q) and 1 / q[B + 1] >= tau:
        B += 1

    def psi(m):
        return Fraction(1) if m == 1 else min((p[m - j] + p[j]) / p[m] for j in range(1, m))

    L = 1
    while L + 1 < len(p) and psi(L + 1) >= tau:
        L += 1
    return L, B


def continuation_R(m, L, B):
    return (max(0, m - B) + L - 1) // L


def theta_a(a, tau, p):
    if a == 1:
        return Fraction(0)
    return max([Fraction(0)] + [(tau * p[a] - p[j]) / p[a - j] for j in range(1, a)])


def spider_formula_exact(legs, lam, tau):
    """delta_{lambda,tau}(Sp(legs)) via min{D_c, D_cbar} in exact arithmetic."""
    lam, tau, d = Fraction(lam), Fraction(tau), len(legs)
    p, q = transfer_sequences(sum(legs) + max(legs) + 5, lam)
    L, B = scales_LB(lam, tau, p, q)
    Dc = 1 + sum(continuation_R(l, L, B) for l in legs)
    options = []
    for l in legs:
        opts = [("F", p[a - 1] / p[a], Fraction(1) / p[a], theta_a(a, tau, p),
                 1 + continuation_R(l - a, L, B)) for a in range(1, min(L, l) + 1)]
        if l <= B:
            opts.append(("N", q[l - 1] / q[l], Fraction(0), tau * q[l], 0))
        options.append(opts)
    Dcbar = None
    for combo in itertools.product(*options):
        A = sum(c[1] for c in combo)
        C = sum(c[2] for c in combo)
        Theta = max([tau] + [c[3] for c in combo])
        K = sum(c[4] for c in combo)
        if lam * C / (d - lam * A) >= Theta:
            Dcbar = K if Dcbar is None else min(Dcbar, K)
    return Dc if Dcbar is None else min(Dc, Dcbar)


def spider_graph(legs):
    G = nx.Graph()
    center = "c"
    G.add_node(center)
    for i, l in enumerate(legs):
        prev = center
        for j in range(1, l + 1):
            v = (i, j)
            G.add_edge(prev, v)
            prev = v
    return G


def spider_delta_bruteforce(legs, lam, tau):
    G = spider_graph(legs)
    nodes = list(G.nodes())
    for k in range(1, len(nodes) + 1):
        for S in itertools.combinations(range(len(nodes)), k):
            if worst_support(G, nodes, float(lam), list(S)) >= float(tau) - 1e-9:
                return k
    return None


def check_spider_formula():
    spiders = [(2, 2, 2), (5, 6), (1, 1, 1), (3, 3), (2, 3, 4), (1, 2, 3),
               (4,), (2, 2), (3, 3, 3), (1, 1, 1, 1), (2, 4), (1, 2, 2),
               (5,), (2, 2, 2, 2)]
    pairs = [(Fraction(4, 5), Fraction(1, 2)), (Fraction(17, 20), Fraction(11, 20)),
             (Fraction(1, 2), Fraction(1, 4)), (Fraction(7, 10), Fraction(2, 5))]
    print("\nExact spider formula versus brute force")
    all_match = True
    for lam, tau in pairs:
        for legs in spiders:
            f = spider_formula_exact(list(legs), lam, tau)
            b = spider_delta_bruteforce(list(legs), lam, tau)
            if f != b:
                all_match = False
                print(f"  MISMATCH: legs={legs}, lambda={lam}, tau={tau}, formula={f}, brute={b}")
    assert all_match, "a spider formula check failed"
    print("  all spider formula checks passed:", all_match)


# ---------------------------------------------------------------------------
# Exact rational complete-graph checks
# ---------------------------------------------------------------------------

def ceil_fraction(x):
    return -(-x.numerator // x.denominator)


def complete_graph_worst_exact(n, lam, q):
    return Fraction(1) if q >= n else lam * q / ((n - 1) - lam * (n - q - 1))


def complete_graph_formula_exact(n, lam, tau):
    if tau == 1:
        return n
    threshold = tau * (n - 1) * (1 - lam) / (lam * (1 - tau))
    return min(n, ceil_fraction(threshold))


def complete_graph_scan_exact(n, lam, tau):
    for q in range(1, n + 1):
        if complete_graph_worst_exact(n, lam, q) >= tau:
            return q
    return n


def check_complete_graph_formula_exact():
    pairs = [(Fraction(1, 2), Fraction(1, 4)), (Fraction(3, 5), Fraction(1, 3)),
             (Fraction(4, 5), Fraction(1, 2)), (Fraction(7, 8), Fraction(2, 3)),
             (Fraction(2, 3), Fraction(4, 5))]
    print("\nExact rational complete-graph checks")
    all_match = True
    for lam, tau in pairs:
        for n in range(2, 16):
            if complete_graph_formula_exact(n, lam, tau) != complete_graph_scan_exact(n, lam, tau):
                all_match = False
                print(f"  MISMATCH: lambda={lam}, tau={tau}, n={n}")
    assert all_match, "a complete-graph exact check failed"
    print("  all complete-graph checks passed:", all_match)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Discounted hitting domination: reproducibility computations")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"NumPy:    {np.__version__}")
    print(f"NetworkX: {nx.__version__}")
    print(f"SciPy:    {scipy.__version__}")

    karate = nx.karate_club_graph()
    result = placement_study(karate, "Zachary karate-club network",
                             lam=0.85, tau=0.55, weighted=True, exhaustive_check=False)
    ws, wc = weighted_karate_baselines(karate, 0.85, 0.55)
    print(f"  weighted-strength sources={ws}; weighted-closeness sources={wc}; "
          f"HD={len(result['hd_sources'])}")

    summary = karate_exhaustive_summary(karate, list(karate.nodes()), 0.85, 0.55)
    print("  exhaustive Karate enumeration:")
    print(f"    best worst-support over all 4-source sets = {summary['best_four_support']:.6f}")
    print(f"    minimum dominating sets (size 4) = {summary['minimum_dominating_sets']}")
    print(f"    best worst-support among them = {summary['best_minimum_dominating_support']:.6f}")
    print(f"    feasible 5-source DHD sets = {summary['feasible_five_sets']}")
    print(f"    of these, classical dominating sets = {summary['feasible_five_dominating']}")

    ba = nx.barabasi_albert_graph(120, 2, seed=7)
    placement_study(ba, "Barabasi-Albert BA(120,2), seed 7",
                    lam=0.85, tau=0.30, weighted=False, exhaustive_check=False)

    check_distance_recovery_examples()
    check_spider_formula()
    check_complete_graph_formula_exact()


if __name__ == "__main__":
    main()
