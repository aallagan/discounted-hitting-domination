"""
Reproducibility code for
"Discounted Hitting Domination on Graphs:
 Submodularity, Complexity, and Exact Algorithms"

This script:
1. reproduces the source-placement study on Zachary's weighted karate-club
   network and on BA(120,2) with seed 7;
2. reports the HD, degree, closeness, weighted-strength, and weighted-closeness
   comparisons used in the manuscript;
3. records the source-selection curves underlying the numerical figure;
4. certifies the weighted-karate optimum by exhaustive search below a known
   feasible upper bound;
5. checks the distance-r recovery identity delta_{lambda,tau}=gamma_r on
   selected small graphs by exhaustive search; and
6. checks the complete-graph formula on a grid of small instances using exact
   rational arithmetic.

Numerical equilibrium computations use double-precision floating-point
arithmetic through numpy.linalg.solve.  Complete-graph checks use exact
rational arithmetic through fractions.Fraction.

Requirements:
    numpy
    networkx
"""

from __future__ import annotations

import itertools
import sys
from fractions import Fraction
from typing import Iterable, Sequence

import networkx as nx
import numpy as np


TOL = 1e-12


# ---------------------------------------------------------------------------
# Equilibrium h^S via the pinned linear system
# ---------------------------------------------------------------------------

def equilibrium(
    G: nx.Graph,
    nodes: Sequence,
    lam: float,
    sources: Iterable[int],
    weighted: bool = False,
) -> np.ndarray:
    """Return the equilibrium support vector h^S.

    The source set is supplied as indices into ``nodes``.  If ``weighted`` is
    True, the NetworkX edge attribute ``weight`` is used to form the
    row-stochastic trust matrix; otherwise the simple-random-walk matrix is
    used.
    """
    adjacency = nx.to_numpy_array(
        G,
        nodelist=list(nodes),
        weight=("weight" if weighted else None),
        dtype=float,
    )
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


def worst_support(
    G: nx.Graph,
    nodes: Sequence,
    lam: float,
    sources: Iterable[int],
    weighted: bool = False,
) -> float:
    """Return min_i h_i^S."""
    sources = list(sources)
    if not sources:
        return 0.0
    return float(equilibrium(G, nodes, lam, sources, weighted).min())


def q_tau(
    G: nx.Graph,
    nodes: Sequence,
    lam: float,
    tau: float,
    sources: Iterable[int],
    weighted: bool = False,
) -> float:
    """Return Q_tau(S)=sum_i min(h_i^S,tau)."""
    sources = list(sources)
    if not sources:
        return 0.0
    h = equilibrium(G, nodes, lam, sources, weighted)
    return float(np.minimum(h, tau).sum())


# ---------------------------------------------------------------------------
# Source-placement study
# ---------------------------------------------------------------------------

def greedy_hd(
    G: nx.Graph,
    nodes: Sequence,
    lam: float,
    tau: float,
    weighted: bool,
) -> tuple[list[int], list[tuple[int, float]]]:
    """Greedy submodular-cover placement, ties broken by smallest node index.

    Returns the selected source indices in order and the curve
    [(number_of_sources, worst_support), ...].
    """
    n = len(nodes)
    sources: list[int] = []
    remaining = list(range(n))
    curve: list[tuple[int, float]] = []
    current_q = 0.0

    while worst_support(G, nodes, lam, sources, weighted) < tau - TOL:
        best_vertex = None
        best_gain = -1.0

        for v in sorted(remaining):
            gain = q_tau(G, nodes, lam, tau, sources + [v], weighted) - current_q
            if gain > best_gain + 1e-15:
                best_gain = gain
                best_vertex = v

        if best_vertex is None:
            raise RuntimeError("Greedy placement failed to select a vertex.")

        sources.append(best_vertex)
        remaining.remove(best_vertex)
        current_q = q_tau(G, nodes, lam, tau, sources, weighted)
        curve.append(
            (len(sources), worst_support(G, nodes, lam, sources, weighted))
        )

        if len(sources) == n:
            break

    return sources, curve


def add_until(
    G: nx.Graph,
    nodes: Sequence,
    lam: float,
    tau: float,
    ranking: Sequence[int],
    weighted: bool,
) -> tuple[list[int], list[tuple[int, float]]]:
    """Add vertices in a fixed ranking until the floor is reached."""
    sources: list[int] = []
    curve: list[tuple[int, float]] = []

    for v in ranking:
        sources.append(v)
        w = worst_support(G, nodes, lam, sources, weighted)
        curve.append((len(sources), w))
        if w >= tau - TOL:
            break

    return sources, curve


def greedy_dominating_set(G: nx.Graph, nodes: Sequence) -> list[int]:
    """Greedy classical dominating set; ties broken by smallest node index."""
    index = {u: i for i, u in enumerate(nodes)}
    closed_neighborhood = {
        index[u]: {index[u]} | {index[w] for w in G.neighbors(u)}
        for u in nodes
    }

    uncovered = set(range(len(nodes)))
    sources: list[int] = []

    while uncovered:
        v = max(
            range(len(nodes)),
            key=lambda x: (len(closed_neighborhood[x] & uncovered), -x),
        )
        sources.append(v)
        uncovered -= closed_neighborhood[v]

    return sources


def is_classical_dominating_set(
    G: nx.Graph,
    nodes: Sequence,
    sources: Iterable[int],
) -> bool:
    """Check whether the indexed source set is a classical dominating set."""
    source_set = set(sources)
    covered = set(source_set)
    for i in source_set:
        covered |= {nodes.index(w) for w in G.neighbors(nodes[i])}
    return len(covered) == len(nodes)


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    """Jaccard coefficient of two finite sets."""
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def certify_optimum_below_feasible_upper_bound(
    G: nx.Graph,
    nodes: Sequence,
    lam: float,
    tau: float,
    weighted: bool,
    feasible_sources: Sequence[int],
) -> int:
    """Exhaustively certify optimality below a known feasible upper bound."""
    upper_bound = len(feasible_sources)
    if worst_support(G, nodes, lam, feasible_sources, weighted) < tau - TOL:
        raise ValueError("The supplied upper-bound source set is not feasible.")

    for k in range(1, upper_bound):
        for S in itertools.combinations(range(len(nodes)), k):
            if worst_support(G, nodes, lam, S, weighted) >= tau - TOL:
                return k
    return upper_bound


def weighted_karate_baselines(
    G: nx.Graph,
    lam: float,
    tau: float,
) -> tuple[int, int]:
    """Return source counts for weighted strength and weighted closeness."""
    nodes = list(G.nodes())

    strength = {u: G.degree(u, weight="weight") for u in nodes}
    strength_order = sorted(
        range(len(nodes)),
        key=lambda i: (-strength[nodes[i]], i),
    )

    H = G.copy()
    for _, _, data in H.edges(data=True):
        data["inverse_weight"] = 1.0 / data["weight"]

    closeness = nx.closeness_centrality(H, distance="inverse_weight")
    closeness_order = sorted(
        range(len(nodes)),
        key=lambda i: (-closeness[nodes[i]], i),
    )

    strength_sources, _ = add_until(
        G, nodes, lam, tau, strength_order, weighted=True
    )
    closeness_sources, _ = add_until(
        G, nodes, lam, tau, closeness_order, weighted=True
    )

    return len(strength_sources), len(closeness_sources)


def placement_study(
    G: nx.Graph,
    name: str,
    lam: float,
    tau: float,
    weighted: bool,
    certify_exact: bool,
) -> dict:
    """Run one placement comparison and print a reproducible summary."""
    nodes = list(G.nodes())
    degrees = [G.degree(u) for u in nodes]

    degree_order = sorted(
        range(len(nodes)),
        key=lambda i: (-degrees[i], i),
    )

    closeness_values = nx.closeness_centrality(G)
    closeness_order = sorted(
        range(len(nodes)),
        key=lambda i: (-closeness_values[nodes[i]], i),
    )

    hd_sources, hd_curve = greedy_hd(G, nodes, lam, tau, weighted)
    degree_sources, degree_curve = add_until(
        G, nodes, lam, tau, degree_order, weighted
    )
    closeness_sources, closeness_curve = add_until(
        G, nodes, lam, tau, closeness_order, weighted
    )

    dominating_set = greedy_dominating_set(G, nodes)
    dominating_worst = worst_support(
        G, nodes, lam, dominating_set, weighted
    )

    exact_delta = None
    if certify_exact:
        exact_delta = certify_optimum_below_feasible_upper_bound(
            G, nodes, lam, tau, weighted, hd_sources
        )

    result = {
        "name": name,
        "n": len(nodes),
        "m": G.number_of_edges(),
        "Delta": max(degrees),
        "walk": "weighted" if weighted else "simple random walk",
        "lambda": lam,
        "tau": tau,
        "hd_sources": hd_sources,
        "hd_curve": hd_curve,
        "degree_sources": degree_sources,
        "degree_curve": degree_curve,
        "closeness_sources": closeness_sources,
        "closeness_curve": closeness_curve,
        "dominating_set": dominating_set,
        "dominating_worst": dominating_worst,
        "exact_delta": exact_delta,
        "jaccard_hd_degree": jaccard(hd_sources, degree_sources),
        "jaccard_hd_closeness": jaccard(hd_sources, closeness_sources),
        "hd_is_classical_dom": is_classical_dominating_set(
            G, nodes, hd_sources
        ),
    }

    print(f"\n{name}")
    print(
        f"  n={result['n']}  m={result['m']}  Delta={result['Delta']}  "
        f"walk={result['walk']}  lambda={lam}  tau={tau}"
    )
    print(f"  HD source set (selection order): {hd_sources}")
    if exact_delta is not None:
        print(f"  HD sources={len(hd_sources)}  exact optimum={exact_delta}")
    else:
        print(f"  HD sources={len(hd_sources)}  (not certified optimal)")
    print(
        f"  degree sources={len(degree_sources)}  "
        f"closeness sources={len(closeness_sources)}"
    )
    print(
        f"  greedy dominating set |D|={len(dominating_set)}  "
        f"worst support={dominating_worst:.6f}  "
        f"meets tau={dominating_worst >= tau - 1e-9}"
    )
    print(
        f"  Jaccard(HD,degree)={result['jaccard_hd_degree']:.2f}  "
        f"Jaccard(HD,closeness)={result['jaccard_hd_closeness']:.2f}"
    )
    print(
        "  HD set is a classical dominating set? "
        f"{result['hd_is_classical_dom']}"
    )
    print("  HD curve:", [(k, round(v, 6)) for k, v in hd_curve])
    print("  degree curve:", [(k, round(v, 6)) for k, v in degree_curve])
    print(
        "  closeness curve:",
        [(k, round(v, 6)) for k, v in closeness_curve],
    )

    return result


# ---------------------------------------------------------------------------
# Distance-r recovery checks
# ---------------------------------------------------------------------------

def gamma_r_bruteforce(G: nx.Graph, r: int) -> int:
    """Compute the distance-r domination number by exhaustive search."""
    distances = dict(nx.all_pairs_shortest_path_length(G))
    vertices = list(G.nodes())

    for k in range(len(vertices) + 1):
        for S in itertools.combinations(vertices, k):
            if all(
                any(distances[v].get(s, 10**9) <= r for s in S)
                for v in vertices
            ):
                return k

    raise RuntimeError("No distance-r dominating set found.")


def delta_bruteforce(
    G: nx.Graph,
    lam: float,
    tau: float,
) -> int:
    """Compute delta_{lambda,tau}(G) by exhaustive floating-point search."""
    nodes = list(G.nodes())

    for k in range(1, len(nodes) + 1):
        for S in itertools.combinations(range(len(nodes)), k):
            if worst_support(G, nodes, lam, S) >= tau - TOL:
                return k

    raise RuntimeError("No feasible source set found.")


def check_distance_recovery_examples() -> None:
    """Check selected small instances of delta_{lambda,tau}=gamma_r."""
    examples = [
        ("P7", nx.path_graph(7), 1),
        ("C8", nx.cycle_graph(8), 1),
        ("Petersen", nx.petersen_graph(), 1),
        ("P9", nx.path_graph(9), 2),
        ("3x3 grid", nx.grid_2d_graph(3, 3), 1),
    ]

    print("\nDistance-r recovery checks")
    all_match = True

    for name, G, r in examples:
        G = nx.convert_node_labels_to_integers(G)
        Delta = max(dict(G.degree()).values())

        # Choose lambda so that lambda*Delta^r < 1, then choose tau strictly
        # inside the recovery window.
        lam = 0.9 / (Delta**r + 0.5)
        lower = lam ** (r + 1)
        upper = (lam / Delta) ** r
        tau = (lower + upper) / 2.0

        gamma = gamma_r_bruteforce(G, r)
        delta = delta_bruteforce(G, lam, tau)
        match = gamma == delta
        all_match &= match

        print(
            f"  {name}: Delta={Delta}, r={r}, lambda={lam:.6f}, "
            f"tau in ({lower:.6f},{upper:.6f}], "
            f"gamma_r={gamma}, delta={delta}, match={match}"
        )

    assert all_match, "A distance-r recovery check failed."
    print("  all listed recovery checks passed: True")


# ---------------------------------------------------------------------------
# Exact rational checks for the complete-graph formula
# ---------------------------------------------------------------------------

def ceil_fraction(x: Fraction) -> int:
    """Exact ceiling of a Fraction."""
    return -(-x.numerator // x.denominator)


def complete_graph_worst_exact(
    n: int,
    lam: Fraction,
    q: int,
) -> Fraction:
    """Exact common non-source support on K_n with q sources."""
    if q >= n:
        return Fraction(1, 1)
    return lam * q / ((n - 1) - lam * (n - q - 1))


def complete_graph_formula_exact(
    n: int,
    lam: Fraction,
    tau: Fraction,
) -> int:
    """Exact value from the closed formula for delta_{lambda,tau}(K_n)."""
    if tau == 1:
        return n

    threshold = (
        tau * (n - 1) * (1 - lam)
        / (lam * (1 - tau))
    )
    return min(n, ceil_fraction(threshold))


def complete_graph_scan_exact(
    n: int,
    lam: Fraction,
    tau: Fraction,
) -> int:
    """Exact source-count scan on K_n using the symmetric equilibrium value."""
    for q in range(1, n + 1):
        if complete_graph_worst_exact(n, lam, q) >= tau:
            return q
    return n


def check_complete_graph_formula_exact() -> None:
    """Verify the complete-graph formula on small rational instances."""
    parameter_pairs = [
        (Fraction(1, 2), Fraction(1, 4)),
        (Fraction(3, 5), Fraction(1, 3)),
        (Fraction(4, 5), Fraction(1, 2)),
        (Fraction(7, 8), Fraction(2, 3)),
        (Fraction(2, 3), Fraction(4, 5)),
    ]

    print("\nExact rational complete-graph checks")
    all_match = True

    for lam, tau in parameter_pairs:
        for n in range(2, 16):
            formula = complete_graph_formula_exact(n, lam, tau)
            scan = complete_graph_scan_exact(n, lam, tau)
            if formula != scan:
                all_match = False
                print(
                    f"  MISMATCH: lambda={lam}, tau={tau}, n={n}, "
                    f"formula={formula}, scan={scan}"
                )

    assert all_match, "A complete-graph exact rational check failed."
    print("  all complete-graph checks passed in exact rational arithmetic: True")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Discounted hitting domination: reproducibility computations")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"NumPy:    {np.__version__}")
    print(f"NetworkX: {nx.__version__}")

    karate = nx.karate_club_graph()
    karate_result = placement_study(
        karate,
        "Zachary karate-club network",
        lam=0.85,
        tau=0.55,
        weighted=True,
        certify_exact=True,
    )

    weighted_strength_count, weighted_closeness_count = (
        weighted_karate_baselines(karate, 0.85, 0.55)
    )
    print(
        "  weighted-strength sources="
        f"{weighted_strength_count}; "
        "weighted-closeness sources="
        f"{weighted_closeness_count}; "
        f"HD={len(karate_result['hd_sources'])}"
    )

    ba = nx.barabasi_albert_graph(120, 2, seed=7)
    placement_study(
        ba,
        "Barabasi-Albert BA(120,2), seed 7",
        lam=0.85,
        tau=0.30,
        weighted=False,
        certify_exact=False,
    )

    check_distance_recovery_examples()
    check_complete_graph_formula_exact()


if __name__ == "__main__":
    main()
