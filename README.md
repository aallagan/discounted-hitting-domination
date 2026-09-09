# Discounted Hitting Domination

Reproducibility code accompanying the manuscript

**Discounted Hitting Domination on Graphs:
Submodularity, Complexity, and Exact Algorithms**

Julian D. Allagan, Kevin Pereyra, and William A. Massey.

## Contents

`reproduce_results.py` reproduces the computational study in the manuscript and:

- reproduces the Zachary karate-club and Barabási–Albert source-placement comparisons;
- reports the source-selection curves used in the computational figure;
- certifies the weighted-karate optimum by exhaustive search below a feasible upper bound;
- reports weighted-strength and weighted-closeness baselines;
- checks selected instances of the distance-r recovery theorem by exhaustive search; and
- verifies the complete-graph formula on small instances using exact rational arithmetic.

## Requirements

- Python 3
- NumPy
- NetworkX

Install dependencies with:

```bash
pip install -r requirements.txt
