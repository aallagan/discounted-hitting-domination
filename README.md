# Discounted Hitting Domination

Reproducibility code accompanying the manuscript

**Discounted Hitting Domination on Graphs with Submodularity, Complexity and Exact Algorithms**

Julian D. Allagan, Kevin Pereyra, and William A. Massey.

## Contents

`reproduce_results.py` reproduces the computational study in the manuscript and:

- reproduces the Zachary karate-club and Barabási–Albert source-placement comparisons;
- reports the source-selection curves used in the computational figure;
- certifies the weighted-karate optimum by exhaustive search below a feasible upper bound;
- reports weighted-strength and weighted-closeness baselines;
- checks selected small instances of the distance-\(r\) recovery theorem by exhaustive search; and
- verifies the complete-graph formula on small instances using exact rational arithmetic.

## Requirements

- Python 3
- NumPy
- NetworkX

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run

Run the reproducibility script from the repository directory with:

```bash
python reproduce_results.py
```

The script reports the Python, NumPy, and NetworkX versions used in the run.

## Computational details

General equilibrium computations and the exhaustive small-graph checks use
double-precision floating-point linear algebra through NumPy. The
complete-graph checks use exact rational arithmetic through Python's
`fractions.Fraction`.

The script reproduces the numerical source-placement comparisons reported
in the manuscript, including the weighted Zachary karate-club network and
the Barabási–Albert graph generated with the fixed seed specified in the
manuscript.

The weighted-karate optimum is certified by exhaustive search below a
known feasible upper bound. The Barabási–Albert HD value is a greedy
placement result and is not claimed to be a certified global optimum.

## Reproducibility

The repository contains:

- `reproduce_results.py` — computational reproduction and verification script;
- `requirements.txt` — Python package requirements;
- `CITATION.cff` — citation metadata;
- `LICENSE` — MIT License.

The network data are obtained through NetworkX. The Barabási–Albert graph
is generated deterministically using the fixed seed stated in the
manuscript and in the script.

## License

This software is released under the MIT License. See [LICENSE](LICENSE)
for details.

## Citation

If you use this code, please cite the associated manuscript and this
software repository. Citation metadata are provided in
[CITATION.cff](CITATION.cff).

A permanent DOI will be added after the software release is archived on
Zenodo.
