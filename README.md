# Discounted Hitting Domination

Reproducibility code accompanying the manuscript

**Discounted Hitting Domination on Graphs with Submodularity, Complexity and Exact Algorithms**

Julian D. Allagan, Kevin Pereyra, and William A. Massey.

**Software author:** Julian D. Allagan

## Overview

This repository contains the computational code used to reproduce and verify
the numerical and small-instance results reported in the manuscript.

The main script, `reproduce_results.py`, implements the discounted hitting
equilibrium, source-placement comparisons, exact optimization certificates,
and independent checks of several theoretical results.

## Contents

`reproduce_results.py`:

- reproduces the source-placement study on Zachary's weighted karate-club
  network and on the Barabási–Albert graph `BA(120,2)` with seed `7`;
- computes the greedy discounted-hitting (HD), degree, closeness,
  weighted-strength, and weighted-closeness placement comparisons;
- solves the exact mixed-integer linear formulation for the discounted
  hitting domination number;
- computes exact classical domination numbers for the network experiments;
- independently checks the weighted-karate optimum by exhaustive search;
- reports source-selection orders, worst-vertex support values, and
  Jaccard similarities used in the computational comparisons;
- checks selected small instances of the distance-\(r\) recovery identity
  \(\delta_{\lambda,\tau}(G)=\gamma_r(G)\) by exhaustive search;
- checks the exact spider formula against exhaustive subset minimization
  on small spiders; and
- verifies the complete-graph formula on small instances using exact
  rational arithmetic.

## Requirements

- Python 3
- NumPy
- NetworkX
- SciPy

Install the required packages with:

```bash
pip install -r requirements.txt
