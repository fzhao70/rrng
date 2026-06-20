#!/usr/bin/env python3
"""End-to-end proof: reproduce R's set.seed(100) wus-only attribution in PURE PYTHON,
using rrng.RRNG for the bootstrap resampling (R's sample()) + the project's validated GEV fit.
Compare to R: F 6.817 [5.262, 9.863]  CF 33.37 [21.12, 76.50]  RR 4.8836 [2.7467, 11.7427].

Run from anywhere:
    /glade/u/apps/opt/conda/envs/npl-2025b/bin/python rrng/examples/validate_bootstrap.py
(the input vectors are produced by examples/r_validate_single.R)."""
import os
import sys

import numpy as np
import pandas as pd

# rrng package root (two levels up) and the project root (three levels up, for attribution_huc)
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PKG_ROOT)
sys.path.insert(0, PKG_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from rrng import RRNG
from attribution_huc import gev_fit_lmom, gev_sf_raw, _q

cf = pd.read_csv(os.path.join(PROJECT_ROOT, "analysis_source_compare/wus_cf.csv"))["z"].values.astype(float)
fa = pd.read_csv(os.path.join(PROJECT_ROOT, "analysis_source_compare/wus_fa.csv"))["z"].values.astype(float)
event = -1.43528365813994
N = 5000
n_cf, n_fa = cf.size, fa.size
x1 = -event

rng = RRNG(100)                       # <-- pure-Python R Mersenne-Twister, seed 100
fact_probs = np.full(N, np.nan)
counter_probs = np.full(N, np.nan)
for i in range(N):
    # R order: sample(counterfactual) THEN sample(factual), each length-preserving w/ replacement
    cf2 = cf[rng.sample_index(n_cf, n_cf)]
    fa2 = fa[rng.sample_index(n_fa, n_fa)]
    pc = gev_fit_lmom(-cf2); pf = gev_fit_lmom(-fa2)
    if pc is None or pf is None:
        continue
    fact_probs[i] = gev_sf_raw(pf, x1)
    counter_probs[i] = gev_sf_raw(pc, x1)

lo, hi = 0.025, 0.975
event_ratios = fact_probs / counter_probs
F_med, F_lo, F_hi = 1/_q(fact_probs, 0.5), 1/_q(fact_probs, hi), 1/_q(fact_probs, lo)
CF_med, CF_lo, CF_hi = 1/_q(counter_probs, 0.5), 1/_q(counter_probs, hi), 1/_q(counter_probs, lo)
RR_med, RR_lo, RR_hi = _q(event_ratios, 0.5), _q(event_ratios, lo), _q(event_ratios, hi)

print("PYTHON (r_rng, seed=100):  "
      f"F {F_med:.3f} [{F_lo:.3f}, {F_hi:.3f}]  CF {CF_med:.2f} [{CF_lo:.2f}, {CF_hi:.2f}]  "
      f"RR {RR_med:.4f} [{RR_lo:.4f}, {RR_hi:.4f}]")
print("R      (seed=100)       :  F 6.817 [5.262, 9.863]  CF 33.37 [21.12, 76.50]  RR 4.8836 [2.7467, 11.7427]")
