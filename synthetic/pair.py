"""The pair I_A, I_B of the lower bound (Section 7.1, Appendix C).

WADAP with the AdaGrad step reads the reports at the declared bound gub = 0.9, with
q_hi = 2/C. For each run we keep the final and the peak deficit on I_A and the conversions
on I_B, for 100 seeds:
  - T = 20,000 and s in {0.1, 0.2, 0.3}            (lower-bound check)
  - T = 5,000 ... 320,000, s = 0.2 and s = sqrt(L/T)  (growth with the horizon)
L = H = 100 throughout. Writes results/pair.json.
"""
import json
import math
import os
import sys
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import sim  # noqa: E402

SEEDS = range(1000, 1100)
HORIZONS = (5000, 10000, 20000, 40000, 80000, 160000, 320000)
GUB = 0.9


def window_run(inst, cfg, gub=GUB):
    """WADAP with the AdaGrad step, computed one window at a time (needs H = L)."""
    assert cfg.H == cfg.L
    C, L = cfg.C, cfg.L
    q_lo, q_hi = 1 / cfg.T, 2 / C
    g0sq = sim.price_cap(cfg) * sim.price_cap(cfg)
    mu = C + 1 / q_lo
    spend = conv = reported = peak = gsq = prev = 0.0
    counts = np.zeros(cfg.K + 1)
    for k in range(cfg.K):
        sl = slice(k * L, (k + 1) * L)
        win = mu * inst["vhat"][sl] >= inst["m"][sl]
        costs = win * inst["m"][sl]
        conversions = win * inst["omega"][sl]
        peak = max(peak, spend - C * conv + np.max(np.cumsum(costs - C * conversions)))
        spend += costs.sum()
        conv += conversions.sum()
        arrived = (np.arange(k * L, (k + 1) * L) + inst["D"][sl]) // L
        credited = conversions * inst["chi"][sl]
        counts[k] += (credited * (arrived == k)).sum()
        counts[k + 1] += (credited * (arrived == k + 1)).sum()
        reported += counts[k]
        ledger = spend - C * reported / gub
        inc = ledger - prev
        gsq += inc * inc
        prev = ledger
        q = min(max(1 / np.sqrt(g0sq + gsq) * max(ledger, 0), q_lo), q_hi)
        mu = C + 1 / q
    return dict(rd=float(max(0, spend - C * conv)), pr=float(peak), conv=float(conv))


def pair_block(T, s):
    cfg_a = sim.Config(T=T, K=T // 100, H=100, family="pair", pair_branch="A", pair_s=s)
    cfg_b = replace(cfg_a, pair_branch="B")
    C, v0, eps = cfg_b.C, cfg_b.pair_v0, cfg_b.pair_eps
    # the closed forms of Section 6: OPT_pop on I_B and the two sides of the trade-off
    optpop = .5 * v0 * (1 + 1 / (1 + 4 * eps)) * T
    coef = C * (.5 + 2 * eps + s)
    rhs = (1 + 2 * eps) / (1 + 4 * eps) * s * C * v0 * T
    out = dict(T=T, s=s, optpop=optpop, coef=coef, rhs=rhs, gap=rhs / coef, rd_A=[], pr_A=[], conv_B=[])
    for seed in SEEDS:
        a = window_run(sim.pair_instance(seed, cfg_a), cfg_a)
        b = window_run(sim.pair_instance(seed, cfg_b), cfg_b)
        out["rd_A"].append(a["rd"])
        out["pr_A"].append(a["pr"])
        out["conv_B"].append(b["conv"])
    return out


def main():
    res = dict(pair={}, horizon_fixed={}, horizon_shrinking={})
    for s in (.1, .2, .3):
        res["pair"][str(s)] = pair_block(20000, s)
        print("pair s =", s, flush=True)
    for T in HORIZONS:
        res["horizon_fixed"][str(T)] = pair_block(T, .2)
        res["horizon_shrinking"][str(T)] = pair_block(T, math.sqrt(100 / T))
        print("horizon T =", T, flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "pair.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
