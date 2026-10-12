import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "synthetic"))
import sim  # noqa: E402
from pair import window_run  # noqa: E402

CFG = sim.Config(T=4000, K=40, H=100)


def test_benchmark_keeps_tied_auctions_together():
    # 100 cheap and 100 expensive auctions, half of each converts: the best feasible fixed
    # multiplier wins the cheap ones only. Splitting the expensive group would claim 93.
    T = 200
    vhat = np.full(T, 0.5)
    m = np.r_[np.full(100, 0.5), np.full(100, 1.6)]
    omega = np.zeros(T, dtype=np.int8)
    omega[:50] = omega[100:150] = 1
    inst = dict(vhat=vhat, m=m, omega=omega)
    assert sim.benchmark(inst, sim.Config(T=T, K=2, H=1), B=1e9) == 50


def test_literal_reading_never_underestimates_the_deficit():
    for seed in range(3):
        inst = sim.make_instance(seed, CFG)
        r = sim.run_campaign("wadap", inst, CFG, 0.36 * CFG.T, 1 / CFG.T, CFG.K ** .25, 0.03, gamma_tilde=1.0)
        assert r["max_gap"] <= 1e-9


@pytest.mark.parametrize("branch", ["A", "B"])
def test_window_run_matches_round_by_round(branch):
    cfg = sim.Config(T=4000, K=40, H=100, family="pair", pair_branch=branch)
    for seed in (11, 12):
        inst = sim.pair_instance(seed, cfg)
        w = window_run(inst, cfg)
        r = sim.run_campaign("wadap-ag", inst, cfg, sim.price_cap(cfg) * cfg.T, 1 / cfg.T, 2 / cfg.C, 0.,
                             gamma_tilde=0.9)
        assert r["conv"] == w["conv"]
        for key in ("rd", "pr"):
            assert r[key] == pytest.approx(w[key], abs=1e-6)


def test_adagrad_step_ignores_eta():
    inst = sim.make_instance(7, CFG)
    runs = [sim.run_campaign("wadap-ag", inst, CFG, 0.36 * CFG.T, 1 / CFG.T, CFG.K ** .25, eta, gamma_tilde=.9)
            for eta in (0., 5.)]
    assert runs[0] == runs[1]


@pytest.mark.parametrize("alg,power", [("wadap", 2), ("wadap-ag", 0), ("dmd", 2), ("pred-pacer", 2),
                                       ("pid-cpa", 0), ("cast-rep", 2), ("breakeven", 0), ("full-throttle", 0)])
def test_currency_invariance(alg, power):
    # Quoting every amount of money in a currency four times smaller changes no decision
    # when the constants are rescaled with u = C/2, as the replay does.
    inst = sim.make_instance(3, CFG)
    out = []
    for f in (1., 4.):
        cfg = sim.Config(T=CFG.T, K=CFG.K, H=CFG.H, C=CFG.C * f)
        x = dict(inst, m=inst["m"] * f)
        u = cfg.C / 2
        mbar = sim.price_cap(CFG) * f
        out.append(sim.run_campaign(alg, x, cfg, mbar * cfg.T, 1 / (u * cfg.T), cfg.K ** .25 / u, 0.03 / u ** power,
                                    gamma_tilde=.9, g0=mbar, alpha=1 / u, money_unit=u))
    assert out[0]["conv"] == out[1]["conv"]
    assert out[1]["rd"] == 4 * out[0]["rd"]
