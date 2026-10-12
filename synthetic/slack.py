"""Cost of a loose reporting bound on the reference campaign (Section 7.1, Appendix C).

WADAP with the AdaGrad step reads the reports at gub in {gamma, 0.87, 0.9, 0.95, 1}, where
gamma = 0.844 is the true credit rate. For each gub we keep the conversions of 100 seeds, and
the target-tightening loss G(C#) = OPT_pop(C) - OPT_pop(C (1 - gamma/gub)).
Writes results/slack.json.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import sim  # noqa: E402

SEEDS = range(1000, 1100)


def main():
    cfg = sim.Config(T=20000, K=200, H=100, rho=.85, delay_mean=20)
    gamma = cfg.rho * (1 - (cfg.delay_mean / (1 + cfg.delay_mean)) ** (cfg.H + 1))
    B = sim.price_cap(cfg) * cfg.T
    Ev, Em = sim.population_curve(cfg, SEEDS)
    opt = sim.optpop(Ev, Em, cfg.C, cfg.T)
    res = dict(gamma=gamma, optpop=opt, runs=[])
    for gub in (gamma, .87, .90, .95, 1.):
        slack = 1 - gamma / gub
        conv = []
        for seed in SEEDS:
            inst = sim.make_instance(seed, cfg)
            r = sim.run_campaign("wadap-ag", inst, cfg, B, 1 / cfg.T, cfg.K ** .25, 0., gamma_tilde=gub)
            conv.append(r["conv"])
        G = opt - sim.optpop(Ev, Em, cfg.C * (1.0 - slack), cfg.T)
        res["runs"].append(dict(gub=gub, slack=slack, G=G, conv=conv))
        print(f"gub = {gub:.3f}", flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "slack.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
