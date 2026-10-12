"""Replay the policies of Table 2 on every kept Criteo campaign (Section 7.2, Appendix D).

A policy runs through the logged auctions of a campaign in their original order. It wins an
auction when its bid reaches the logged cost and then pays this cost. The target C is the
campaign's training-week CPA. All constants are stated for C = 2 and rescaled with u = C/2:
q_hi = K^(1/4)/u, q_lo = 1/(u T), the AdaGrad scale alpha C = 2, and constant steps GRID/u^p.

For every campaign this computes:
  - the evaluation run of WADAP (AdaGrad step) at gub and at gub = 1, break-even and unpaced bidding;
  - each constant-step policy at every grid step, on the training week (to choose one common
    step) and on the evaluation period (the table rows and the trade-off curves);
  - WADAP at gub and at gub = 1 for a grid of alpha C, and the static bid kappa C vhat for a
    grid of kappa (the trade-off curves of Figure 1).
Writes results/criteo.json.gz.
"""
import dataclasses
import gzip
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import sim  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data", "criteo")
CAMPAIGN_DIR = os.path.join(DATA_DIR, "campaigns")
B_SLACK = 1e-9                          # budget B = mbar T (1 + B_SLACK): never binding
GRID = np.logspace(-5, 1, 13)           # constant steps, stated for C = 2
# the power of 1/money in each step's unit: a step for C = 2 becomes GRID / u^p
STEPS = dict(pilot=("wadap", 2), dmd=("dmd", 2), pid=("pid-cpa", 0), cast_rep=("cast-rep", 2),
             pred=("pred-pacer", 2))
AC_GRID = tuple(sorted({round(2 * 10 ** (k / 2), 12) for k in range(-6, 7)} | {.5, 1., 4.}))
KAPPA_LO = .25


def load(campaign):
    with np.load(os.path.join(CAMPAIGN_DIR, f"c{campaign}.npz")) as z:
        d = {a: z[a] for a in z.files}
    split = lambda p: {a[len(p):]: d[a] for a in d if a.startswith(p)}  # noqa: E731
    return split("eval_"), split("train_"), dict(mbar=float(d["mbar"]), cpa=float(d["cpa"]),
                                                 gub=float(d["gub"]))


def setting(s, C):
    T = s["m"].size
    cfg = sim.Config(T=T, K=int(s["K"]), H=max(int(s["H"]), 1), C=C)
    u = C / 2.
    return cfg, dict(u=u, q_hi=cfg.K ** .25 / u, q_lo=1. / (u * T), alpha=1. / u)


def run(alg, s, cfg, p, B, mbar, eta=0., credit=1., alpha=None):
    return sim.run_campaign(alg, s, cfg, B, p["q_lo"], p["q_hi"], float(eta), gamma_tilde=float(credit),
                            g0=mbar, alpha=p["alpha"] if alpha is None else float(alpha), money_unit=p["u"])


def outcome(r, C):
    return dict(spend=r["spend"], conv=r["conv"], rd=max(r["spend"] - C * r["conv"], 0.), pr=r["pr"])


def kappa_max(campaign):
    ev, _, c = load(campaign)
    return float((ev["m"] / (c["cpa"] * ev["vhat"])).max())


def campaign_runs(task):
    campaign, kappas = task
    ev, tr, c = load(campaign)
    C, gub, mbar = c["cpa"], c["gub"], c["mbar"]
    cfg, p = setting(ev, C)
    B = mbar * cfg.T * (1 + B_SLACK)
    out = dict(campaign=campaign, C=C, T=cfg.T, gub=gub,
               gamma_eval=float(ev["chi"].sum() / ev["omega"].sum()) if ev["omega"].sum() else float("nan"),
               opt=sim.benchmark(ev, cfg, B))
    out["rows"] = dict(
        ag=outcome(run("wadap-ag", ev, cfg, p, B, mbar, credit=gub), C),
        ag_lit=outcome(run("wadap-ag", ev, cfg, p, B, mbar, credit=1.), C),
        breakeven=outcome(run("breakeven", ev, cfg, p, B, mbar, credit=gub), C),
        full=outcome(run("full-throttle", ev, cfg, p, B, mbar, credit=gub), C))

    # training week: the observed ledger [S - C R / gub]_+ at the end, for every step
    tcfg, tp = setting(tr, C)
    tB = mbar * tcfg.T * (1 + B_SLACK)
    out["train"] = dict(spend_full=float(tr["m"].sum()))
    out["grid"] = {}
    for arm, (alg, power) in STEPS.items():
        steps = GRID / p["u"] ** power
        tsteps = GRID / tp["u"] ** power
        ledger, pred = [], []
        for eta in tsteps:
            r = run(alg, tr, tcfg, tp, tB, mbar, eta, gub)
            ledger.append(max(0., r["spend"] - C * r["reported"] / gub))
            pred.append(r["pred_conv"])
        out["train"][arm] = dict(ledger=ledger, pred_conv=pred)
        out["grid"][arm] = [outcome(run(alg, ev, cfg, p, B, mbar, eta, gub), C) for eta in steps]

    out["alpha_c"] = {arm: [outcome(run("wadap-ag", ev, cfg, p, B, mbar, credit=cr, alpha=ac / C), C)
                            for ac in AC_GRID] for arm, cr in (("ag", gub), ("ag_lit", 1.))}
    kfull = 1. + 1. / (C * p["q_lo"])       # kappa of unpaced bidding
    out["kappa"] = [outcome(run("breakeven", ev, dataclasses.replace(cfg, C=k * C), p, B, mbar, credit=gub), C)
                    for k in list(kappas) + [kfull]]
    return out


def main():
    t0 = time.time()
    ids = sorted(int(f[1:-4]) for f in os.listdir(CAMPAIGN_DIR) if f.endswith(".npz"))
    with Pool(int(os.environ.get("WORKERS", os.cpu_count()))) as pool:
        kmax = max(pool.map(kappa_max, ids))
        # static bids: kappa on a sqrt(2) grid from 1/4 up to the largest m / (C vhat)
        kappas, j = [], round(2 * math.log2(KAPPA_LO))
        while not kappas or kappas[-1] < kmax:
            kappas.append(2 ** (j / 2))
            j += 1
        camps = []
        for i, o in enumerate(pool.imap_unordered(campaign_runs, [(c, kappas) for c in ids])):
            camps.append(o)
            print(f"{i + 1}/{len(ids)} campaigns", flush=True)
    camps.sort(key=lambda o: o["campaign"])
    with open(os.path.join(DATA_DIR, "summary.json")) as f:
        data = json.load(f)
    res = dict(data=data, K=64, grid=GRID.tolist(), alpha_c=list(AC_GRID), kappa=kappas + ["full"],
               campaigns=camps)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with gzip.GzipFile(os.path.join(ROOT, "results", "criteo.json.gz"), "wb", mtime=0) as f:
        f.write(json.dumps(res).encode())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
