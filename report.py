"""Print the experimental numbers of the paper and draw its two figures from results/.

  Section 7.1 and Appendix C   synthetic checks       results/pair.json, results/slack.json
  Appendix D, Table 2          Criteo replay          results/criteo.json.gz
  figures/fig_real_frontier.pdf (Figure 1), figures/fig_synth.pdf (Figure 2)

Intervals are 95% percentile-bootstrap intervals with 2,000 resamples (seed 68120).
"""
import gzip
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
NBOOT, BOOT_SEED = 2000, 68120


def interval(values):
    values = np.array(values, dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    means = values[rng.integers(0, len(values), (NBOOT, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return float(values.mean()), float(low), float(high)


def load(name):
    path = os.path.join(RESULTS, name)
    with (gzip.open(path, "rt") if name.endswith(".gz") else open(path)) as f:
        return json.load(f)


# ------------------------------------------------------------------------------ synthetic
def excess(b):
    """Left side minus right side of the trade-off bound, divided by the regret coefficient."""
    return [(rd + b["coef"] * (b["optpop"] - n) - b["rhs"]) / b["coef"] for rd, n in zip(b["rd_A"], b["conv_B"])]


def loglog_slope(block, values):
    Ts = sorted(block, key=int)
    return np.polyfit(np.log([int(T) for T in Ts]), np.log([np.mean(values(block[T])) for T in Ts]), 1)[0]


def slope_interval(block, values):
    """The slope with the seeds resampled jointly at every T."""
    Ts = sorted(block, key=int)
    x = np.log([int(T) for T in Ts])
    M = np.array([values(block[T]) for T in Ts])
    idx = np.random.default_rng(BOOT_SEED).integers(0, M.shape[1], (NBOOT, M.shape[1]))
    s = [np.polyfit(x, np.log(M[:, i].mean(axis=1)), 1)[0] for i in idx]
    return np.percentile(s, [2.5, 97.5])


def regret_B(b):
    return [b["optpop"] - n for n in b["conv_B"]]


def synthetic(pair, slack):
    fixed, shrinking = pair["horizon_fixed"], pair["horizon_shrinking"]
    peak = lambda b: b["pr_A"]  # noqa: E731
    lo, hi = slope_interval(fixed, peak)
    err = np.array([r["slack"] for r in slack["runs"]])
    reg = np.array([np.mean([slack["optpop"] - n for n in r["conv"]]) for r in slack["runs"]])
    fit = np.polyval(np.polyfit(err, reg, 1), err)
    r2 = 1 - ((reg - fit) ** 2).sum() / ((reg - reg.mean()) ** 2).sum()
    print("Section 7.1 and Appendix C: synthetic instances (WADAP, AdaGrad step, 100 seeds)")
    print(f"  slope of log peak deficit on I_A against log T, s = 0.2       {loglog_slope(fixed, peak):.2f} "
          f"[{lo:.2f}, {hi:.2f}]")
    print(f"  slope of log regret on I_B against log T, s = sqrt(L/T)       {loglog_slope(shrinking, regret_B):.2f}")
    print(f"  slope of log regret on I_B against log T, s = 0.2             {loglog_slope(fixed, regret_B):.2f}")
    print(f"  R^2 of regret against 1 - gamma/gub, reference campaign       {r2:.3f}")
    pct = {s: 100 * np.mean(excess(pair["pair"][s])) / pair["pair"][s]["gap"] for s in ("0.1", "0.2", "0.3")}
    print("  excess over the lower bound, T = 20,000, s = 0.1 / 0.2 / 0.3   "
          + " / ".join(f"{pct[s]:.1f}%" for s in ("0.1", "0.2", "0.3")))
    m, lo, hi = interval(excess(pair["pair"]["0.2"]))
    print(f"  at s = 0.2: {m:.1f} [{lo:.1f}, {hi:.1f}] conversions of I_B, bound {pair['pair']['0.2']['gap']:.0f}")
    print()


def synthetic_figure(pair, slack):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
    fixed, shrinking = pair["horizon_fixed"], pair["horizon_shrinking"]
    Ts = sorted(fixed, key=int)
    T = np.array([int(t) for t in Ts])
    pr = [np.mean(fixed[t]["pr_A"]) for t in Ts]
    rg = [np.mean(regret_B(shrinking[t])) for t in Ts]
    ax[0].loglog(T, pr, "k-o", ms=4, label=r"Def$_A^{\max}$, $s=0.2$")
    ax[0].loglog(T, rg, "k-s", ms=4, label=r"Reg$_B$, $s=\sqrt{L/T}$")
    ax[0].loglog(T, rg[0] * np.sqrt(T / T[0]), "--", color="0.5", label="slope 1/2")
    ax[0].set_xlabel(r"horizon $T$")
    ax[0].set_title("(a) growth with the horizon", fontsize=9)
    ax[0].legend(fontsize=7, loc="upper left")

    err = np.array([r["slack"] for r in slack["runs"]])
    ci = np.array([interval([slack["optpop"] - n for n in r["conv"]]) for r in slack["runs"]])
    reg, lo, hi = ci.T
    c = np.polyfit(err, reg, 1)
    r2 = 1 - ((reg - np.polyval(c, err)) ** 2).sum() / ((reg - reg.mean()) ** 2).sum()
    xs = np.linspace(err.min(), err.max(), 50)
    ax[1].axhline(0, color="0.85", lw=0.8)
    ax[1].plot(xs, np.polyval(c, xs), "k-", label=rf"linear fit, $R^2={r2:.3f}$")
    ax[1].plot(err, [r["G"] for r in slack["runs"]], "--^", color="0.55", ms=4, label=r"$\mathcal{G}(C^\sharp)$")
    ax[1].errorbar(err, reg, yerr=[reg - lo, hi - reg], fmt="ko", ms=4, capsize=2, label="WADAP")
    ax[1].set_xlabel(r"relative error $1-\gamma/\bar\gamma$")
    ax[1].set_ylabel("regret Reg")
    ax[1].set_title("(b) cost of a loose reporting bound", fontsize=9)
    ax[1].legend(fontsize=7, loc="upper left")

    for i, s in enumerate(("0.1", "0.2", "0.3")):
        b = pair["pair"][s]
        x, y = np.mean(regret_B(b)), np.mean(b["rd_A"])
        ax[2].plot([0, b["gap"]], [b["rhs"], 0], "--", color="0.5", lw=1, label=None if i else "bound (5)")
        ax[2].plot(x, y, "ko", ms=5, label=None if i else "WADAP")
        ax[2].annotate(rf"$s={float(s):.1f}$", (x, y), textcoords="offset points", xytext=(-14, 7), fontsize=7)
    ax[2].set_xlim(0, None)
    ax[2].set_ylim(0, 2500)
    ax[2].set_xlabel(r"regret on $I_B$")
    ax[2].set_ylabel(r"final deficit on $I_A$")
    ax[2].set_title(rf"(c) the pair $I_A/I_B$, $T={pair['pair']['0.2']['T']}$", fontsize=9)
    ax[2].legend(fontsize=7, loc="upper right")
    for a in ax:
        a.tick_params(labelsize=7)
        a.xaxis.label.set_size(8)
        a.yaxis.label.set_size(8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_synth.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------------------ Criteo
ARMS = ("pilot", "dmd", "pid", "cast_rep", "pred")


def common_step(camps, arm):
    """The one step used on every campaign: the smallest mean, over campaigns, of the
    end-of-training-week ledger divided by the unpaced training-week spend; ties go to the
    step with more predicted conversions."""
    L = np.array([c["train"][arm]["ledger"] for c in camps])
    P = np.array([c["train"][arm]["pred_conv"] for c in camps])
    spend_full = np.array([c["train"]["spend_full"] for c in camps], dtype=float)[:, None]
    score = (L / spend_full).mean(axis=0)
    total = P.sum(axis=0)
    return min(range(score.size), key=lambda j: (score[j], -total[j]))


def aggregate(camps, outs):
    """Mean final and peak deficit per unit spent, and pooled regret over campaigns with OPT > 0."""
    rd_s = np.array([o["rd"] / o["spend"] if o["spend"] > 0 else 0. for o in outs], dtype=float).mean()
    pr_s = np.array([o["pr"] / o["spend"] if o["spend"] > 0 else 0. for o in outs], dtype=float).mean()
    pos = [(c["opt"], o["conv"]) for c, o in zip(camps, outs) if c["opt"] > 0]
    reg = np.asarray([opt - n for opt, n in pos], dtype=float).sum() / np.asarray([opt for opt, _ in pos],
                                                                                  dtype=float).sum()
    return dict(rd_s=rd_s, pr_s=pr_s, regret=reg)


def criteo(res):
    camps = res["campaigns"]
    d, grid = res["data"], np.array(res["grid"])
    steps = {arm: common_step(camps, arm) for arm in ARMS}
    print("Appendix D: settings of the Criteo replay")
    print(f"  log: {d['n_impressions']:,} auctions of {d['n_campaigns']} campaigns")
    print(f"  training week: days {d['train_days'][0]}-{d['train_days'][1]}; "
          f"evaluation: days {d['eval_days'][0]}-{d['eval_days'][1]}; conversion window {d['window_days']} days")
    print(f"  kept: >= {d['min_train_conv']} training-week conversions and >= {d['min_eval_impr']:,} evaluation "
          f"auctions: {len(camps)} campaigns, {sum(c['opt'] > 0 for c in camps)} with OPT > 0")
    print(f"  K = {res['K']} windows, one report about every {d['eval_span_sec'] / res['K'] / 3600:g} hours; "
          f"H = {d['horizon_days']} day; median T = {np.percentile([c['T'] for c in camps], 50):,.0f}")
    print(f"  gub at confidence 1 - {d['delta']}: median {np.percentile([c['gub'] for c in camps], 50):.3f}; "
          f"median gamma on the evaluation period {np.median([c['gamma_eval'] for c in camps]):.3f}")
    print(f"  {grid.size} constant steps; chosen: "
          + ", ".join(f"{arm} {grid[i]:.3g}" for arm, i in steps.items())
          + f"; {sum(i == grid.size - 1 for i in steps.values())} of {len(steps)} at the largest step")
    print()
    rows = [("Wadap (ours)", [c["rows"]["ag"] for c in camps]),
            ("Wadap, constant step", [c["grid"]["pilot"][steps["pilot"]] for c in camps]),
            ("Wadap, gub = 1", [c["rows"]["ag_lit"] for c in camps]),
            ("Dual mirror descent", [c["grid"]["dmd"][steps["dmd"]] for c in camps]),
            ("P controller", [c["grid"]["pid"][steps["pid"]] for c in camps]),
            ("Castiglioni et al.", [c["grid"]["cast_rep"][steps["cast_rep"]] for c in camps]),
            ("Prediction-only pacing", [c["grid"]["pred"][steps["pred"]] for c in camps]),
            ("Break-even bid C vhat", [c["rows"]["breakeven"] for c in camps]),
            ("Unpaced bidding", [c["rows"]["full"] for c in camps])]
    print("Table 2: Criteo replay")
    print(f"  {'policy':<26}{'pooled regret':>15}{'final RD/S':>13}{'peak PR/S':>12}")
    for name, outs in rows:
        a = aggregate(camps, outs)
        print(f"  {name:<26}{a['regret']:>15.3f}{a['rd_s']:>13.4f}{a['pr_s']:>12.4f}")
    print()
    return steps


# Figure 1: WADAP against the other rows of Table 2, one panel per group of rows
STYLE = dict(ag=("#0b0b0b", "o", True, "-"), pilot=("#2a78d6", "s", False, (0, (5, 1.5))),
             ag_lit=("#eb6834", "^", True, (0, (1, 1.2))), dmd=("#1baf7a", "D", False, (0, (6, 1.5, 1, 1.5))),
             pid=("#eda100", "v", True, (0, (3, 1.2))), cast_rep=("#e87ba4", "X", True, (0, (8, 2))),
             pred=("#008300", "P", False, (0, (3, 1, 1, 1, 1, 1))), static=("#4a3aa7", "h", False, (0, (1.5, 1.5))))
PANELS = (("WADAP Variants", ("pilot", "ag_lit")),
          ("Report-Based Pacers", ("dmd", "pid", "cast_rep", "pred")),
          ("Static Bids", ("static",)))
# direct labels: curve -> (anchor point, label position in data coordinates)
LABELS = (dict(ag=(6, (.0078, .03)), pilot=(5, (.040, .125)), ag_lit=(5, (.020, .33))),
          dict(ag=(6, (.0078, .03)), dmd=(12, (.036, .30)), pid=(10, (.046, .23)),
               cast_rep=(8, (.0125, .45)), pred=(12, (.058, .165))),
          dict(ag=(6, (.0078, .03))))
NAME = dict(ag="WADAP (Ours)", pilot="WADAP, Constant Step", ag_lit=r"WADAP, $\bar\gamma=1$",
            dmd="Dual Mirror Descent", pid="P Controller", cast_rep="Castiglioni et al.",
            pred="Prediction-Only\nPacing")
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#d9d8d3"
YMAX, XLIM = .52, (.0072, .135)


def curves(res, steps):
    """Per curve: its points (final deficit, pooled regret) and the points of Table 2 on it."""
    camps = res["campaigns"]
    pt = lambda outs: (lambda a: (a["rd_s"], a["regret"]))(aggregate(camps, outs))  # noqa: E731
    out = {}
    for arm in ("ag", "ag_lit"):
        pts = [pt([c["alpha_c"][arm][i] for c in camps]) for i in range(len(res["alpha_c"]))]
        out[arm] = (pts, [(None, pts[res["alpha_c"].index(2.0)])])
    for arm in ARMS:
        pts = [pt([c["grid"][arm][i] for c in camps]) for i in range(len(res["grid"]))]
        out[arm] = (pts, [(None, pts[steps[arm]])])
    pts = [pt([c["kappa"][i] for c in camps]) for i in range(len(res["kappa"]))]
    out["static"] = (pts, [("Break-Even ($\\kappa=1$)", pts[res["kappa"].index(1.0)]), ("Unpaced Bidding", pts[-1])])
    return out


def criteo_figure(res, steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.labelcolor": INK, "pdf.fonttype": 42})
    cv = curves(res, steps)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), sharex=True, sharey=True)
    for k, (ax, (title, arms)) in enumerate(zip(axes, PANELS)):
        ax.grid(True, color=GRIDC, lw=.4)
        for z, arm in enumerate(arms + ("ag",)):       # WADAP on top
            col, mk, filled, ls = STYLE[arm]
            ours = arm == "ag"
            mfc = col if filled else "white"
            pts, marks = cv[arm]
            ax.plot(*zip(*pts), color=col, ls=ls, lw=2.0 if ours else 1.2, marker=mk, ms=4.0 if ours else 3.8,
                    mfc=mfc, mec=col, mew=.8, zorder=10 + z)
            for name, (x, y) in marks:
                ax.plot([x], [y], marker=mk, ms=9 if ours else 8.5, mfc=mfc, mec=INK, mew=1.1, ls="none",
                        zorder=30 + z)
                if name:
                    up = name.startswith("Break")
                    ax.annotate(name, (x, y), fontsize=7, color=INK, xytext=(8, 0) if up else (-6, 26),
                                textcoords="offset points", ha="left" if up else "right", va="center", zorder=40,
                                arrowprops=None if up else dict(arrowstyle="-", color=INK2, lw=.5, shrinkB=4))
            if arm in LABELS[k]:
                i, xy = LABELS[k][arm]
                ax.annotate(NAME[arm], pts[i], xytext=xy, textcoords="data", fontsize=7, color=INK, ha="left",
                            va="center", zorder=40, arrowprops=dict(arrowstyle="-", color=INK2, lw=.5, shrinkA=1,
                                                                    shrinkB=2))
        ax.set_title(title, fontsize=8, color=INK, pad=4)
        ax.set_xlabel(r"Final deficit $\mathrm{Def}/S$ (log)")
    # the static bid with kappa < 1 leaves the axes at the top
    axes[2].annotate("", xy=(.0084, YMAX + .005), xytext=(.0084, YMAX - .07), annotation_clip=False,
                     arrowprops=dict(arrowstyle="-|>", color=STYLE["static"][0], lw=1.0))
    axes[0].legend(handles=[Line2D([], [], color="none", marker="o", ms=8.5, mfc="white", mec=INK, mew=1.1,
                                   label="Table setting")],
                   loc="upper right", frameon=False, fontsize=7, handletextpad=.3, borderaxespad=.2)
    axes[0].set_xscale("log")
    axes[0].set_xlim(*XLIM)
    axes[0].set_ylim(-.075, YMAX)
    axes[0].set_ylabel("Pooled regret")
    fig.tight_layout(pad=.3, w_pad=.8)
    fig.savefig(os.path.join(FIGURES, "fig_real_frontier.pdf"))
    plt.close(fig)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    pair, slack = load("pair.json"), load("slack.json")
    synthetic(pair, slack)
    synthetic_figure(pair, slack)
    if os.path.exists(os.path.join(RESULTS, "criteo.json.gz")):
        res = load("criteo.json.gz")
        criteo_figure(res, criteo(res))
    print("figures written to", os.path.relpath(FIGURES, ROOT))


if __name__ == "__main__":
    main()
