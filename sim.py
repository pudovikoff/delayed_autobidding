"""Simulator of one autobidding campaign with delayed, aggregated and censored conversion reports.

A campaign is a sequence of T second-price auctions. In round t the bidder sees the predicted
conversion probability vhat_t, bids mu * vhat_t, and wins if the bid reaches the price m_t; it
then pays m_t. A won auction converts (omega_t = 1) with probability v_t. The bidder never sees
omega_t: it only receives K window totals of credited conversions (chi_t = 1), each conversion
arriving D_t rounds after its auction.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    T: int = 20000            # rounds
    C: float = 2.0            # target cost per conversion
    K: int = 200              # report windows
    H: int = 100              # attribution horizon, rounds
    rho: float = 0.85         # probability that a conversion is credited
    delay_mean: float = 20.0  # mean of the geometric delay
    v_lo: float = 0.01        # v ~ U[v_lo, v_hi]
    v_hi: float = 0.10
    pred_sigma: float = 0.30  # vhat = v exp(sigma Z - sigma^2/2)
    kappa_lo: float = 0.40    # m = C v kappa, kappa ~ U[kappa_lo, kappa_hi]
    kappa_hi: float = 1.80
    family: str = "reference"  # "reference" or "pair"
    pair_branch: str = "B"    # the pair I_A / I_B of the lower bound
    pair_s: float = 0.20
    pair_eps: float = 0.05
    pair_v0: float = 0.50
    pair_gub: float = 0.90

    @property
    def L(self):
        assert self.T % self.K == 0
        return self.T // self.K


def make_instance(seed, cfg):
    """One campaign of the reference law."""
    rng = np.random.default_rng(seed)
    T, s = cfg.T, cfg.pred_sigma
    v = rng.uniform(cfg.v_lo, cfg.v_hi, T)
    vhat = np.clip(v * np.exp(s * rng.standard_normal(T) - 0.5 * s * s), 1e-6, 1.0)
    m = cfg.C * v * rng.uniform(cfg.kappa_lo, cfg.kappa_hi, T)
    omega = (rng.random(T) < v).astype(np.int8)
    D = rng.geometric(1.0 / (1.0 + cfg.delay_mean), T) - 1
    chi = ((D <= cfg.H) & (rng.random(T) < cfg.rho)).astype(np.int8)
    return dict(v=v, vhat=vhat, m=m, omega=omega, D=D, chi=chi)


def pair_instance(seed, cfg):
    """One campaign of I_A or I_B: vhat = v0, two prices, delay uniform on {0, ..., H}.

    I_A converts with probability v0 (1 - s) and credits with probability gub; I_B converts
    with probability v0 and credits with probability (1 - s) gub. Both look the same to a bidder.
    """
    rng = np.random.default_rng(seed)
    T, C, v0, eps = cfg.T, cfg.C, cfg.pair_v0, cfg.pair_eps
    cheap = rng.random(T) < 0.5
    m = np.where(cheap, 0.5 * C * v0, (1.5 + 2.0 * eps) * C * v0)
    if cfg.pair_branch == "A":
        v, gamma = np.full(T, v0 * (1.0 - cfg.pair_s)), cfg.pair_gub
    else:
        v, gamma = np.full(T, v0), cfg.pair_gub * (1 - cfg.pair_s)
    omega = (rng.random(T) < v).astype(np.int8)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7101]))
    D = rng.integers(0, cfg.H + 1, T)
    chi = (rng.random(T) < gamma).astype(np.int8)
    return dict(v=v, vhat=np.full(T, v0), m=m, omega=omega, D=D, chi=chi)


def price_cap(cfg):
    """Largest price of the law, mbar."""
    if cfg.family == "pair":
        return (1.5 + 2 * cfg.pair_eps) * cfg.C * cfg.pair_v0
    return cfg.C * cfg.v_hi * cfg.kappa_hi


# ------------------------------------------------------------------------------ comparators
def threshold_prefixes(m, vhat):
    """Rounds sorted by m / vhat, and the ends of the groups of equal ratio.

    A fixed multiplier mu wins exactly the rounds with m / vhat <= mu, so the candidate win
    sets are the prefixes of this order that end at a group boundary.
    """
    ratio = np.divide(m, vhat, out=np.full_like(m, np.inf, dtype=float), where=vhat > 0)
    ratio = np.where(m == 0, 0., ratio)
    order = np.argsort(ratio, kind="stable")
    rs = ratio[order]
    ends = np.r_[np.flatnonzero(rs[:-1] != rs[1:]) + 1, len(rs)]
    ends = ends[np.isfinite(rs[ends - 1])]
    take = np.r_[0, ends] if len(rs) == 0 or rs[0] > 0 else ends
    lo = np.array([rs[i - 1] if i else 0. for i in take])
    hi = np.array([rs[i] if i < len(rs) else np.inf for i in take])
    return order, take, lo, hi


def benchmark(inst, cfg, B):
    """OPT: conversions of the best fixed multiplier that meets the budget and the target."""
    order, take, lo, hi = threshold_prefixes(inst["m"], inst["vhat"])
    spend = np.concatenate([[0.0], np.cumsum(inst["m"][order])])[take]
    conv = np.concatenate([[0.0], np.cumsum(inst["omega"][order].astype(float))])[take]
    ok = (lo < hi) & (spend <= B) & (spend <= cfg.C * conv)
    return float(conv[int(np.argmax(np.where(ok, conv, -1.0)))]) if ok.any() else 0.0


def population_curve(cfg, seeds):
    """Expected conversions and spend per round of every fixed multiplier, on the pooled seeds."""
    insts = [make_instance(sd, cfg) for sd in seeds]
    v, vh, m = (np.concatenate([x[k] for x in insts]) for k in ("v", "vhat", "m"))
    order, take, _, _ = threshold_prefixes(m, vh)
    Ev = np.concatenate([[0.0], np.cumsum(v[order])])[take] / len(v)
    Em = np.concatenate([[0.0], np.cumsum(m[order])])[take] / len(v)
    return Ev, Em


def optpop(Ev, Em, c, T):
    """OPT_pop at target c: best mixture of fixed multipliers, i.e. the upper concave envelope
    of the points (E[spend] - c E[conv], E[conv]) at zero, times T."""
    h = Em - c * Ev
    best = float(np.max(np.where(h <= 0, Ev, -1.0)))
    hull = []
    for x, y in sorted(zip(h.tolist(), Ev.tolist())):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append((x, y))
    for (x1, y1), (x2, y2) in zip(hull, hull[1:]):
        if x1 <= 0.0 <= x2 and x2 > x1:
            best = max(best, y1 + (y2 - y1) * (0.0 - x1) / (x2 - x1))
    return best * T


# ------------------------------------------------------------------------------ policies
# wadap          WADAP with a constant step: q = clip(eta [Uhat]_+) at the end of each window
# wadap-ag       WADAP with the AdaGrad step eta_k = alpha / sqrt(G0^2 + sum_{j<=k} g_j^2)
# pred-pacer     the same update on the ledger S - C sum z_t vhat_t (no reports)
# dmd            dual mirror descent: q = clip(q + eta (Uhat_k - Uhat_{k-1}))
# pid-cpa        multiplicative P controller on the reported CPA
# cast-rep       primal-dual method of Castiglioni et al. (2024) fed with the reports
# breakeven      bid C vhat_t
# full-throttle  bid (C + 1/q_lo) vhat_t
Q_ARMS = ("wadap", "wadap-ag", "pred-pacer", "dmd")


def run_campaign(alg, inst, cfg, B, q_lo, q_hi, eta, gamma_tilde=1.0, g0=None, alpha=1.0, money_unit=1.0):
    """Play one campaign. Reports are read at the credit level gamma_tilde: Uhat = S - (C/gamma_tilde) R.

    g0 is the AdaGrad offset (default: the price cap). money_unit is the amount of money in
    which the fixed constants of pid-cpa and cast-rep are stated; it makes them currency-free.
    """
    T, C, L, K = cfg.T, cfg.C, cfg.L, cfg.K
    G0 = price_cap(cfg) if g0 is None else float(g0)
    G0sq = G0 * G0
    vhat, m, omega = inst["vhat"].tolist(), inst["m"].tolist(), inst["omega"].tolist()
    # window in which the conversion of round t is reported (-1: never)
    arr = np.arange(T) + inst["D"]
    creditable = (inst["chi"] == 1) & (arr < T)
    report_window = np.where(creditable, np.minimum(arr // L, K - 1), -1).tolist()

    counts = [0] * K
    S = 0.0          # spend
    N = 0            # true conversions won
    R = 0            # conversions reported so far
    U = Uhat = 0.0   # true and observed ledgers S - C N and S - (C/gamma_tilde) R
    PR = 0.0         # largest U so far
    max_gap = -math.inf
    pred_conv = 0.0
    gsq, Uhat_prev, Uhat_prev_dmd = 0.0, 0.0, 0.0
    q = q_lo
    mu_min, mu_max = C + 1.0 / q_hi, C + 1.0 / q_lo
    mu = C if alg == "breakeven" else mu_max

    cast_lam = cast_mu = 0.0
    cast_rho = B / T
    cast_etaB = 1.0 / (cast_rho * math.sqrt(T) * money_unit)
    cast_floor = 1.0 / (T * money_unit)
    if alg == "cast-rep":
        mu = 1.0 / cast_floor

    for t in range(T):
        b_raw = mu * vhat[t]
        rem = B - S
        b = b_raw if b_raw <= rem else rem        # never bid more than the remaining budget
        z = 1 if (b > 0.0 and b >= m[t]) else 0
        if z:
            S += m[t]
            pred_conv += vhat[t]
            if omega[t]:
                N += 1
                w = report_window[t]
                if w >= 0:
                    counts[w] += 1

        boundary = (t + 1) % L == 0
        if boundary:
            R += counts[t // L]
        U = S - C * N
        Uhat = S - C * pred_conv if alg == "pred-pacer" else S - (C / gamma_tilde) * R
        if U > PR:
            PR = U
        if U - Uhat > max_gap:
            max_gap = U - Uhat
        if boundary:
            g = abs(Uhat - Uhat_prev)
            Uhat_prev = Uhat
            gsq += g * g

        if alg in ("wadap", "wadap-ag", "pred-pacer"):
            if boundary:
                step = alpha / math.sqrt(G0sq + gsq) if alg == "wadap-ag" else eta
                q = min(q_hi, max(q_lo, step * max(Uhat, 0.0)))
        elif alg == "dmd":
            if boundary:
                q = min(q_hi, max(q_lo, q + eta * (Uhat - Uhat_prev_dmd)))
                Uhat_prev_dmd = Uhat
        elif alg == "cast-rep":
            c_t = m[t] if z else 0.0
            h = c_t - ((C / gamma_tilde) * counts[t // L] if boundary else 0.0)
            cast_lam = min(1.0 / cast_rho, max(0.0, cast_lam + cast_etaB * (c_t - cast_rho)))
            cast_mu = max(0.0, cast_mu + eta * h)
            mu = (1.0 + cast_mu * C) / max(cast_lam + cast_mu, cast_floor)
        elif alg == "pid-cpa":
            if boundary:
                err = (C * R / gamma_tilde - S) / max(S, money_unit)
                mu = min(mu_max, max(mu_min, mu * math.exp(eta * err)))
        if alg in Q_ARMS:
            mu = (1.0 + q * C) / q                # the bid (C + 1/q) vhat_t

    return dict(spend=S, conv=N, rd=max(U, 0.0), pr=max(PR, 0.0), reported=R,
                pred_conv=pred_conv, Uhat_T=Uhat, max_gap=max_gap)
