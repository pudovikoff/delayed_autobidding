"""Turn the Criteo Attribution log into one replay file per campaign (Appendix D).

  1. Read the needed columns of the log (cached as .npy on first use).
  2. Keep the campaigns with at least 50 converting impressions in the training week
     (days 0-6) and at least 2,000 impressions in the evaluation period (days 14-29).
  3. Fit a hashed logistic model for vhat on the training week; its regularisation is chosen
     on days 0-4 against days 5-6.
  4. Per campaign: price cap, training-week CPA and the reporting bound gub.
  5. Write the evaluation stream and the training-week stream of every kept campaign to
     data/criteo/campaigns/c<id>.npz, and the log counts to data/criteo/summary.json.

A conversion counts if it happens within W = 7 days of the impression. It is credited if
Criteo attributes it and it happens within H = 1 day. K = 64 report windows.
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "criteo")
LOG_FILE = os.path.join(DATA_DIR, "criteo_attribution_dataset.tsv.gz")
CACHE = os.path.join(DATA_DIR, "cache")
CAMPAIGN_DIR = os.path.join(DATA_DIR, "campaigns")

DAY = 86400
W_DAYS = 7
H_DAYS = 1
K = 64
TRAIN, FIT, VAL, EVAL = (0, 6), (0, 4), (5, 6), (14, 29)
MIN_TRAIN_CONV, MIN_EVAL_IMPR = 50, 2000
DELTA = 0.05          # gub holds with probability 1 - DELTA
GAMMA_MIN = 0.05
HASH_BITS = 22
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
CATS = [f"cat{i}" for i in range(1, 10)]
COLS = dict(timestamp=np.int32, campaign=np.int32, conversion=np.int8, conversion_timestamp=np.int32,
            attribution=np.int8, cost=np.float64, uid=np.int32, **{c: np.int32 for c in CATS})


def read_columns():
    paths = {c: os.path.join(CACHE, c + ".npy") for c in COLS}
    missing = [c for c, p in paths.items() if not os.path.exists(p)]
    if missing:
        os.makedirs(CACHE, exist_ok=True)
        parts = {c: [] for c in missing}
        for chunk in pd.read_csv(LOG_FILE, sep="\t", usecols=missing, compression="gzip", chunksize=2_000_000):
            for c in missing:
                x = chunk[c]
                if np.issubdtype(COLS[c], np.integer):
                    x = x.fillna(-999)
                parts[c].append(x.to_numpy().astype(COLS[c]))
        for c, v in parts.items():
            np.save(paths[c], np.concatenate(v))
    return {c: np.load(p, mmap_mode="r") for c, p in paths.items()}


def load():
    c = read_columns()
    k = dict(ts=np.asarray(c["timestamp"], dtype=np.int64),
             camp=np.asarray(c["campaign"], dtype=np.int64),
             conv=np.asarray(c["conversion"]) == 1,
             cts=np.asarray(c["conversion_timestamp"], dtype=np.int64),
             att=np.asarray(c["attribution"]) == 1,
             cost=np.asarray(c["cost"], dtype=np.float64),
             uid=np.asarray(c["uid"], dtype=np.int64),
             cats=c)
    k["day"] = k["ts"] // DAY
    k["delay"] = np.where(k["conv"], k["cts"] - k["ts"], -1)
    k["omega"] = k["conv"] & (k["delay"] <= W_DAYS * DAY)
    k["credited"] = k["conv"] & k["att"] & (k["delay"] <= H_DAYS * DAY)
    k["ucamp"], k["cinv"] = np.unique(k["camp"], return_inverse=True)
    return k


def days(k, span):
    return (k["day"] >= span[0]) & (k["day"] <= span[1])


# ------------------------------------------------------------------------------ model
_M1, _M2, _M3 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def _mix(z):
    """splitmix64 finaliser."""
    z = z + _M1
    z = (z ^ (z >> np.uint64(30))) * _M2
    z = (z ^ (z >> np.uint64(27))) * _M3
    return z ^ (z >> np.uint64(31))


def features(k, rows):
    """Hashed one-hot design: campaign, cat1-9 and hour of day."""
    fields = [k["camp"][rows]] + [np.asarray(k["cats"][c][rows], dtype=np.int64) for c in CATS] \
        + [(k["ts"][rows] // 3600) % 24]
    idx = np.empty((rows.size, len(fields)), dtype=np.int32)
    with np.errstate(over="ignore"):
        for f, x in enumerate(fields):
            key = (np.uint64(f + 1) << np.uint64(40)) | (x.astype(np.uint64) & np.uint64(0xFFFFFFFF))
            idx[:, f] = (_mix(key) >> np.uint64(64 - HASH_BITS)).astype(np.int32)
    return sp.csr_matrix((np.ones(idx.size), idx.ravel(), np.arange(0, idx.size + 1, len(fields), dtype=np.int64)),
                         shape=(rows.size, 1 << HASH_BITS))


def fit(X, y, C):
    return LogisticRegression(C=C, solver="lbfgs", max_iter=500, tol=1e-6).fit(X, y)


def fit_model(k):
    """Choose C by validation log-loss on days 0-4 -> 5-6, then refit on days 0-6."""
    y = k["omega"].astype(np.int8)
    fit_rows, val_rows = np.flatnonzero(days(k, FIT)), np.flatnonzero(days(k, VAL))
    Xf, Xv = features(k, fit_rows), features(k, val_rows)
    loss = {}
    for C in C_GRID:
        p = fit(Xf, y[fit_rows], C).predict_proba(Xv)[:, 1]
        loss[C] = log_loss(y[val_rows], np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])
        print(f"  C = {C:g}: validation log-loss {loss[C]:.6f}", flush=True)
    C = min(C_GRID, key=loss.get)
    train_rows = np.flatnonzero(days(k, TRAIN))
    return fit(features(k, train_rows), y[train_rows], C)


def predict(model, k, rows):
    return np.concatenate([model.predict_proba(features(k, rows[i:i + 2_000_000]))[:, 1]
                           for i in range(0, rows.size, 2_000_000)])


# ------------------------------------------------------------------------------ constants
def kish(cinv, uid, n):
    """Kish effective sample size of the rows of each campaign, clustered by user."""
    key = (cinv.astype(np.int64) << 32) | (uid & 0xFFFFFFFF)
    uk, m = np.unique(key, return_counts=True)
    cc = (uk >> 32).astype(np.int64)
    s1 = np.bincount(cc, weights=m.astype(np.float64), minlength=n)
    s2 = np.bincount(cc, weights=m.astype(np.float64) ** 2, minlength=n)
    return s1 ** 2 / np.maximum(s2, 1.0)


def train_constants(k, tr):
    """Price cap mbar, CPA and gub = gamma_train + Hoeffding margin, on the training week."""
    n = k["ucamp"].size
    ci, cost, om = k["cinv"][tr], k["cost"][tr], k["omega"][tr]
    mbar = np.zeros(n)
    np.maximum.at(mbar, ci, cost)
    n_om = np.bincount(ci[om], minlength=n)
    neff = kish(ci[om], k["uid"][tr][om], n)
    n_cr = np.bincount(ci[om & k["credited"][tr]], minlength=n)
    g = n_cr / np.maximum(n_om, 1)
    margin = np.sqrt(math.log(1.0 / DELTA) / (2 * np.maximum(neff, 1.0)))
    gub = np.where(n_om > 0, np.minimum(1.0, np.maximum(GAMMA_MIN, g + margin)), 1.0)
    cpa = np.bincount(ci, weights=cost, minlength=n) / np.maximum(n_om, 1)
    return dict(mbar=mbar, cpa=cpa, gub=gub, n_omega=n_om)


# ------------------------------------------------------------------------------ streams
def stream(k, rows, vhat, n_windows):
    """The rounds of one campaign in time order, cut to T = K L, with the round at which each
    credited conversion is reported (T if never) and H in rounds of this stream's clock."""
    n = rows.size
    T = n_windows * (n // n_windows)
    ts, cts, conv = k["ts"][rows], k["cts"][rows], k["conv"][rows]
    arrival = np.searchsorted(ts[:T], cts, side="left")
    arrival = np.minimum(np.where(conv, np.maximum(arrival, np.arange(n)), T), T)[:T]
    span = int(ts[T - 1]) - int(ts[0])
    return dict(vhat=vhat[:T], m=k["cost"][rows][:T], omega=k["omega"][rows][:T].astype(np.int8),
                chi=k["credited"][rows][:T].astype(np.int8), D=arrival - np.arange(T),
                K=n_windows, H=int(round(H_DAYS * DAY * T / max(span, 1))))


def by_campaign(k, rows):
    """Split row indices (in file order) by campaign."""
    rows = rows[np.argsort(k["cinv"][rows], kind="stable")]
    ci = k["cinv"][rows]
    cut = np.searchsorted(ci, np.arange(k["ucamp"].size + 1))
    return rows, cut


def main():
    t0 = time.time()
    k = load()
    tr, ev = days(k, TRAIN), days(k, EVAL)
    n = k["ucamp"].size
    n_conv_train = np.bincount(k["cinv"][tr & k["omega"]], minlength=n)
    n_eval = np.bincount(k["cinv"][ev], minlength=n)
    keep = (n_conv_train >= MIN_TRAIN_CONV) & (n_eval >= MIN_EVAL_IMPR)
    print(f"{k['ts'].size:,} impressions, {n} campaigns, {int(keep.sum())} kept", flush=True)

    model = fit_model(k)
    cst = train_constants(k, tr)
    ev_rows, ev_cut = by_campaign(k, np.flatnonzero(ev & keep[k["cinv"]]))
    tr_rows, tr_cut = by_campaign(k, np.flatnonzero(tr & keep[k["cinv"]]))
    ev_vhat, tr_vhat = predict(model, k, ev_rows), predict(model, k, tr_rows)

    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    for c in np.flatnonzero(keep):
        e, t = slice(ev_cut[c], ev_cut[c + 1]), slice(tr_cut[c], tr_cut[c + 1])
        ev_s = stream(k, ev_rows[e], ev_vhat[e], K)
        T = ev_s["m"].size
        # the training week gets about as many rounds per window as the evaluation period
        k_train = max(1, int(math.floor(K * (t.stop - t.start) / T + 0.5)))
        tr_s = stream(k, tr_rows[t], tr_vhat[t], k_train)
        np.savez_compressed(os.path.join(CAMPAIGN_DIR, f"c{int(k['ucamp'][c])}.npz"),
                            mbar=cst["mbar"][c], cpa=cst["cpa"][c], gub=cst["gub"][c],
                            n_train_conv=cst["n_omega"][c],
                            **{f"eval_{a}": v for a, v in ev_s.items()},
                            **{f"train_{a}": v for a, v in tr_s.items()})
    summary = dict(n_impressions=int(k["ts"].size), n_campaigns=int(n), n_kept=int(keep.sum()),
                   train_days=TRAIN, eval_days=EVAL, window_days=W_DAYS, horizon_days=H_DAYS,
                   min_train_conv=MIN_TRAIN_CONV, min_eval_impr=MIN_EVAL_IMPR, delta=DELTA,
                   eval_span_sec=(EVAL[1] - EVAL[0] + 1) * DAY)
    with open(os.path.join(DATA_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
