# Autobidding with delayed, aggregated and censored constraint feedback

Code for the experiments of the paper (Section 7, Appendices C and D).

| Path | What it does |
|---|---|
| `sim.py` | Campaign simulator: second-price auctions, delayed and censored window reports, WADAP and the baseline pacers, the comparators OPT and OPT_pop |
| `synthetic/pair.py` | Section 7.1: growth with the horizon and the lower bound on the pair I_A, I_B |
| `synthetic/slack.py` | Section 7.1: cost of a loose reporting bound on the reference campaign |
| `criteo/` | Section 7.2: download the Criteo log, build one stream per campaign, replay the policies |
| `report.py` | Prints the numbers of Section 7.1, Appendices C and D and Table 2; draws Figures 1 and 2 |
| `results/` | Outputs of the runs reported in the paper |
| `tests/` | Unit tests of the simulator |

## Quick check

With Python 3.14:

```
make setup      # virtual environment with the versions in requirements.txt
make test
make report     # a few seconds, reads results/
```

`make report` prints the tables so they can be compared with the paper. It also writes
`figures/fig_real_frontier.pdf` (Figure 1) and `figures/fig_synth.pdf` (Figure 2).

## Rerunning the experiments

```
make synthetic  # about 1 minute
make criteo     # downloads the log (653 MB); about 15 minutes to prepare and 2 to replay on 8 cores
make report
```

Every run is deterministic, and a rerun reproduces the files in `results/`.

- **Synthetic campaigns.** Seeds 1000–1099.
- **Bootstrap.** 2,000 resamples, seed 68120.
- **Criteo replay.**
  - Training week: days 0–6. Evaluation period: days 14–29.
  - Target C: each campaign's CPA on its training week.
  - Reports: K = 64 windows, attribution horizon H = 1 day.
  - `criteo/replay.py` lists the policy constants.

## Data

The replay uses the Criteo Attribution Modeling for Bidding dataset of Diemert et al.,
"Attribution Modeling Increases Efficiency of Bidding in Display Advertising" (AdKDD 2017).
It is licensed CC BY-NC-SA 4.0 and is not included here. `criteo/download.py` fetches it from
Criteo's mirror on the Hugging Face Hub (`criteo/criteo-attribution-dataset`) and checks its
SHA-256.
