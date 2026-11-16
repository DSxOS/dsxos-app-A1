# OPD: evidence, 2026-11-16 (A1)

Requirements: [requirements-opd-operational-datapoints.md](requirements-opd-operational-datapoints.md). Implementation 951ce81.

| AC | Verdict |
|---|---|
| AC1 | PASS |
| AC2 | PASS |
| AC3 | PASS (DSxOS-NRG, see below) |

Independent validator (NRG VM `dsxos`, python:3.11-slim + glpk-utils, `git archive 951ce81`): py_compile 0; pytest 86 passed, exit 0. AC1 (a)-(f) PASS, AC2 PASS. In-test old-model copy matches `89bdc62:model.py`. Spot check: 60 random equal-limit L≤0 cases — same status and objective (1e-5); 4 had alternative optimal plans with identical objective (GLPK tie-break). Solver probe (1900×60 s steps, Pch 50/Pdis 30 kW, U 150, L +15 kW): new model 28.4 s ok/optimal vs old 49.8 s; an artificially infeasible variant hits tmlim (61 s, no incumbent) in BOTH old and new → existing fallback. GLPK tmlim honoured.

## AC3: live on DSxOS-NRG (site `8c71b7f`; update-config 21:09:40Z and deploy 21:10:56Z, both Succeeded; first runs 21:15Z)

- **Operator datapoints**, created by update-config with `initialReadingValue`: `ess_max_P_charge` 50000 W,
  `ess_max_P_discharge` 30000 W, `ess_min_soc` 15, `ess_max_soc` 80. `pcc_p_max_p` is 150000 and `pcc_p_min_p` 15000.
- **Logs.** Each app logged all 7 limits with `(dp <id>)` sources and finished with exit 0:
  - A5 at 21:15:10: `ok/optimal`, 1531 steps, about 4 s, plan posted to `B1_PIXII_704_WSet`.
  - A4 at 21:15:10: `ok/optimal`, 1806 steps, about 1 s.
  - A1 at 21:15:40: `PCC window [15.0, 150.0] kW`, `ok/optimal`, 1531 steps, about 7 s, total correction 0 kWh. This is
    A1's first successful run since `3f93fd1` (2026-11-15).
- **Plans in the DB** (latest prognoses 49236, 49237, 49235):
  - A5 and A1: −30.00 … +50.00 kW;
  - A4: 0 kW (idle).
- **SoC recomputed from the plans** (E0 294.0 kWh, capacity 438 kWh, η 0.95):
  - A5 and A1: 16.92 … 80.00 %;
  - A4: 67.12 % throughout;
  - all end at 294.0 kWh.
- So the plans stay within the 15–80 % window and the [−30, +50] kW caps.
