# OPD: evidence, 2026-11-16 (A1)

Requirements: [requirements-opd-operational-datapoints.md](requirements-opd-operational-datapoints.md). Implementation 951ce81.

| AC | Verdict |
|---|---|
| AC1 | PASS |
| AC2 | PASS |
| AC3 | pending (live on DSxOS-NRG after the site config and pin update) |

Independent validator (NRG VM `dsxos`, python:3.11-slim + glpk-utils, `git archive 951ce81`): py_compile 0; pytest 86 passed, exit 0. AC1 (a)-(f) PASS, AC2 PASS. In-test old-model copy matches `89bdc62:model.py`. Spot check: 60 random equal-limit L≤0 cases — same status and objective (1e-5); 4 had alternative optimal plans with identical objective (GLPK tie-break). Solver probe (1900×60 s steps, Pch 50/Pdis 30 kW, U 150, L +15 kW): new model 28.4 s ok/optimal vs old 49.8 s; an artificially infeasible variant hits tmlim (61 s, no incumbent) in BOTH old and new → existing fallback. GLPK tmlim honoured.
