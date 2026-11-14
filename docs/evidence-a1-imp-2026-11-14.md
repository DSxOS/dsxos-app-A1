# A1-IMP evidence (2026-11-14)

Independent validator runs; commands and outputs abbreviated from the validator reports.

## AC1, AC2 — efc55b0
- Host Taltech-ps-dt, throwaway `python:3.11-slim` (Python 3.11.18, pytest 8.3.5), clone at efc55b0.
- `pytest -v` → `24 passed in 0.42s`, rc 0; `python -m py_compile *.py` rc 0; greps for `import *` and assignments to
  np/pd/value → no match (rc 1). Cleanup: containers, /tmp dir and image removed.

## AC3 — live DSxOS-NRG (site main 8ef2866, container code md5 = efc55b0)
- Runs 13:45:40Z and 14:00:40Z: exit 0 (no non-zero/ERROR lines), ~4 s.
- 14:00:40Z: `solved (ok/optimal): 1881 steps 2026-11-14T14:00:00Z .. 2026-11-15T21:20:00Z, total correction 0.00 kWh`;
  `corrected plan posted to 'ess_p_a1_test'`.
- DB: input = A5's plan `B1_PIXII_704_WSet` prognosis 46605 (1881 readings, 230 nonzero steps, ±113 kW); output
  `ess_p_a1_test` 46606: 1881 readings, 60 s, aligned start, horizon 31.3 h; equal to the input (max |diff| 1e-9 W).
- SoC 15.00–43.78 %; implied PCC (load + PV + plan) within 0..170 kW at every step, input and output.
- AC3 PASS. Caveat: A5's plan already met the PCC limits, so a non-trivial correction is exercised only by pytest.
