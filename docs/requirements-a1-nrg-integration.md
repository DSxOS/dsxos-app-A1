# A1 fixes and NRG integration (A1-IMP)

Status: **FROZEN 2026-11-14**. The owner applied the A5 process and decisions ("same process and logic as A5") and
answered the A1-specific questions on 2026-11-14. Branch `feature/nrg-integration` off `main` `566b702`. Changes only as
numbered amendments in §6.

The reference is dsxos-app-A5 `docs/requirements-a5-improvements.md` (A5-IMP) and
`docs/requirements-client-credentials-glpk.md` (A5-AUTH). Requirements marked "as A5" mean the same as there.

## 1. Facts (main `566b702`)

- **F1.** A1 is a peak-shaving **correction**. It takes a long-term ESS plan (`essCurrentPowerPlanIdentifier`) and
  changes it by the smallest total amount (Σ|correction|) so that the PCC stays within the import and export limits.
  The corrected plan goes to `essResultPowerPlanIdentifier`.
- **F2.** It shares the A5 helper code and A5's audit findings:
  - no paging (500 values);
  - the start is not aligned;
  - one POST per value;
  - `essMinSafeLim` and `essEffectiveEndCharge` are read but unused (the end target is commented out);
  - `print`s, a debug file on every run, unpinned dependencies, no tests;
  - **no `packages.txt`, so the image has no GLPK.**
- **F3.** The login result is not checked (`TypeError` on NRG, 2026-11-13 22:00Z).
- **F4.** Model specifics:
  - ESS power and capacity are fixed numbers.
  - SoC is `E/capacity` with bounds 0–100 %; there is no SoC-window parameter and no efficiency.
  - The PCC limits are hard, so no solution means no plan, and the fallback branch is dead code.
- **F5. Bug:** the plan value used before the first plan point (`initial`) is taken in **W**, while the rest of the plan
  is in **kW** (`main.py` lines 157 and 164).
- **F6.** The config contract uses `clientID` in `example_config.yaml`, but the code reads `clientId`.

## 2. Owner decisions

| # | Decision |
|---|---|
| D1 | Apply A5-IMP and A5-AUTH, as for A4: login check, paging, aligned start, batch output, input/config errors, configurable values, code split, tests, pinned dependencies, cleanup, `packages.txt` with GLPK. |
| D2 | The start energy is the **total** stored energy in Wh. A SoC window `ess_min_soc`/`ess_max_soc` is enforced: new parameters, default 0/100 so that old configs keep working. Efficiencies default 0.95/0.95. |
| D3 | Soft PCC limits with a penalty; the overrun is reported. |
| D4 | **No end target** (A1 follows the long-term plan; only the correction is minimised). `essEffectiveEndCharge` is accepted and ignored. |
| D5 | Input on NRG: the **latest A5 plan** (a forecast on `B1_PIXII_704_WSet`). A1 runs about 30 s after A5. **Fallback:** pass the long-term plan through uncorrected. |
| D6 | ESS power and capacity from datapoints (`ess_max_P_dp_id`, `ess_max_E_dp_id`), falling back to the fixed numbers. |
| D7 | NRG: interval 60 s; output to the test datapoint `ess_p_a1_test`; own API credential. |

## 3. Requirements

- **R1. Shared behaviour (as A5).**
  - **Authentication:** as A5-AUTH; `clientID` is a legacy alias.
  - **Inputs:** paged forecasts, as A5-IMP R7. The long-term plan is the latest prognosis of
    `essCurrentPowerPlanIdentifier`. Its values are converted W → kW **including the value before the first plan
    point**, which fixes F5.
  - **Time grid, output and outcomes:** aligned grid (R8); batch output (R9); outcomes and exit codes (R10).
  - **Code:** structure and cleanup (R12–R13); pinned dependencies; `packages.txt` = `glpk-utils`.
- **R2. Common horizon.** The common horizon is that of the consumption forecast, the production forecast **and** the
  long-term plan. If it is shorter than `min_period`, use the fallback (R6).
- **R3. ESS and energy model.**
  - Same as A5-IMP R1–R2: total start energy; efficiencies; window `[ess_min_soc, ess_max_soc]`, widened to E0;
    `ch`/`dis` exclusive through a `Binary`; `|ch|, |dis| ≤ Pmax`.
  - No end target.
- **R4. Correction and PCC.**
  - `ess[k] = ch[k] − dis[k]` and `ess[k] − plan[k] = cUp[k] − cDown[k]`, with `cUp, cDown ≥ 0`.
  - Balance: `imp − exp = load + pv + ess`.
  - Limits: `imp ≤ pccImportLimitW/1000 + oImp` and `exp ≤ −pccExportLimitW/1000 + oExp` (soft). No import/export
    binaries are needed (there is no price).
- **R5. Objective (A1's goal, in kWh).** Minimise Σ_k Δt · ( `cUp + cDown` + `(oImp + oExp)·pccLimitPenaltyPerKWh` ).
  The result is reported as the total correction (kWh) and the overruns.
- **R6. Fallback.** On too little data or a failed solve, the long-term plan's readings at or after `start` are posted
  unchanged to the result datapoint: WARNING, exit 0. Without a long-term plan: ERROR, exit 1.
- **R7. Parameters.**
  - **Required:** `apiEndpoint`, `clientId` (or `clientID`), `token`, `consumptionPrognosisIdentifier`,
    `productionPrognosisIdentifier`, `essCurrentPowerPlanIdentifier`, `essEffectiveCharge`, `pccImportLimitW`,
    `pccExportLimitW`, `essResultPowerPlanIdentifier`, `interval`, `min_period`. Also either `ess_max_P_dp_id` or
    `essPowerLimitW`, and either `ess_max_E_dp_id` or `essMaxCharge`.
  - **Optional (default):** `ess_min_soc` [0], `ess_max_soc` [100], `essChargeEfficiency`/`essDischargeEfficiency`
    [0.95], `pccLimitPenaltyPerKWh` [1000], `solverTimeLimitS` [300], `maxInputAgeS` [900], `lokiUrl`,
    `writeDebugFile` [on-failure].
  - `essMinSafeLim`, `essEffectiveEndCharge` and `npPriceIdentifier` are accepted and ignored, with an INFO line.

## 4. Acceptance criteria

- **AC1 (pytest, Python 3.11 + GLPK).**
  - (a) Paging, the time grid, input errors, configuration errors, the `clientID` alias.
  - (b) A model case: a plan that breaks the import limit gets the smallest correction that fixes it (hand-checked
    numbers).
  - (c) A plan within all limits stays unchanged (correction 0).
  - (d) The SoC window is enforced, and a correction is limited by it.
  - (e) An impossible limit gives a soft overrun, which is reported.
  - (f) The initial value is converted to kW (the F5 regression).
  - (g) Fallback on too little data and on a failed solve passes the plan through; exit 1 without a plan; a failed login
    gives exit 1.
- **AC2.** `py_compile` passes; no `import *`; no assignment to `np`, `pd` or `value`.
- **AC3 (live, DSxOS-NRG).** With its own credential, input = A5's plan, output = `ess_p_a1_test`:
  - exit 0;
  - a corrected plan over the common horizon (more than 20 h, interval 60, aligned);
  - the PCC implied by load + PV + the corrected plan stays within 0..170 kW wherever the original plan exceeded it and
    the ESS can fix it;
  - SoC within 15–80 %;
  - the run is shorter than the cron period.
  - No device write.

## 5. Out of scope

Credentials from the site secrets store; the platform's core `A1` task; choosing which plan reaches the device.

## 6. Amendments

None yet.
