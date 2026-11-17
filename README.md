# dsxos-app-A1 - Peak-shaving correction

A1 takes a long-term ESS plan (for example A5's economic plan) and changes it **as little as possible** so that the
PCC power stays within its limit window (upper and lower limit). The corrected plan is written as a prognosis of a datapoint
(W, positive = charging). A1 runs as a DSxOS python app on its `cron`, shortly after the app that writes the
long-term plan.

Requirements: [docs/requirements-a1-nrg-integration.md](docs/requirements-a1-nrg-integration.md) (A1-IMP; it reuses
the decisions of dsxos-app-A5 A5-IMP and A5-AUTH).

## Model

Per time step (`interval`):
- **ESS:** charge in [0, Pch] and discharge in [0, Pdis] (separate limits), never both. Stored energy `E[k+1] = E[k] + (ch·η_ch − dis/η_dis)·Δt`,
  within [0, capacity] (kWh; the capacity is the **usable** capacity, so the SoC limits are already embedded in it and in the energy input).
- **Correction:** `ess − plan = cUp − cDown`.
- **PCC:** net = load + pv + ess, import positive, pv negative. The window `L − oL ≤ net ≤ U + oU` is soft (L = `pccExportLimitW`, signed: negative = export allowed, 0 = no export, positive = minimum import): an overrun that cannot be fixed
  is penalised and reported as a WARNING ("import" or "below the lower limit").
- **Minimised (kWh):** Σ(cUp + cDown)·Δt + `pccLimitPenaltyPerKWh` · overrun. No end target: A1 follows the
  long-term plan.

The long-term plan is step-held onto the time grid (start = run time rounded down to `interval`); before its first
point its first value is used. With too little input data, or no usable solver result, A1 passes the long-term plan
through uncorrected (fallback). Exit 0 = corrected plan or fallback posted; 1 = error or no long-term plan.

## Configuration (`params`)

See [example_config.yaml](example_config.yaml).

**Operational parameters** (the rows below marked "datapoint id or number"): a string is a datapoint id whose latest value is read at every run (missing or no value = input error, exit 1, nothing posted; no staleness check); a number is a constant. The resolved values and their sources (`dp <id>` or `const`) are logged once per run. Checked per run: limits and capacity > 0, `pccExportLimitW ≤ pccImportLimitW` (a violation is an input error for a datapoint source and a configuration error for a number). Stale or removed params (see OPD R5) are a configuration error that names the replacement; `ess_min_soc` and `ess_max_soc` are removed (SFU-A1: the SoC limits are embedded in `ess_max_E_dp_id` and `essEffectiveCharge`) and a config that still has them fails.

| Param | Unit | Default | Meaning |
|---|---|---|---|
| `apiEndpoint`, `clientId` (legacy `clientID`), `token` | | required | Data server API and credential (client ID + secret) |
| `consumptionPrognosisIdentifier` / `productionPrognosisIdentifier` | W | required | Load / PV forecast (PV negative) |
| `essCurrentPowerPlanIdentifier` | W | required | Datapoint whose latest prognosis is the long-term plan |
| `essResultPowerPlanIdentifier` | W | required | Datapoint that receives the corrected plan |
| `pccImportLimitW` | W | required | Upper limit of the net PCC power (import positive): datapoint id or number |
| `pccExportLimitW` | W | required | Lower limit of the net PCC power, signed: negative = export allowed, 0 = no export, positive = minimum import. Datapoint id or number |
| `ess_max_P_charge_dp_id`, `ess_max_P_discharge_dp_id` | W | required | Charge / discharge power limit, positive magnitudes (the same datapoint or number may be given for both): datapoint id or number |
| `ess_max_E_dp_id` | Wh | required | **Usable** ESS capacity (gross x (max SoC - min SoC)): datapoint id or number |
| `essEffectiveCharge` | Wh | required | Datapoint: **usable** energy now, in [0, capacity] (at most `maxInputAgeS` old). Used directly as clamp(u, 0, capacity); a value outside the range is clamped with a WARNING |
| `interval`, `min_period` | s | required | Time step; minimum common horizon |
| `essChargeEfficiency`, `essDischargeEfficiency` | | 0.95 | Efficiencies |
| `pccLimitPenaltyPerKWh` | | 1000 | Penalty per kWh of PCC overrun |
| `solverTimeLimitS`, `maxInputAgeS` | s | 300, 900 | GLPK limit; max age of the stored-energy reading |
| `lokiUrl`, `writeDebugFile` | | Loki on localhost, `on-failure` | Logging; debug file `on-failure`/`always`/`never` |

## Tests

`pip install -r requirements-dev.txt && pytest` (the model tests need `glpsol`, apt package `glpk-utils`, see
`packages.txt`).
