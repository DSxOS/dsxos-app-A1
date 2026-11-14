# dsxos-app-A1 - Peak-shaving correction

A1 takes a long-term ESS plan (for example A5's economic plan) and changes it **as little as possible** so that the
PCC power stays within the import and export limits. The corrected plan is written as a prognosis of a datapoint
(W, positive = charging). A1 runs as a DSxOS python app on its `cron`, shortly after the app that writes the
long-term plan.

Requirements: [docs/requirements-a1-nrg-integration.md](docs/requirements-a1-nrg-integration.md) (A1-IMP; it reuses
the decisions of dsxos-app-A5 A5-IMP and A5-AUTH).

## Model

Per time step (`interval`):
- **ESS:** charge/discharge in [0, Pmax], never both. Stored energy `E[k+1] = E[k] + (ch·η_ch − dis/η_dis)·Δt`,
  within [`ess_min_soc`, `ess_max_soc`] % of the capacity (widened to the start value if it is already outside).
- **Correction:** `ess − plan = cUp − cDown`.
- **PCC:** import − export = load + pv + ess, with pv negative. The limits are soft: an overrun that cannot be fixed
  is penalised and reported as a WARNING.
- **Minimised (kWh):** Σ(cUp + cDown)·Δt + `pccLimitPenaltyPerKWh` · overrun. No end target: A1 follows the
  long-term plan.

The long-term plan is step-held onto the time grid (start = run time rounded down to `interval`); before its first
point its first value is used. With too little input data, or no usable solver result, A1 passes the long-term plan
through uncorrected (fallback). Exit 0 = corrected plan or fallback posted; 1 = error or no long-term plan.

## Configuration (`params`)

See [example_config.yaml](example_config.yaml).

| Param | Unit | Default | Meaning |
|---|---|---|---|
| `apiEndpoint`, `clientId` (legacy `clientID`), `token` | | required | Data server API and credential (client ID + secret) |
| `consumptionPrognosisIdentifier` / `productionPrognosisIdentifier` | W | required | Load / PV forecast (PV negative) |
| `essCurrentPowerPlanIdentifier` | W | required | Datapoint whose latest prognosis is the long-term plan |
| `essResultPowerPlanIdentifier` | W | required | Datapoint that receives the corrected plan |
| `ess_max_P_dp_id` or `essPowerLimitW` | W | one required | ESS power limit: datapoint, or a fixed number |
| `ess_max_E_dp_id` or `essMaxCharge` | Wh | one required | ESS capacity: datapoint, or a fixed number |
| `essEffectiveCharge` | Wh | required | Datapoint: total stored energy now (at most `maxInputAgeS` old) |
| `ess_min_soc`, `ess_max_soc` | % | 0, 100 | SoC window |
| `pccImportLimitW`, `pccExportLimitW` | W | required | PCC limits; export as a number ≤ 0 (0 = no export) |
| `interval`, `min_period` | s | required | Time step; minimum common horizon |
| `essChargeEfficiency`, `essDischargeEfficiency` | | 0.95 | Efficiencies |
| `pccLimitPenaltyPerKWh` | | 1000 | Penalty per kWh of PCC overrun |
| `solverTimeLimitS`, `maxInputAgeS` | s | 300, 900 | GLPK limit; max age of the stored-energy reading |
| `lokiUrl`, `writeDebugFile` | | Loki on localhost, `on-failure` | Logging; debug file `on-failure`/`always`/`never` |
| `essMinSafeLim`, `essEffectiveEndCharge`, `npPriceIdentifier` | | | Accepted, ignored |

## Tests

`pip install -r requirements-dev.txt && pytest` (the model tests need `glpsol`, apt package `glpk-utils`, see
`packages.txt`).
