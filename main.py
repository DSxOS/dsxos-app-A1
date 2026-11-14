"""A1 peak-shaving correction: corrects a long-term ESS plan as little as possible so that the PCC stays within limits.

Run by the platform's python-app scheduler with the site's config (default /app/config.yaml).
Requirements: docs/requirements-a1-nrg-integration.md (A1-IMP).
Exit codes: 0 = corrected plan (or the long-term plan as fallback) posted, 1 = configuration/input error or nothing to post.
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

import yaml

import model as a1model
import query_utils
import Util
from debug import debug_model
from logger import setup_logger

logger = logging.getLogger("A1Runner")

DEFAULTS = {
    "ess_min_soc": 0,
    "ess_max_soc": 100,
    "essChargeEfficiency": 0.95,
    "essDischargeEfficiency": 0.95,
    "pccLimitPenaltyPerKWh": 1000.0,
    "solverTimeLimitS": 300,
    "maxInputAgeS": 900,
    "lokiUrl": "http://localhost:3100/loki/api/v1/push",
    "writeDebugFile": "on-failure",
    "ess_max_P_dp_id": None,
    "ess_max_E_dp_id": None,
    "essPowerLimitW": None,
    "essMaxCharge": None,
}
REQUIRED = [
    "apiEndpoint", "clientId", "token",
    "consumptionPrognosisIdentifier", "productionPrognosisIdentifier", "essCurrentPowerPlanIdentifier",
    "essEffectiveCharge", "pccImportLimitW", "pccExportLimitW",
    "essResultPowerPlanIdentifier", "interval", "min_period",
]
IGNORED = ["essMinSafeLim", "essEffectiveEndCharge", "npPriceIdentifier"]


class ConfigError(Exception):
    pass


class InputError(Exception):
    pass


def empty(v):
    return v is None or v == ""


def load_params(raw):
    """Merge defaults into params and validate (A1-IMP R7). Raises ConfigError."""
    if not isinstance(raw, dict) or not isinstance(raw.get("params"), dict):
        raise ConfigError("config has no 'params' section")
    params = dict(DEFAULTS)
    params.update(raw["params"])
    if empty(params.get("clientId")) and not empty(params.get("clientID")):
        params["clientId"] = params["clientID"]          # legacy spelling
    missing = [k for k in REQUIRED if empty(params.get(k))]
    if empty(params["ess_max_P_dp_id"]) and empty(params["essPowerLimitW"]):
        missing.append("ess_max_P_dp_id or essPowerLimitW")
    if empty(params["ess_max_E_dp_id"]) and empty(params["essMaxCharge"]):
        missing.append("ess_max_E_dp_id or essMaxCharge")
    if missing:
        raise ConfigError(f"missing params: {', '.join(missing)}")
    checks = [
        (0 < float(params["essChargeEfficiency"]) <= 1, "essChargeEfficiency must be in (0, 1]"),
        (0 < float(params["essDischargeEfficiency"]) <= 1, "essDischargeEfficiency must be in (0, 1]"),
        (float(params["pccLimitPenaltyPerKWh"]) >= 0, "pccLimitPenaltyPerKWh must be >= 0"),
        (float(params["solverTimeLimitS"]) > 0, "solverTimeLimitS must be > 0"),
        (float(params["maxInputAgeS"]) > 0, "maxInputAgeS must be > 0"),
        (int(params["interval"]) > 0, "interval must be > 0"),
        (int(params["min_period"]) >= 0, "min_period must be >= 0"),
        (0 <= float(params["ess_min_soc"]) < float(params["ess_max_soc"]) <= 100,
         "need 0 <= ess_min_soc < ess_max_soc <= 100"),
        (float(params["pccExportLimitW"]) <= 0, "pccExportLimitW must be <= 0 (export as a negative limit)"),
        (params["writeDebugFile"] in ("on-failure", "always", "never"),
         "writeDebugFile must be on-failure, always or never"),
    ]
    errors = [msg for ok, msg in checks if not ok]
    if errors:
        raise ConfigError("; ".join(errors))
    for name in IGNORED:
        if not empty(params.get(name)):
            logger.info(f"param {name} is ignored (not used by A1).")
    return params


def read_value(identifier, param, now=None, max_age_s=None):
    """Last value of a datapoint, with clear errors. Raises InputError."""
    dp = query_utils.get_datapoint(identifier)
    if dp is None:
        raise InputError(f"reading datapoint '{identifier}' ({param}) failed")
    if not dp:
        raise InputError(f"datapoint '{identifier}' ({param}) does not exist")
    val = dp[0].get("lastReadingValue")
    if val is None:
        raise InputError(f"datapoint '{identifier}' ({param}) has no value")
    if max_age_s is not None:
        last_time = dp[0].get("lastReadingTime")
        if not last_time:
            raise InputError(f"datapoint '{identifier}' ({param}) has no reading time")
        age = (now - Util.parse_time(last_time)).total_seconds()
        if age > max_age_s:
            raise InputError(f"datapoint '{identifier}' ({param}) is {int(age)} s old (max {int(max_age_s)} s)")
    return float(val)


def value_or_datapoint(params, dp_param, static_param):
    """ESS limit from its datapoint if configured, otherwise the fixed number (A1-IMP D6). In the param's unit."""
    if not empty(params.get(dp_param)):
        return read_value(params[dp_param], dp_param)
    return float(params[static_param])


def read_forecast(identifier, param):
    readings = query_utils.get_last_prognosis_readings(identifier)
    if readings is None:
        raise InputError(f"reading the forecast of '{identifier}' ({param}) failed")
    return [{"time": r["time"], "value": r["value"]} for r in readings if r.get("value") is not None]


def build_series(load, pv, plan, start, interval_s, min_period_s):
    """Common grid of load, pv and the long-term plan from start (A1-IMP R2). Values in kW.
    The plan is step-held; before its first point its first value is used, converted to kW like the rest (F5 fix).
    Returns (times, load_kw, pv_kw, plan_kw) or None."""
    if not load or not pv or not plan:
        logger.warning(f"input series missing: load={len(load)}, pv={len(pv)}, plan={len(plan)} values")
        return None
    end = Util.parse_time(Util.find_common_time_range([load, pv, plan])["end"])
    if (end - start).total_seconds() < min_period_s:
        logger.warning(f"common horizon {start.isoformat()} .. {end.isoformat()} is shorter than "
                       f"min_period {min_period_s} s")
        return None
    plan_sorted = sorted(plan, key=lambda r: Util.parse_time(r["time"]))
    try:
        load_s = Util.extract_prognosis_values(load, "consumption", start, end, interval_s)
        pv_s = Util.extract_prognosis_values(pv, "production", start, end, interval_s)
        plan_s = Util.generate_result_series(plan_sorted, start, end, interval_s, plan_sorted[0]["value"])
    except (Util.TaskFailException, ValueError) as e:
        logger.warning(f"input series cannot cover the planning window: {e}")
        return None
    n = min(len(load_s), len(pv_s), len(plan_s))
    times = [start + timedelta(seconds=k * interval_s) for k in range(n)]
    return (times,
            [r["value"] / 1000.0 for r in load_s[:n]],
            [r["value"] / 1000.0 for r in pv_s[:n]],
            [r["value"] / 1000.0 for r in plan_s[:n]])


def iso(t):
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def run_fallback(params, start, result_dp_id, plan, reason):
    """Pass the long-term plan through uncorrected (A1-IMP R6). Returns the exit code."""
    remaining = [r for r in (plan or []) if Util.parse_time(r["time"]) >= start]
    if not remaining:
        logger.error(f"{reason}; no long-term plan in '{params['essCurrentPowerPlanIdentifier']}' "
                     f"to pass through - nothing posted.")
        return 1
    readings = [{"time": r["time"] if isinstance(r["time"], str) else iso(r["time"]), "value": float(r["value"])}
                for r in remaining]
    query_utils.post_datapoint_prognosis(result_dp_id, iso(start), readings)
    logger.warning(f"{reason}; passed the long-term plan through uncorrected ({len(readings)} values).")
    return 0


def run(config_path, now=None):
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    app_name = raw.get("appModule", "a1") if isinstance(raw, dict) else "a1"
    loki_url = (raw.get("params") or {}).get("lokiUrl", DEFAULTS["lokiUrl"]) if isinstance(raw, dict) else None
    setup_logger(app_name="A1Runner", log_file="query.log", loki_url=loki_url, loki_tags={"app_name": app_name})
    logger.info(f"{app_name} start.")

    try:
        params = load_params(raw)
    except (ConfigError, ValueError, TypeError) as e:
        logger.error(f"configuration error: {e}")
        return 1

    query_utils.init(params["apiEndpoint"], {})
    jwt_token = query_utils.get_token(params["clientId"], params["token"])
    if not jwt_token:
        logger.error(f"{app_name}: could not obtain an API token from {params['apiEndpoint']}/auth/token "
                     f"for clientId '{params['clientId']}' (check the credential in the app config). Stopping.")
        return 1
    query_utils.add_header("Authorization", f"Bearer {jwt_token}")

    now = now or datetime.now(timezone.utc)
    interval_s = int(params["interval"])
    start = Util.align_start(now, interval_s)

    try:
        result_dp = query_utils.get_datapoint(params["essResultPowerPlanIdentifier"])
        if not result_dp:
            raise InputError(f"result datapoint '{params['essResultPowerPlanIdentifier']}' does not exist")
        result_dp_id = result_dp[0]["id"]
        p_max_kw = value_or_datapoint(params, "ess_max_P_dp_id", "essPowerLimitW") / 1000.0
        capacity_kwh = value_or_datapoint(params, "ess_max_E_dp_id", "essMaxCharge") / 1000.0
        e0_kwh = read_value(params["essEffectiveCharge"], "essEffectiveCharge",
                            now=now, max_age_s=float(params["maxInputAgeS"])) / 1000.0
        if p_max_kw <= 0 or capacity_kwh <= 0:
            raise InputError(f"ESS power limit ({p_max_kw} kW) and capacity ({capacity_kwh} kWh) must be > 0")
        load = read_forecast(params["consumptionPrognosisIdentifier"], "consumptionPrognosisIdentifier")
        pv = read_forecast(params["productionPrognosisIdentifier"], "productionPrognosisIdentifier")
        plan = read_forecast(params["essCurrentPowerPlanIdentifier"], "essCurrentPowerPlanIdentifier")
    except InputError as e:
        logger.error(f"input error: {e}")
        return 1

    logger.info(f"inputs: Pmax={p_max_kw} kW, capacity={capacity_kwh} kWh, E0={e0_kwh:.1f} kWh "
                f"(SoC {100 * e0_kwh / capacity_kwh:.1f} %), values load/pv/plan={len(load)}/{len(pv)}/{len(plan)}, "
                f"start={iso(start)}")

    series = build_series(load, pv, [dict(r) for r in plan], start, interval_s, int(params["min_period"]))
    if series is None:
        return run_fallback(params, start, result_dp_id, plan, "not enough input data")
    times, load_s, pv_s, plan_s = series

    inp = a1model.PlanInputs(
        interval_s=interval_s, load_kw=load_s, pv_kw=pv_s, plan_kw=plan_s,
        p_max_kw=p_max_kw, capacity_kwh=capacity_kwh, e0_kwh=e0_kwh,
        soc_min_pct=float(params["ess_min_soc"]), soc_max_pct=float(params["ess_max_soc"]),
        import_limit_kw=float(params["pccImportLimitW"]) / 1000.0,
        export_limit_kw=-float(params["pccExportLimitW"]) / 1000.0,
        eta_charge=float(params["essChargeEfficiency"]), eta_discharge=float(params["essDischargeEfficiency"]),
        pcc_penalty_per_kwh=float(params["pccLimitPenaltyPerKWh"]),
    )
    m = a1model.build_model(inp)
    result = a1model.solve(m, time_limit_s=float(params["solverTimeLimitS"]))
    if params["writeDebugFile"] == "always" or (params["writeDebugFile"] == "on-failure" and not result.ok):
        debug_model(m)
    if not result.ok:
        return run_fallback(params, start, result_dp_id, plan, f"solver result {result.status}")

    logger.info(f"solved ({result.status}): {len(times)} steps {iso(times[0])} .. {iso(times[-1])}, "
                f"total correction {result.correction_kwh:.2f} kWh")
    if result.import_overrun_kwh > 1e-6 or result.export_overrun_kwh > 1e-6:
        logger.warning(f"corrected plan still exceeds the PCC limits: import {result.import_overrun_kwh:.2f} kWh, "
                       f"export {result.export_overrun_kwh:.2f} kWh over the horizon")
    readings = [{"time": iso(t), "value": float(v * 1000.0)} for t, v in zip(times, result.ess_kw)]
    query_utils.post_datapoint_prognosis(result_dp_id, iso(start), readings)
    logger.info(f"{app_name} finished: corrected plan posted to '{params['essResultPowerPlanIdentifier']}'.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run A1 with config file")
    parser.add_argument("-c", "--config", required=False, help="Path to config YAML file", default="/app/config.yaml")
    args = parser.parse_args(argv)
    try:
        return run(args.config)
    except Exception:  # last resort: a clear message instead of a bare traceback
        logger.exception("A1 failed with an unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
