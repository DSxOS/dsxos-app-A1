"""A1 peak-shaving correction: corrects a long-term ESS plan as little as possible so that the PCC stays within limits.

Run by the platform's python-app scheduler with the site's config (default /app/config.yaml).
Requirements: docs/requirements-a1-nrg-integration.md (A1-IMP), docs/requirements-opd-operational-datapoints.md (OPD:
operational parameters - PCC window, SoC window, charge/discharge power, capacity - are a datapoint id or a number).
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
    "essChargeEfficiency": 0.95,
    "essDischargeEfficiency": 0.95,
    "pccLimitPenaltyPerKWh": 1000.0,
    "solverTimeLimitS": 300,
    "maxInputAgeS": 900,
    "lokiUrl": "http://localhost:3100/loki/api/v1/push",
    "writeDebugFile": "on-failure",
}
# Operational parameters (OPD R1): a datapoint identifier (string) or a number.
OPERATIONAL = [
    "pccImportLimitW", "pccExportLimitW", "ess_min_soc", "ess_max_soc",
    "ess_max_P_charge_dp_id", "ess_max_P_discharge_dp_id", "ess_max_E_dp_id",
]
REQUIRED = [
    "apiEndpoint", "clientId", "token",
    "consumptionPrognosisIdentifier", "productionPrognosisIdentifier", "essCurrentPowerPlanIdentifier",
    "essEffectiveCharge", "essResultPowerPlanIdentifier", "interval", "min_period",
] + OPERATIONAL
# Removed or replaced params (OPD R5): present in the config = configuration error naming the replacement.
REMOVED = {
    "essMinSafeLim": "removed (unused)",
    "npPriceIdentifier": "removed (unused)",
    "essEffectiveEndCharge": "removed (unused)",
    "ess_max_P_dp_id": "replaced by ess_max_P_charge_dp_id and ess_max_P_discharge_dp_id",
    "essPowerLimitW": "replaced by ess_max_P_charge_dp_id and ess_max_P_discharge_dp_id (a number or a datapoint)",
    "essMaxCharge": "replaced by ess_max_E_dp_id (a number or a datapoint)",
}


class ConfigError(Exception):
    pass


class InputError(Exception):
    pass


def empty(v):
    return v is None or v == ""


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def soc_message(a, b):
    """R1 (SFU): the SoC limits are fractions; hint when a value looks like percent."""
    msg = f"need 0 <= ess_min_soc < ess_max_soc <= 1 (fractions, got ess_min_soc={a:g}, ess_max_soc={b:g})"
    for name, v in (("ess_min_soc", a), ("ess_max_soc", b)):
        if v > 1:
            msg += f"; the limits are fractions: {name}={v:g} looks like percent" + (f"; use {v / 100:g}" if v <= 100 else "")
    return msg


def limit_checks(vals):
    """R2 rules over the (possibly partial) dict of resolved operational values. Returns [(params involved, message)]
    for every violated rule whose values are all present."""
    out = []

    def rule(names, ok, msg):
        if all(n in vals for n in names) and not ok(*[vals[n] for n in names]):
            out.append((names, msg(*[vals[n] for n in names]) if callable(msg) else msg))
    for n in ("ess_max_P_charge_dp_id", "ess_max_P_discharge_dp_id", "ess_max_E_dp_id"):
        rule([n], lambda v: v > 0, f"{n} must be > 0")
    rule(["ess_min_soc", "ess_max_soc"], lambda a, b: 0 <= a < b <= 1, soc_message)
    rule(["pccExportLimitW", "pccImportLimitW"], lambda lo, hi: lo <= hi,
         "pccExportLimitW (lower limit) must be <= pccImportLimitW (upper limit)")
    return out


def load_params(raw):
    """Merge defaults into params and validate (A1-IMP R7, OPD R5/R6). Raises ConfigError."""
    if not isinstance(raw, dict) or not isinstance(raw.get("params"), dict):
        raise ConfigError("config has no 'params' section")
    stale = [f"{k}: {why}" for k, why in REMOVED.items() if k in raw["params"]]
    if stale:
        raise ConfigError("removed/replaced params in the config: " + "; ".join(stale))
    params = dict(DEFAULTS)
    params.update(raw["params"])
    if empty(params.get("clientId")) and not empty(params.get("clientID")):
        params["clientId"] = params["clientID"]          # legacy spelling
    missing = [k for k in REQUIRED if empty(params.get(k))]
    if missing:
        raise ConfigError(f"missing params: {', '.join(missing)}")
    bad = [k for k in OPERATIONAL if not (is_number(params[k]) or isinstance(params[k], str))]
    if bad:
        raise ConfigError(f"{', '.join(bad)} must be a number or a datapoint identifier (string)")
    checks = [
        (0 < float(params["essChargeEfficiency"]) <= 1, "essChargeEfficiency must be in (0, 1]"),
        (0 < float(params["essDischargeEfficiency"]) <= 1, "essDischargeEfficiency must be in (0, 1]"),
        (float(params["pccLimitPenaltyPerKWh"]) >= 0, "pccLimitPenaltyPerKWh must be >= 0"),
        (float(params["solverTimeLimitS"]) > 0, "solverTimeLimitS must be > 0"),
        (float(params["maxInputAgeS"]) > 0, "maxInputAgeS must be > 0"),
        (int(params["interval"]) > 0, "interval must be > 0"),
        (int(params["min_period"]) >= 0, "min_period must be >= 0"),
        (params["writeDebugFile"] in ("on-failure", "always", "never"),
         "writeDebugFile must be on-failure, always or never"),
    ]
    errors = [msg for ok, msg in checks if not ok]
    consts = {k: float(params[k]) for k in OPERATIONAL if is_number(params[k])}
    errors += [msg for _, msg in limit_checks(consts)]
    if errors:
        raise ConfigError("; ".join(errors))
    return params


def resolve_operational(params):
    """Resolve the R1 params: a string is a datapoint (latest value, read now), a number is a constant.
    Returns (values, sources). Raises InputError (datapoint missing or without value, or an R2 rule violated
    through a datapoint) or ConfigError (R2 rule violated by constants)."""
    vals, srcs = {}, {}
    for name in OPERATIONAL:
        v = params[name]
        if is_number(v):
            vals[name], srcs[name] = float(v), "const"
        else:
            vals[name], srcs[name] = read_value(v, name), f"dp {v}"
    for names, msg in limit_checks(vals):
        if any(srcs[n] != "const" for n in names):
            raise InputError(msg + " (" + ", ".join(f"{n}={vals[n]:g} [{srcs[n]}]" for n in names) + ")")
        raise ConfigError(msg)
    return vals, srcs


def usable_to_total_kwh(usable_kwh, capacity_kwh, soc_min, soc_max, name):
    """R3 (SFU): usable energy above the minimum SoC -> total stored energy, clamped to the SoC window."""
    span = capacity_kwh * (soc_max - soc_min)
    clamped = min(max(usable_kwh, 0.0), span)
    if clamped != usable_kwh:
        logger.warning(f"{name}: usable energy {usable_kwh:.3f} kWh is outside [0, {span:.3f}] kWh; "
                       f"clamped to {clamped:.3f} kWh")
    return capacity_kwh * soc_min + clamped


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
        ops, srcs = resolve_operational(params)
        e0_usable_kwh = read_value(params["essEffectiveCharge"], "essEffectiveCharge",
                                   now=now, max_age_s=float(params["maxInputAgeS"])) / 1000.0
        load = read_forecast(params["consumptionPrognosisIdentifier"], "consumptionPrognosisIdentifier")
        pv = read_forecast(params["productionPrognosisIdentifier"], "productionPrognosisIdentifier")
        plan = read_forecast(params["essCurrentPowerPlanIdentifier"], "essCurrentPowerPlanIdentifier")
    except InputError as e:
        logger.error(f"input error: {e}")
        return 1
    except ConfigError as e:
        logger.error(f"configuration error: {e}")
        return 1

    p_charge_kw = ops["ess_max_P_charge_dp_id"] / 1000.0
    p_discharge_kw = ops["ess_max_P_discharge_dp_id"] / 1000.0
    capacity_kwh = ops["ess_max_E_dp_id"] / 1000.0
    upper_kw = ops["pccImportLimitW"] / 1000.0
    lower_kw = ops["pccExportLimitW"] / 1000.0
    soc_min, soc_max = ops["ess_min_soc"], ops["ess_max_soc"]
    e0_kwh = usable_to_total_kwh(e0_usable_kwh, capacity_kwh, soc_min, soc_max, "essEffectiveCharge")
    logger.info("inputs: " + ", ".join(f"{k}={ops[k]:g} ({srcs[k]})" for k in OPERATIONAL))
    logger.info(f"inputs: ess_min_soc={soc_min:g}, ess_max_soc={soc_max:g}, capacity={capacity_kwh:g} kWh, "
                f"usable start energy={e0_usable_kwh:.3f} kWh, total start energy={e0_kwh:.3f} kWh, "
                f"Pch={p_charge_kw} kW, Pdis={p_discharge_kw} kW, PCC limits [{lower_kw}, {upper_kw}] kW, "
                f"values load/pv/plan={len(load)}/{len(pv)}/{len(plan)}, start={iso(start)}")

    series = build_series(load, pv, [dict(r) for r in plan], start, interval_s, int(params["min_period"]))
    if series is None:
        return run_fallback(params, start, result_dp_id, plan, "not enough input data")
    times, load_s, pv_s, plan_s = series

    inp = a1model.PlanInputs(
        interval_s=interval_s, load_kw=load_s, pv_kw=pv_s, plan_kw=plan_s,
        p_charge_kw=p_charge_kw, p_discharge_kw=p_discharge_kw, capacity_kwh=capacity_kwh, e0_kwh=e0_kwh,
        soc_min=soc_min, soc_max=soc_max,
        import_limit_kw=upper_kw, lower_limit_kw=lower_kw,
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
    if result.import_overrun_kwh > 1e-6 or result.lower_overrun_kwh > 1e-6:
        logger.warning(f"corrected plan still exceeds the PCC window: import {result.import_overrun_kwh:.2f} kWh, "
                       f"below the lower limit {result.lower_overrun_kwh:.2f} kWh over the horizon")
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
