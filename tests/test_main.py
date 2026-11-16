"""A1-IMP AC1(a), (f), (g): end to end through main.run with a fake data server."""
from datetime import datetime, timedelta, timezone

import pytest
import yaml

import main
import model as a1model
import query_utils

NOW = datetime(2026, 11, 14, 10, 0, 40, tzinfo=timezone.utc)     # A1 runs ~30 s after A5
START = datetime(2026, 11, 14, 10, 0, 0, tzinfo=timezone.utc)    # aligned to 60 s and 900 s


def iso(t):
    return t.isoformat().replace("+00:00", "Z")


def series(hours, value, step_min=1, begin=START - timedelta(minutes=30)):
    return [{"time": iso(begin + timedelta(minutes=step_min * i)), "value": value(i)}
            for i in range(int(hours * 60 / step_min) + 1)]


class FakeServer:
    def __init__(self, hours=10, plan_begin=START):
        self.datapoints = {
            "ess_p_a1_test": {"id": 532},
            "pchg": {"lastReadingValue": 50000.0},
            "pdis": {"lastReadingValue": 50000.0},
            "cap": {"lastReadingValue": 100000.0},
            "charge": {"lastReadingValue": 50000.0, "lastReadingTime": iso(NOW - timedelta(seconds=20))},
        }
        self.forecasts = {
            "cons": series(hours, lambda i: 100000.0),           # 100 kW: above the 80 kW limit
            "prod": series(hours, lambda i: 0.0),
            "a5plan": series(hours, lambda i: 0.0, begin=plan_begin),
        }
        self.posted = []

    def install(self, monkeypatch, token="T"):
        monkeypatch.setattr(query_utils, "get_token", lambda cid, secret: token)
        monkeypatch.setattr(query_utils, "get_datapoint",
                            lambda ident: [self.datapoints[ident]] if ident in self.datapoints else [])
        monkeypatch.setattr(query_utils, "get_last_prognosis_readings",
                            lambda ident, page_size=10000: [dict(r) for r in self.forecasts.get(ident, [])])
        monkeypatch.setattr(query_utils, "post_datapoint_prognosis",
                            lambda dp_id, t, readings: self.posted.append((dp_id, t, readings)) or {"id": 1})


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = {
        "apiEndpoint": "http://x/api", "clientId": "c", "token": "s", "lokiUrl": "",
        "consumptionPrognosisIdentifier": "cons", "productionPrognosisIdentifier": "prod",
        "essCurrentPowerPlanIdentifier": "a5plan", "essResultPowerPlanIdentifier": "ess_p_a1_test",
        "ess_max_P_charge_dp_id": "pchg", "ess_max_P_discharge_dp_id": "pdis", "ess_max_E_dp_id": "cap",
        "essEffectiveCharge": "charge",
        "ess_min_soc": 15, "ess_max_soc": 80, "pccImportLimitW": 80000, "pccExportLimitW": 0,
        "interval": 900, "min_period": 21600, "solverTimeLimitS": 60,
    }
    path = tmp_path / "config.yaml"

    def write(drop=(), **over):
        p = dict(params, **over)
        for k in drop:
            p.pop(k, None)
        path.write_text(yaml.safe_dump({"appModule": "a1", "params": p}))
        return str(path)
    return write


def test_corrected_plan_is_posted_on_an_aligned_grid(cfg, monkeypatch, caplog):
    srv = FakeServer(hours=2.5)
    srv.install(monkeypatch)
    assert main.run(cfg(min_period=3600), now=NOW) == 0
    dp_id, t, readings = srv.posted[0]
    assert dp_id == 532 and t == "2026-11-14T10:00:00Z"
    assert readings[0]["time"] == "2026-11-14T10:00:00Z" and readings[1]["time"] == "2026-11-14T10:15:00Z"
    assert all(r["value"] == pytest.approx(-20000.0, abs=1e-3) for r in readings[:4])  # 100 kW -> 80 kW limit
    assert "total correction" in caplog.text


def test_initial_plan_value_is_converted_to_kw(cfg, monkeypatch):
    # F5 regression: the plan starts 15 min after the run start; the value used before it must be in kW, not W.
    srv = FakeServer(plan_begin=START + timedelta(minutes=15))
    srv.forecasts["a5plan"] = series(10, lambda i: -20000.0, begin=START + timedelta(minutes=15))
    srv.install(monkeypatch)
    _, _, load_kw, pv_kw, plan_kw = (None,) + main.build_series(
        srv.forecasts["cons"], srv.forecasts["prod"], [dict(r) for r in srv.forecasts["a5plan"]],
        START, 900, 21600)
    assert plan_kw[0] == pytest.approx(-20.0)
    assert plan_kw[1] == pytest.approx(-20.0)


def test_clientID_legacy_alias_is_accepted(cfg, monkeypatch):
    srv = FakeServer()
    srv.install(monkeypatch)
    assert main.run(cfg(drop=("clientId",), clientID="legacy"), now=NOW) == 0


def test_numbers_are_used_as_constants(cfg, monkeypatch, caplog):
    srv = FakeServer()
    srv.install(monkeypatch)
    path = cfg(ess_max_P_charge_dp_id=40000, ess_max_P_discharge_dp_id=40000, ess_max_E_dp_id=90000)
    assert main.run(path, now=NOW) == 0
    assert "Pch=40.0 kW, Pdis=40.0 kW, capacity=90.0 kWh" in caplog.text
    assert "ess_max_E_dp_id=90000 (const)" in caplog.text


def test_failed_login_exits_1_without_data_requests(cfg, monkeypatch, caplog):
    srv = FakeServer()
    srv.install(monkeypatch, token=None)
    monkeypatch.setattr(query_utils, "get_datapoint", lambda ident: pytest.fail("data request after failed login"))
    assert main.run(cfg(), now=NOW) == 1
    assert "could not obtain an API token" in caplog.text


@pytest.mark.parametrize("change,needle", [
    (lambda s: s.datapoints.pop("pchg"), "does not exist"),
    (lambda s: s.datapoints["cap"].update(lastReadingValue=None), "has no value"),
    (lambda s: s.datapoints["charge"].update(lastReadingTime=iso(NOW - timedelta(hours=2))), "s old"),
])
def test_input_errors_stop_without_posting(cfg, monkeypatch, caplog, change, needle):
    srv = FakeServer()
    change(srv)
    srv.install(monkeypatch)
    assert main.run(cfg(), now=NOW) == 1
    assert srv.posted == [] and needle in caplog.text


def test_configuration_error_stops(cfg, monkeypatch, caplog):
    FakeServer().install(monkeypatch)
    assert main.run(cfg(drop=("essCurrentPowerPlanIdentifier",)), now=NOW) == 1
    assert "essCurrentPowerPlanIdentifier" in caplog.text


def test_not_enough_data_passes_the_plan_through(cfg, monkeypatch, caplog):
    srv = FakeServer(hours=2)                      # horizon < min_period (6 h)
    srv.forecasts["a5plan"] = series(2, lambda i: 1000.0 * (i % 3), begin=START - timedelta(minutes=30))
    srv.install(monkeypatch)
    assert main.run(cfg(), now=NOW) == 0
    _, _, readings = srv.posted[0]
    assert readings[0]["time"] == "2026-11-14T10:00:00Z"
    assert len(readings) == len([r for r in srv.forecasts["a5plan"] if r["time"] >= "2026-11-14T10:00:00Z"])
    assert "passed the long-term plan through uncorrected" in caplog.text


def test_failed_solve_passes_the_plan_through(cfg, monkeypatch, caplog):
    srv = FakeServer()
    srv.install(monkeypatch)
    monkeypatch.setattr(main.a1model, "solve", lambda m, time_limit_s=300: a1model.PlanResult(ok=False, status="x/y"))
    assert main.run(cfg(writeDebugFile="never"), now=NOW) == 0
    assert len(srv.posted) == 1 and "solver result x/y" in caplog.text


def test_no_plan_exits_1(cfg, monkeypatch, caplog):
    srv = FakeServer()
    srv.forecasts["a5plan"] = []
    srv.install(monkeypatch)
    assert main.run(cfg(), now=NOW) == 1
    assert srv.posted == [] and "no long-term plan" in caplog.text


# ---- OPD AC1 (a)-(f) ----

# name -> (value, overrides that make the value matter). Every case changes the plan relative to the default config.
R1_CASES = {
    "pccImportLimitW": (70000, {}, 100000.0),
    "pccExportLimitW": (20000, {}, 10000.0),
    "ess_min_soc": (45, {}, 100000.0),
    "ess_max_soc": (52, {"pccExportLimitW": 20000}, 10000.0),
    "ess_max_P_charge_dp_id": (5000, {"pccExportLimitW": 20000}, 10000.0),
    "ess_max_P_discharge_dp_id": (10000, {}, 100000.0),
    "ess_max_E_dp_id": (60000, {"pccExportLimitW": 20000}, 10000.0),
}
OPERATIONAL = list(R1_CASES)


def run_plan(cfg, monkeypatch, cons_w, **over):
    srv = FakeServer(hours=2.5)
    srv.forecasts["cons"] = series(2.5, lambda i: cons_w)
    srv.datapoints.update({f"dp_{k}": {"lastReadingValue": v} for k, v in over.pop("_dps", {}).items()})
    srv.install(monkeypatch)
    assert main.run(cfg(min_period=3600, **over), now=NOW) == 0
    return [r["value"] for r in srv.posted[0][2]]


@pytest.mark.parametrize("name", OPERATIONAL)
def test_r1_param_in_datapoint_and_number_form_gives_the_same_plan(cfg, monkeypatch, caplog, name):
    value, extra, cons_w = R1_CASES[name]
    baseline = run_plan(cfg, monkeypatch, cons_w, **extra)
    as_number = run_plan(cfg, monkeypatch, cons_w, **extra, **{name: value})
    caplog.clear()
    as_dp = run_plan(cfg, monkeypatch, cons_w, _dps={name: value}, **extra, **{name: f"dp_{name}"})
    assert as_dp == pytest.approx(as_number, abs=1e-6)
    assert as_number != pytest.approx(baseline, abs=1e-6)       # the value really is used
    assert f"{name}={value:g} (dp dp_{name})" in caplog.text


@pytest.mark.parametrize("name", OPERATIONAL)
@pytest.mark.parametrize("fault,needle", [("missing", "does not exist"), ("novalue", "has no value")])
def test_r1_datapoint_missing_or_without_value_is_an_input_error(cfg, monkeypatch, caplog, name, fault, needle):
    srv = FakeServer()
    if fault == "novalue":
        srv.datapoints["dp_x"] = {"lastReadingValue": None}
    srv.install(monkeypatch)
    assert main.run(cfg(**{name: "dp_x"}), now=NOW) == 1
    assert srv.posted == []
    assert "input error" in caplog.text and name in caplog.text and "dp_x" in caplog.text and needle in caplog.text


@pytest.mark.parametrize("over,needle", [
    ({"ess_max_P_charge_dp_id": 0}, "ess_max_P_charge_dp_id must be > 0"),
    ({"ess_max_P_discharge_dp_id": -5}, "ess_max_P_discharge_dp_id must be > 0"),
    ({"ess_max_E_dp_id": 0}, "ess_max_E_dp_id must be > 0"),
    ({"ess_min_soc": 80}, "0 <= ess_min_soc < ess_max_soc <= 100"),
    ({"ess_min_soc": -1}, "0 <= ess_min_soc < ess_max_soc <= 100"),
    ({"ess_max_soc": 101}, "0 <= ess_min_soc < ess_max_soc <= 100"),
    ({"pccExportLimitW": 90000}, "pccExportLimitW (lower limit) must be <= pccImportLimitW"),
])
def test_r2_constants_violating_the_rules_are_configuration_errors(cfg, monkeypatch, caplog, over, needle):
    srv = FakeServer()
    srv.install(monkeypatch)
    monkeypatch.setattr(query_utils, "get_token", lambda cid, secret: pytest.fail("data request after config error"))
    assert main.run(cfg(**over), now=NOW) == 1
    assert srv.posted == [] and "configuration error" in caplog.text and needle in caplog.text


@pytest.mark.parametrize("name,bad,needle", [
    ("ess_max_P_charge_dp_id", 0.0, "ess_max_P_charge_dp_id must be > 0"),
    ("ess_max_P_discharge_dp_id", -1.0, "ess_max_P_discharge_dp_id must be > 0"),
    ("ess_max_E_dp_id", 0.0, "ess_max_E_dp_id must be > 0"),
    ("ess_min_soc", 90.0, "0 <= ess_min_soc < ess_max_soc <= 100"),
    ("ess_max_soc", 120.0, "0 <= ess_min_soc < ess_max_soc <= 100"),
    ("pccExportLimitW", 90000.0, "pccExportLimitW (lower limit) must be <= pccImportLimitW"),
    ("pccImportLimitW", -5000.0, "pccExportLimitW (lower limit) must be <= pccImportLimitW"),
])
def test_r2_datapoint_values_violating_the_rules_are_input_errors(cfg, monkeypatch, caplog, name, bad, needle):
    srv = FakeServer()
    srv.datapoints["dp_bad"] = {"lastReadingValue": bad}
    srv.install(monkeypatch)
    assert main.run(cfg(**{name: "dp_bad"}), now=NOW) == 1
    assert srv.posted == [] and "input error" in caplog.text and needle in caplog.text and "[dp dp_bad]" in caplog.text


def test_asymmetric_limits_charge_50_discharge_30(cfg, monkeypatch):
    # 100 kW load, import limit 40 kW: wants -60 kW, only 30 kW may be discharged.
    p = run_plan(cfg, monkeypatch, 100000.0, pccImportLimitW=40000,
                 ess_max_P_charge_dp_id=50000, ess_max_P_discharge_dp_id=30000)
    assert all(v == pytest.approx(-30000.0, abs=1e-3) for v in p[:4])
    assert max(p) <= 50000.0 + 1e-3 and min(p) >= -30000.0 - 1e-3
    # PV surplus: charge is capped at 50 kW
    srv = FakeServer(hours=2.5)
    srv.forecasts["cons"] = series(2.5, lambda i: 0.0)
    srv.forecasts["prod"] = series(2.5, lambda i: -80000.0)
    srv.install(monkeypatch)
    assert main.run(cfg(min_period=3600, ess_max_P_charge_dp_id=50000, ess_max_P_discharge_dp_id=30000), now=NOW) == 0
    assert srv.posted[0][2][0]["value"] == pytest.approx(50000.0, abs=1e-3)


def test_equal_limits_give_the_old_single_limit_result(cfg, monkeypatch):
    p = run_plan(cfg, monkeypatch, 100000.0)          # pre-OPD expectation: 100 kW -> 80 kW limit = -20 kW
    assert all(v == pytest.approx(-20000.0, abs=1e-3) for v in p[:4])


def test_positive_lower_limit_is_respected_end_to_end(cfg, monkeypatch, caplog):
    # Minimum import 20 kW at a 10 kW load: A1 charges 10 kW.
    p = run_plan(cfg, monkeypatch, 10000.0, pccExportLimitW=20000)
    assert all(v == pytest.approx(10000.0, abs=1e-3) for v in p[:4])
    # Infeasible: charge capped at 5 kW, so 5 kW below the lower limit is reported.
    caplog.clear()
    p = run_plan(cfg, monkeypatch, 10000.0, pccExportLimitW=20000, ess_max_P_charge_dp_id=5000)
    assert p[0] == pytest.approx(5000.0, abs=1e-3)
    assert "below the lower limit" in caplog.text


@pytest.mark.parametrize("old,extra,needle", [
    ("essMinSafeLim", {}, "essMinSafeLim"),
    ("npPriceIdentifier", {}, "npPriceIdentifier"),
    ("essEffectiveEndCharge", {}, "essEffectiveEndCharge"),
    ("ess_max_P_dp_id", {}, "ess_max_P_charge_dp_id and ess_max_P_discharge_dp_id"),
    ("essPowerLimitW", {}, "ess_max_P_charge_dp_id and ess_max_P_discharge_dp_id"),
    ("essMaxCharge", {}, "ess_max_E_dp_id"),
])
def test_removed_or_replaced_params_are_configuration_errors(cfg, monkeypatch, caplog, old, extra, needle):
    srv = FakeServer()
    srv.install(monkeypatch)
    assert main.run(cfg(**{old: "x"}), now=NOW) == 1
    assert srv.posted == [] and "configuration error" in caplog.text
    assert old in caplog.text and needle in caplog.text


@pytest.mark.parametrize("name", OPERATIONAL)
def test_required_operational_params_have_no_defaults(cfg, monkeypatch, caplog, name):
    FakeServer().install(monkeypatch)
    assert main.run(cfg(drop=(name,)), now=NOW) == 1
    assert "missing params" in caplog.text and name in caplog.text
