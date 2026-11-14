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
            "pmax": {"lastReadingValue": 50000.0},
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
        "ess_max_P_dp_id": "pmax", "ess_max_E_dp_id": "cap", "essEffectiveCharge": "charge",
        "essMinSafeLim": "unused", "essEffectiveEndCharge": "unused",
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


def test_static_limits_are_used_when_no_datapoints(cfg, monkeypatch, caplog):
    srv = FakeServer()
    srv.install(monkeypatch)
    path = cfg(drop=("ess_max_P_dp_id", "ess_max_E_dp_id"), essPowerLimitW=40000, essMaxCharge=90000)
    assert main.run(path, now=NOW) == 0
    assert "Pmax=40.0 kW, capacity=90.0 kWh" in caplog.text


def test_failed_login_exits_1_without_data_requests(cfg, monkeypatch, caplog):
    srv = FakeServer()
    srv.install(monkeypatch, token=None)
    monkeypatch.setattr(query_utils, "get_datapoint", lambda ident: pytest.fail("data request after failed login"))
    assert main.run(cfg(), now=NOW) == 1
    assert "could not obtain an API token" in caplog.text


@pytest.mark.parametrize("change,needle", [
    (lambda s: s.datapoints.pop("pmax"), "does not exist"),
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
