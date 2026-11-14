"""A1-IMP AC1(b)-(e): correction model (needs glpsol)."""
import pytest

import model as a1model


def inputs(**over):
    base = dict(interval_s=900, load_kw=[100.0] * 4, pv_kw=[0.0] * 4, plan_kw=[0.0] * 4,
                p_max_kw=50.0, capacity_kwh=100.0, e0_kwh=50.0, soc_min_pct=15, soc_max_pct=80,
                import_limit_kw=80.0, export_limit_kw=0.0, eta_charge=0.95, eta_discharge=0.95,
                pcc_penalty_per_kwh=1000.0)
    base.update(over)
    return a1model.PlanInputs(**base)


def solve(inp):
    r = a1model.solve(a1model.build_model(inp), time_limit_s=60)
    assert r.ok, r.status
    return r


def test_smallest_correction_that_respects_the_import_limit():
    # PCC would be 100 kW > 80 kW: the only and smallest fix is to discharge 20 kW in every step.
    r = solve(inputs())
    assert all(p == pytest.approx(-20.0, abs=1e-6) for p in r.ess_kw)
    assert r.correction_kwh == pytest.approx(20.0, abs=1e-6)          # 4 steps x 20 kW x 0.25 h
    assert r.import_overrun_kwh == pytest.approx(0.0, abs=1e-6)
    assert r.energy_kwh[-1] == pytest.approx(50.0 - 20.0 / 0.95, abs=1e-6)


def test_plan_within_limits_is_unchanged():
    r = solve(inputs(load_kw=[50.0] * 4, plan_kw=[10.0, -10.0, 10.0, 0.0]))
    assert r.ess_kw == pytest.approx([10.0, -10.0, 10.0, 0.0], abs=1e-6)
    assert r.correction_kwh == pytest.approx(0.0, abs=1e-6)


def test_soc_window_limits_the_correction():
    # Only 5 kWh stored above Emin (15 kWh): at most 5 * 0.95 = 4.75 kWh can be delivered; the rest is overrun.
    r = solve(inputs(e0_kwh=20.0))
    assert min(r.energy_kwh) >= 15.0 - 1e-6
    assert r.import_overrun_kwh == pytest.approx(20.0 - 5.0 * 0.95, abs=1e-4)


def test_impossible_limit_gives_a_reported_overrun():
    r = solve(inputs(load_kw=[300.0] * 4, p_max_kw=20.0))
    assert r.import_overrun_kwh > 0


def test_export_limit_is_also_corrected():
    # PV surplus 30 kW, export 0 allowed, plan says idle: the correction charges 30 kW.
    r = solve(inputs(load_kw=[10.0] * 4, pv_kw=[-40.0] * 4))
    assert all(p == pytest.approx(30.0, abs=1e-6) for p in r.ess_kw)
    assert r.export_overrun_kwh == pytest.approx(0.0, abs=1e-6)
