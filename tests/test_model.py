"""A1-IMP AC1(b)-(e): correction model (needs glpsol)."""
import pytest

import model as a1model


def inputs(**over):
    base = dict(interval_s=900, load_kw=[100.0] * 4, pv_kw=[0.0] * 4, plan_kw=[0.0] * 4,
                p_charge_kw=50.0, p_discharge_kw=50.0, capacity_kwh=100.0, e0_kwh=50.0,
                import_limit_kw=80.0, lower_limit_kw=0.0, eta_charge=0.95, eta_discharge=0.95,
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


def test_energy_floor_limits_the_correction():
    # Only 5 kWh of usable energy (the floor is 0): at most 5 * 0.95 = 4.75 kWh can be delivered; the rest is overrun.
    r = solve(inputs(e0_kwh=5.0))
    assert min(r.energy_kwh) >= -1e-6
    assert r.import_overrun_kwh == pytest.approx(20.0 - 5.0 * 0.95, abs=1e-4)


def test_impossible_limit_gives_a_reported_overrun():
    r = solve(inputs(load_kw=[300.0] * 4, p_charge_kw=20.0, p_discharge_kw=20.0))
    assert r.import_overrun_kwh > 0


def test_export_limit_is_also_corrected():
    # PV surplus 30 kW, export 0 allowed, plan says idle: the correction charges 30 kW.
    r = solve(inputs(load_kw=[10.0] * 4, pv_kw=[-40.0] * 4))
    assert all(p == pytest.approx(30.0, abs=1e-6) for p in r.ess_kw)
    assert r.lower_overrun_kwh == pytest.approx(0.0, abs=1e-6)


# ---- OPD R3 / R4 (AC1 d, e) ----

import pyomo.environ as pyo


def old_single_limit_ess(inp, p_max):
    """Independent reference: the pre-OPD formulation (one p_max, imp/exp variables, export limit as -L)."""
    n = len(inp.load_kw)
    dt = inp.interval_s / 3600.0
    e_min, e_max = a1model.energy_bounds(inp)
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, n - 1)
    m.K = pyo.RangeSet(0, n)
    m.ch = pyo.Var(m.T, bounds=(0, p_max))
    m.dis = pyo.Var(m.T, bounds=(0, p_max))
    m.zc = pyo.Var(m.T, domain=pyo.Binary)
    m.a = pyo.Constraint(m.T, rule=lambda m, t: m.ch[t] <= p_max * m.zc[t])
    m.b = pyo.Constraint(m.T, rule=lambda m, t: m.dis[t] <= p_max * (1 - m.zc[t]))
    m.cu = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.cd = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.c = pyo.Constraint(m.T, rule=lambda m, t: m.ch[t] - m.dis[t] - inp.plan_kw[t] == m.cu[t] - m.cd[t])
    m.imp = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.exp = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.oi = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.oe = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.bal = pyo.Constraint(m.T, rule=lambda m, t:
                           m.imp[t] - m.exp[t] == inp.load_kw[t] + inp.pv_kw[t] + m.ch[t] - m.dis[t])
    m.il = pyo.Constraint(m.T, rule=lambda m, t: m.imp[t] <= inp.import_limit_kw + m.oi[t])
    m.el = pyo.Constraint(m.T, rule=lambda m, t: m.exp[t] <= -inp.lower_limit_kw + m.oe[t])
    m.E = pyo.Var(m.K)
    m.E[0].fix(inp.e0_kwh)
    for k in range(1, n + 1):
        m.E[k].setlb(e_min)
        m.E[k].setub(e_max)
    m.en = pyo.Constraint(m.T, rule=lambda m, t:
                          m.E[t + 1] == m.E[t] + (m.ch[t] * inp.eta_charge - m.dis[t] / inp.eta_discharge) * dt)
    m.obj = pyo.Objective(rule=lambda m: sum(dt * (m.cu[t] + m.cd[t] + (m.oi[t] + m.oe[t]) * inp.pcc_penalty_per_kwh)
                                             for t in m.T))
    pyo.SolverFactory("glpk").solve(m)
    return [pyo.value(m.ch[t]) - pyo.value(m.dis[t]) for t in m.T], pyo.value(m.obj)


SCENARIOS = [
    dict(),                                                                   # import limit exceeded
    dict(load_kw=[50.0] * 4, plan_kw=[10.0, -10.0, 10.0, 0.0]),               # within limits
    dict(load_kw=[10.0] * 4, pv_kw=[-40.0] * 4),                              # export corrected
    dict(e0_kwh=20.0),                                                        # SoC-limited
    dict(load_kw=[300.0] * 4, p_charge_kw=20.0, p_discharge_kw=20.0),         # impossible
    dict(load_kw=[90.0, 20.0, 100.0, 5.0], pv_kw=[0.0, -60.0, 0.0, -30.0], plan_kw=[5.0, 0.0, -5.0, 10.0],
         lower_limit_kw=-10.0),                                               # export allowed up to 10 kW
]


@pytest.mark.parametrize("over", SCENARIOS)
def test_equal_power_limits_equal_the_single_limit_model(over):
    inp = inputs(**over)
    assert inp.p_charge_kw == inp.p_discharge_kw
    r = solve(inp)
    ref_ess, ref_obj = old_single_limit_ess(inp, inp.p_charge_kw)
    assert r.ess_kw == pytest.approx(ref_ess, abs=1e-6)
    assert r.objective == pytest.approx(ref_obj, abs=1e-6)


def test_asymmetric_power_limits_are_respected():
    # 100 kW load, import limit 40: needs 60 kW of discharge, but only 30 kW is allowed -> 30 kW, rest overrun.
    r = solve(inputs(load_kw=[100.0] * 4, import_limit_kw=40.0, p_charge_kw=50.0, p_discharge_kw=30.0))
    assert all(p == pytest.approx(-30.0, abs=1e-6) for p in r.ess_kw)
    assert r.import_overrun_kwh == pytest.approx(30.0 * 4 * 0.25, abs=1e-4)
    # PV surplus 80 kW, no export: charge is capped at 50 kW.
    r = solve(inputs(load_kw=[0.0] * 4, pv_kw=[-80.0] * 4, e0_kwh=15.0, p_charge_kw=50.0, p_discharge_kw=30.0))
    assert all(p == pytest.approx(50.0, abs=1e-6) for p in r.ess_kw)
    assert r.lower_overrun_kwh == pytest.approx(30.0 * 4 * 0.25, abs=1e-4)


def test_asymmetric_limits_never_exceeded_with_a_wild_plan():
    r = solve(inputs(plan_kw=[200.0, -200.0, 45.0, -45.0], load_kw=[50.0] * 4,
                     p_charge_kw=50.0, p_discharge_kw=30.0))
    assert all(-30.0 - 1e-6 <= p <= 50.0 + 1e-6 for p in r.ess_kw)
    assert r.ess_kw[1] == pytest.approx(-30.0, abs=1e-6)       # the -200 kW plan is cut to the discharge limit


def test_positive_lower_limit_forces_a_minimum_import():
    # Load 10 kW, plan idle: net 10 < L = 20. A1 charges 10 kW to import at least 20 kW.
    r = solve(inputs(load_kw=[10.0] * 4, lower_limit_kw=20.0, import_limit_kw=80.0))
    assert all(p == pytest.approx(10.0, abs=1e-6) for p in r.ess_kw)
    assert r.lower_overrun_kwh == pytest.approx(0.0, abs=1e-6)


def test_positive_lower_limit_corrects_a_plan_that_exports_or_drops_below():
    # Plan discharges 40 kW at load 50 kW: net 10 < L = 25 -> plan corrected to discharge only 25 kW.
    r = solve(inputs(load_kw=[50.0] * 4, plan_kw=[-40.0] * 4, lower_limit_kw=25.0))
    assert all(p == pytest.approx(-25.0, abs=1e-6) for p in r.ess_kw)
    # PV export with a minimum import: net would be -30; charge 45 kW to reach L = 15.
    r = solve(inputs(load_kw=[10.0] * 4, pv_kw=[-40.0] * 4, e0_kwh=15.0, lower_limit_kw=15.0))
    assert all(p == pytest.approx(45.0, abs=1e-6) for p in r.ess_kw)


def test_positive_lower_limit_infeasible_reports_the_overrun():
    # Needs 60 kW of charge for net >= 60 at load 0; the charge limit is 20 kW -> 40 kW below the limit.
    r = solve(inputs(load_kw=[0.0] * 4, lower_limit_kw=60.0, p_charge_kw=20.0, p_discharge_kw=20.0))
    assert all(p == pytest.approx(20.0, abs=1e-6) for p in r.ess_kw)
    assert r.lower_overrun_kwh == pytest.approx(40.0 * 4 * 0.25, abs=1e-4)
    assert r.import_overrun_kwh == pytest.approx(0.0, abs=1e-6)
