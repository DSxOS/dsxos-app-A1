"""SFU-A1 AC1'(a)-(d): capacity is the usable capacity, energy inputs are usable energy in [0, capacity]."""
import logging

import pytest

import main
import model as a1model

CAP = 227.5


def base_params(**over):
    p = {"apiEndpoint": "http://x", "clientId": "c", "token": "t",
         "consumptionPrognosisIdentifier": "a", "productionPrognosisIdentifier": "b",
         "essCurrentPowerPlanIdentifier": "c", "essResultPowerPlanIdentifier": "d",
         "ess_max_P_charge_dp_id": 1000, "ess_max_P_discharge_dp_id": 1000, "ess_max_E_dp_id": 1000,
         "essEffectiveCharge": "e", "pccImportLimitW": 1000, "pccExportLimitW": 0,
         "interval": 900, "min_period": 3600}
    p.update(over)
    return {"params": p}


@pytest.mark.parametrize("name", ["ess_min_soc", "ess_max_soc"])
def test_a_soc_limits_are_a_stale_parameter_error(name):
    with pytest.raises(main.ConfigError, match=name) as e:
        main.load_params(base_params(**{name: 0.5}))
    assert "removed (SFU-A1)" in str(e.value)


def test_a_config_without_soc_limits_passes():
    main.load_params(base_params())
    assert "ess_min_soc" not in main.OPERATIONAL and "ess_max_soc" not in main.OPERATIONAL
    assert {"ess_min_soc", "ess_max_soc"} <= set(main.REMOVED)


def inputs(**over):
    base = dict(interval_s=900, load_kw=[100.0] * 8, pv_kw=[0.0] * 8, plan_kw=[0.0] * 8,
                p_charge_kw=100.0, p_discharge_kw=100.0, capacity_kwh=CAP, e0_kwh=34.3,
                import_limit_kw=80.0, lower_limit_kw=0.0)
    base.update(over)
    return a1model.PlanInputs(**base)


def test_b_bounds_are_zero_and_capacity():
    assert a1model.energy_bounds(inputs()) == (0.0, 227.5)
    m = a1model.build_model(inputs())
    assert all(m.E[k].lb == 0.0 and m.E[k].ub == 227.5 for k in range(1, len(inputs().load_kw) + 1))


@pytest.mark.parametrize("u,used,warns", [(34.3, 34.3, False), (-5.0, 0.0, True), (300.0, 227.5, True)])
def test_c_energy_is_clamped_to_zero_capacity(caplog, u, used, warns):
    with caplog.at_level(logging.WARNING):
        got = main.clamp_energy_kwh(u, CAP, "essEffectiveCharge")
    assert got == pytest.approx(used, abs=1e-9)
    assert (any(r.levelno == logging.WARNING for r in caplog.records)) is warns
    if warns:
        assert "clamped" in caplog.text and f"{u:.3f}" in caplog.text and f"{used:.3f}" in caplog.text


@pytest.mark.parametrize("u", [34.3, -5.0, 300.0])
def test_d_solve_keeps_energy_in_the_bounds(u):
    e0 = main.clamp_energy_kwh(u, CAP, "essEffectiveCharge")
    # Heavy discharge demand and a charge-heavy variant both stay inside [0, capacity].
    for load, pv in (([300.0] * 8, [0.0] * 8), ([0.0] * 8, [-300.0] * 8)):
        r = a1model.solve(a1model.build_model(inputs(e0_kwh=e0, load_kw=load, pv_kw=pv)), time_limit_s=60)
        assert r.ok, r.status
        assert all(-1e-6 <= e <= CAP + 1e-6 for e in r.energy_kwh), r.energy_kwh
