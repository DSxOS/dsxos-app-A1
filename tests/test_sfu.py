"""SFU AC1 (a)-(d): SoC limits as fractions, energy input as usable energy."""
import logging

import pytest

import main
import model as a1model

CAP, SMIN, SMAX = 350.0, 0.15, 0.80


def consts(lo, hi):
    return {"ess_min_soc": lo, "ess_max_soc": hi}


@pytest.mark.parametrize("lo,hi", [(0.15, 0.80), (0.0, 1.0)])
def test_a_fractions_are_accepted(lo, hi):
    assert main.limit_checks(consts(lo, hi)) == []


@pytest.mark.parametrize("lo,hi", [(0.8, 0.15), (-0.1, 0.5), (0.5, 0.5), (0.1, 1.5)])
def test_a_invalid_fractions_are_rejected(lo, hi):
    assert main.limit_checks(consts(lo, hi))


def test_a_percent_is_rejected_with_a_fractions_hint():
    (names, msg), = main.limit_checks(consts(15, 80))
    assert names == ["ess_min_soc", "ess_max_soc"]
    assert "fractions" in msg and "ess_min_soc=15 looks like percent; use 0.15" in msg
    assert "ess_max_soc=80 looks like percent; use 0.8" in msg


def test_a_percent_constants_are_a_configuration_error():
    params = {"apiEndpoint": "http://x", "clientId": "c", "token": "t",
              "consumptionPrognosisIdentifier": "a", "productionPrognosisIdentifier": "b",
              "essCurrentPowerPlanIdentifier": "c", "essResultPowerPlanIdentifier": "d",
              "ess_max_P_charge_dp_id": 1000, "ess_max_P_discharge_dp_id": 1000, "ess_max_E_dp_id": 1000,
              "essEffectiveCharge": "e", "ess_min_soc": 15, "ess_max_soc": 80,
              "pccImportLimitW": 1000, "pccExportLimitW": 0, "interval": 900, "min_period": 3600}
    with pytest.raises(main.ConfigError, match="fractions"):
        main.load_params({"params": params})
    params.update(ess_min_soc=0.15, ess_max_soc=0.8)
    main.load_params({"params": params})


def inputs(**over):
    base = dict(interval_s=900, load_kw=[100.0] * 8, pv_kw=[0.0] * 8, plan_kw=[0.0] * 8,
                p_charge_kw=100.0, p_discharge_kw=100.0, capacity_kwh=CAP, e0_kwh=86.8,
                soc_min=SMIN, soc_max=SMAX, import_limit_kw=80.0, lower_limit_kw=0.0)
    base.update(over)
    return a1model.PlanInputs(**base)


def test_b_bounds_for_350_kwh():
    assert a1model.energy_bounds(inputs(e0_kwh=100.0)) == pytest.approx((52.5, 280.0))


@pytest.mark.parametrize("usable,total,warns", [(34.3, 86.8, False), (-5.0, 52.5, True), (300.0, 280.0, True)])
def test_c_usable_to_total(caplog, usable, total, warns):
    with caplog.at_level(logging.WARNING):
        got = main.usable_to_total_kwh(usable, CAP, SMIN, SMAX, "essEffectiveCharge")
    assert got == pytest.approx(total, abs=0.01)
    assert (any(r.levelno == logging.WARNING for r in caplog.records)) is warns
    if warns:
        assert "clamped" in caplog.text and f"{usable:.3f}" in caplog.text


@pytest.mark.parametrize("usable", [34.3, -5.0, 300.0])
def test_d_solve_keeps_energy_in_the_window(usable):
    e0 = main.usable_to_total_kwh(usable, CAP, SMIN, SMAX, "essEffectiveCharge")
    # Heavy discharge demand and a charge-heavy variant both stay inside the window.
    for load, pv in (([300.0] * 8, [0.0] * 8), ([0.0] * 8, [-300.0] * 8)):
        r = a1model.solve(a1model.build_model(inputs(e0_kwh=e0, load_kw=load, pv_kw=pv)), time_limit_s=60)
        assert r.ok, r.status
        assert all(52.5 - 1e-6 <= e <= 280.0 + 1e-6 for e in r.energy_kwh), r.energy_kwh
