"""A1 correction model (A1-IMP R3-R5): the smallest change to a long-term ESS plan that keeps the PCC within limits.

Units: power kW, energy kWh. Positive ESS power = charging. PCC = load + pv + ESS (pv negative = generation).
PCC window (OPD R4): L - oL <= net <= U + oU, net = load + pv + ESS, import positive; separate charge/discharge power limits (OPD R3).
Objective in kWh: total |correction| + penalty for PCC overrun. No end target (A1 follows the long-term plan).
"""
from dataclasses import dataclass, field
from typing import List, Optional

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition


@dataclass
class PlanInputs:
    interval_s: int
    load_kw: List[float]
    pv_kw: List[float]                  # negative = generation
    plan_kw: List[float]                # long-term ESS plan to correct (positive = charging)
    p_charge_kw: float                  # charge power limit (positive magnitude)
    p_discharge_kw: float               # discharge power limit (positive magnitude)
    capacity_kwh: float
    e0_kwh: float                       # total stored energy at the start
    soc_min_pct: float
    soc_max_pct: float
    import_limit_kw: float              # U: soft upper limit of the net PCC power (import positive)
    lower_limit_kw: float               # L: soft lower limit of the net PCC power, signed (negative = export allowed)
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    pcc_penalty_per_kwh: float = 1000.0


@dataclass
class PlanResult:
    ok: bool
    status: str
    ess_kw: List[float] = field(default_factory=list)
    energy_kwh: List[float] = field(default_factory=list)
    correction_kwh: float = 0.0
    import_overrun_kwh: float = 0.0
    lower_overrun_kwh: float = 0.0
    objective: Optional[float] = None


def energy_bounds(inp: PlanInputs):
    """[Emin, Emax] from the SoC window, widened to include E[0] if it is already outside."""
    e_min = inp.capacity_kwh * inp.soc_min_pct / 100.0
    e_max = inp.capacity_kwh * inp.soc_max_pct / 100.0
    return min(e_min, inp.e0_kwh), max(e_max, inp.e0_kwh)


def build_model(inp: PlanInputs) -> pyo.ConcreteModel:
    n = len(inp.load_kw)
    if n == 0 or len(inp.pv_kw) != n or len(inp.plan_kw) != n:
        raise ValueError("load, pv and plan series must be non-empty and of equal length")
    dt_h = inp.interval_s / 3600.0
    e_min, e_max = energy_bounds(inp)
    p_ch, p_dis = inp.p_charge_kw, inp.p_discharge_kw

    m = pyo.ConcreteModel(name="A1")
    m.dt_h = dt_h
    m.T = pyo.RangeSet(0, n - 1)
    m.K = pyo.RangeSet(0, n)

    m.ch = pyo.Var(m.T, bounds=(0, p_ch))
    m.dis = pyo.Var(m.T, bounds=(0, p_dis))
    m.zc = pyo.Var(m.T, domain=pyo.Binary)
    m.ch_on = pyo.Constraint(m.T, rule=lambda m, t: m.ch[t] <= p_ch * m.zc[t])
    m.dis_on = pyo.Constraint(m.T, rule=lambda m, t: m.dis[t] <= p_dis * (1 - m.zc[t]))

    # Correction relative to the long-term plan (R4)
    m.c_up = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.c_down = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.correction = pyo.Constraint(m.T, rule=lambda m, t:
                                  m.ch[t] - m.dis[t] - inp.plan_kw[t] == m.c_up[t] - m.c_down[t])

    # Signed net PCC power (import positive) in the window [L - oL, U + oU] (OPD R4)
    m.net = pyo.Var(m.T)
    m.o_imp = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.o_low = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.balance = pyo.Constraint(m.T, rule=lambda m, t:
                               m.net[t] == inp.load_kw[t] + inp.pv_kw[t] + m.ch[t] - m.dis[t])
    m.imp_limit = pyo.Constraint(m.T, rule=lambda m, t: m.net[t] <= inp.import_limit_kw + m.o_imp[t])
    m.low_limit = pyo.Constraint(m.T, rule=lambda m, t: m.net[t] >= inp.lower_limit_kw - m.o_low[t])

    m.E = pyo.Var(m.K)
    m.E[0].fix(inp.e0_kwh)
    for k in range(1, n + 1):
        m.E[k].setlb(e_min)
        m.E[k].setub(e_max)
    m.energy = pyo.Constraint(m.T, rule=lambda m, t:
                              m.E[t + 1] == m.E[t] + (m.ch[t] * inp.eta_charge - m.dis[t] / inp.eta_discharge) * dt_h)

    m.cost = pyo.Objective(
        rule=lambda m: sum(dt_h * (m.c_up[t] + m.c_down[t] + (m.o_imp[t] + m.o_low[t]) * inp.pcc_penalty_per_kwh)
                           for t in m.T),
        sense=pyo.minimize)
    return m


def solve(m: pyo.ConcreteModel, time_limit_s: float = 300, solver_name: str = "glpk") -> PlanResult:
    solver = pyo.SolverFactory(solver_name)
    if solver_name == "glpk":
        solver.options["tmlim"] = int(time_limit_s)
    results = solver.solve(m, load_solutions=False)
    term = results.solver.termination_condition
    status = f"{results.solver.status}/{term}"
    usable = term in (TerminationCondition.optimal, TerminationCondition.feasible) or (
        term == TerminationCondition.maxTimeLimit and len(results.solution) > 0)
    if results.solver.status not in (SolverStatus.ok, SolverStatus.warning) or not usable:
        return PlanResult(ok=False, status=status)
    m.solutions.load_from(results)
    return PlanResult(
        ok=True,
        status=status,
        ess_kw=[pyo.value(m.ch[t]) - pyo.value(m.dis[t]) + 0.0 for t in m.T],
        energy_kwh=[pyo.value(m.E[k]) for k in m.K],
        correction_kwh=sum(pyo.value(m.c_up[t]) + pyo.value(m.c_down[t]) for t in m.T) * m.dt_h,
        import_overrun_kwh=sum(pyo.value(m.o_imp[t]) for t in m.T) * m.dt_h,
        lower_overrun_kwh=sum(pyo.value(m.o_low[t]) for t in m.T) * m.dt_h,
        objective=pyo.value(m.cost),
    )
