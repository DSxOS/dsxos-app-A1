import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.environ import SolverFactory, value
from pyomo.util.infeasible import log_infeasible_constraints
from pathlib import Path
from datetime import datetime, timezone
from datetime import timedelta
import argparse
import yaml
import query_utils
import Util
from logger import setup_logger
from debug import debug_model


# create parser
parser = argparse.ArgumentParser(description="Run A1 with config file")
parser.add_argument("-c", "--config", required=True, help="Path to config YAML file")
args = parser.parse_args()              # Read arguments
with open(args.config, "r") as f:       # Open and read config-file
    raw_data = yaml.safe_load(f)
    
# Extract API URL and Token
api_url = raw_data["params"]["apiEndpoint"]
api_token = raw_data["params"]["token"]
api_headers = {"Authorization": api_token}

# Initialize query_utils with URL + headers    
query_utils.init(api_url, api_headers)
logger = setup_logger(
    log_file="query.log",
    loki_url="http://localhost:3100/loki/api/v1/push",  # Loki address
    loki_tags={"app_name": "A1Runner"},        # add more tags if needed
    level="INFO"
)

logger.info("A1Runner start")

start_time = datetime.now(timezone.utc)

########################################################################
# Read and validate input
########################################################################
interval = raw_data["params"]["interval"]
min_period = raw_data["params"]["min_period"]

prod = [{"time": r["time"], "value": r["value"]/1000} for r in query_utils.get_last_prognosis_readings(raw_data["params"]["productionPrognosisIdentifier"])] # production prognosis
cons = [{"time": r["time"], "value": r["value"]/1000} for r in query_utils.get_last_prognosis_readings(raw_data["params"]["consumptionPrognosisIdentifier"])] # consumption prognosis
ess_lt_plan = [{"time": r["time"], "value": r["value"]/1000} for r in query_utils.get_last_prognosis_readings(raw_data["params"]["essCurrentPowerPlanIdentifier"])] # kui andmeid pole, siis nullid

# Model parameters
ESS_kW = raw_data["params"]["essPowerLimitW"]/1000 # salvesti võimsus 
ESS_kWh = query_utils.get_datapoint(raw_data["params"]["essEffectiveCharge"])[0]["lastReadingValue"]/1000 # salvesti efektiivne laetus optimeerimise alguses kWh-des (võib olla negatiivne) - ess_charge.last_reading
ESS_max_kWh = raw_data["params"]["essMaxCharge"]/1000 # salvesti absoluutne mahutavus
P_imp_lim_kW = raw_data["params"]["pccImportLimitW"]/1000 # pcc max tarbimine - pccImportLimitW
P_exp_lim_kW = raw_data["params"]["pccExportLimitW"]/1000 # pcc max müük/tootlus (negatiivne) - pccExportLimitW
ESS_safe_min = query_utils.get_datapoint(raw_data["params"]["essMinSafeLim"])[0]["lastReadingValue"] # - ess_min_batt_safe_lim.last_reading
ESS_END_kWh = query_utils.get_datapoint(raw_data["params"]["essEffectiveEndCharge"])[0]["lastReadingValue"]/1000 # salvesti efektiivne laetus optimeerimise lõpus kWh-des - ess_charge.last_reading
kW_to_kWh = interval / 3600 # kordaja võimsuse teisendamiseks energiaks (intervallist tulenev)

current_dp_id = raw_data["params"]["essCurrentPowerPlanIdentifier"]
result_dp_id = query_utils.get_datapoint(raw_data["params"]["essResultPowerPlanIdentifier"])[0]["id"]

########################################################################
logger.info(f"len(prod): {len(prod)}")
logger.info(f"len(cons): {len(cons)}")
logger.info(f"len(ess_lt_plan): {len(ess_lt_plan)}")
########################################################################
print("=========================")
print(f"ESS_kW -- ,{ESS_kW}!")
print(f"ESS_kWh -- ,{ESS_kWh}!")
print(f"ESS_max_kWh -- ,{ESS_max_kWh}!")
print(f"P_imp_lim_kW -- ,{P_imp_lim_kW}!")
print(f"P_exp_lim_kW -- ,{P_exp_lim_kW}!")
print(f"ESS_safe_min -- ,{ESS_safe_min}!")
print(f"ESS_END_kWh -- ,{ESS_END_kWh}!")
print(f"kW_to_kWh -- ,{kW_to_kWh}!")
print(f"interval -- ,{interval}!")
print(f"current_dp_id -- ,{current_dp_id}!")
print(f"result_dp_id -- ,{result_dp_id}!")
########################################################################

ESS_eff_kWh = ESS_max_kWh  # (ESS_max_kWh*(ESS_SOC_max - ESS_SOC_min)/100) # Salvesti efektiivne mahutavus
ESS_SOC_0 = (ESS_kWh / ESS_eff_kWh) * 100  # + ESS_safe_min
ESS_SOC_0 = 0 if ESS_SOC_0 < 0 else ESS_SOC_0
ESS_SOC_END = ESS_END_kWh / ESS_eff_kWh * 100

########################################################################
print("=========================")
print(f"ESS_eff_kWh -- ,{ESS_eff_kWh}!")
print(f"ESS_SOC_0 -- ,{ESS_SOC_0}!")
print(f"ESS_SOC_END -- ,{ESS_SOC_END}!")
########################################################################

# Divide ESS long term plan into charge and discharge plans
ess_lt_plan_array = np.array([r["value"] for r in ess_lt_plan])
ESS_LT_PLAN_C_data = np.where(ess_lt_plan_array < 0, 0, ess_lt_plan_array)
ESS_LT_PLAN_D_data = np.where(ess_lt_plan_array > 0, 0, ess_lt_plan_array)


def model_to_df(m):
    st = 0
    en = len(m.T)

    periods = range(st, en)
    load = [value(m.P_kW[i]) for i in periods]
    pv = [value(m.PV_kW[i]) for i in periods]
    ess_LT_PLAN = [value(m.ESS_LT_PLAN_kW[i]) for i in periods]
    ess = [value(m.ESS_kW[i]) for i in periods]
    ess_C = [value(m.ESS_C_kW[i]) for i in periods]
    ess_D = [value(m.ESS_D_kW[i]) for i in periods]
    ess_change = [value(m.change[i]) for i in periods]
    ess_soc = [value(m.ESS_SoC[i]) for i in periods]
    pcc = [value(m.PCC_kW[i]) for i in periods]

    df_dict = {
        "Period": periods,
        "Load": load,
        "PV": pv,
        "ESS": ess,
        "ESS_LT_PLAN": ess_LT_PLAN,
        "ESS SoC": ess_soc,
        "ESS +corr": ess_C,
        "ESS -corr": ess_D,
        "ESS abs corr": ess_change,
        "PCC": pcc,
    }

    df = pd.DataFrame(df_dict)
    print(df)
    return df

time_range = Util.find_common_time_range([ess_lt_plan, cons, prod])
period = datetime.fromisoformat(time_range["end"]) - datetime.fromisoformat(time_range["start"])

if int(period.total_seconds()) > min_period:
    time_range_start = datetime.fromisoformat(time_range["start"])
    time_range_end = datetime.fromisoformat(time_range["end"])
    # count = Util.calculate_count(ess_lt_plan, time_range_start, interval)
    initial = query_utils.get_last_prognosis_readings(raw_data["params"]["essCurrentPowerPlanIdentifier"])[0]["value"]

    # print(f"count: {count}")
    print(f"start_time: {start_time}")
    print(f"time_range: {time_range}")
    print(f"initial: {initial}")
    print(f"period: {period.total_seconds()} seconds")
    ess_lt_plan_extracted = Util.generate_result_series(ess_lt_plan, start_time, time_range_end, interval, initial)
    cons_extracted = Util.extract_prognosis_values(cons, "consumption", start_time, time_range_end, interval)
    prod_extracted = Util.extract_prognosis_values(prod, "production", start_time, time_range_end, interval)

    assert len(prod_extracted) == len(cons_extracted), "len(prod_extracted) != len(cons_extracted)"
    assert len(prod_extracted) == len(ess_lt_plan_extracted), "len(prod) != len(np_extracted)"

    ########################################################################
    logger.info(f"len(prod_extracted): {len(prod_extracted)}")
    logger.info(f"len(cons_extracted): {len(cons_extracted)}")
    logger.info(f"len(ess_lt_plan_extracted): {len(ess_lt_plan_extracted)}")
    # logger.info(f"prod_extracted: {prod_extracted}")
    # logger.info(f"cons_extracted: {cons_extracted}")
    # logger.info(f"ess_lt_plan_extracted: {ess_lt_plan_extracted}")
    ########################################################################

    data = pd.DataFrame(
        {
            "Load": Util.extract_values_only(cons_extracted),
            "PV": Util.extract_values_only(prod_extracted),  # kui PV andmed on juba vattides, siis pole nominaali vaja.
            "ESS_LT_PLAN": Util.extract_values_only(ess_lt_plan_extracted),
            # "ESS_F": ess #ess_w,
            # "ESS_F_C": ess_C #ESS_F_C_data,
            # "ESS_F_D": ess_D #ESS_F_D_data,
        }
    )
else: logger.error(f"Not enough current prognosis")

########################################################################
# Build Model
########################################################################
m = ConcreteModel()

# Fixed Parameters
m.T = Set(initialize=data.index.tolist(), doc="Indexes", ordered=True)
m.P_kW = Param(m.T, initialize=data.Load, doc="Load [kW]", within=Any)
m.PV_kW = Param(m.T, initialize=data.PV, doc="PV generation [kW]", within=Any)
m.ESS_LT_PLAN_kW = Param(m.T, initialize=data.ESS_LT_PLAN, doc="ESS Long-Term plan [kW]", within=Any)

# Variable Parameters
m.PCC_kW = Var(m.T, bounds=(P_exp_lim_kW, P_imp_lim_kW), doc="PCC P [kW]", initialize=0.0)
m.ESS_kW = Var(m.T, bounds=(-ESS_kW, ESS_kW), doc="ESS P [kW]")
m.ESS_C_kW = Var(m.T, bounds=(0, 2 * ESS_kW), doc="ESS P Positive Correction [kW]", initialize=0.0, within=Reals,)
m.ESS_D_kW = Var(m.T, bounds=(-ESS_kW * 2, 0), doc="ESS P Negative Correction [kW]", initialize=0.0, within=Reals,)
m.ESS_C_z = Var(m.T, bounds=(0, 1), within=NonNegativeIntegers)
m.ESS_D_z = Var(m.T, bounds=(0, 1), within=NonNegativeIntegers)
m.ESS_SoC = Var( m.T, bounds=(0, 100), doc="ESS SoC [%]", initialize=ESS_SOC_0, within=NonNegativeReals,)  # väärtused 10 ja 100 ka parameetrid
m.change = Var( m.T, bounds=(-ESS_kW, ESS_kW), doc="ESS P Absolute Correction [kW]", initialize=0.0, within=Reals,)


# Model Constraints

def ess_output_rule(m, t):
    "ESS output Calculation"
    return m.ESS_kW[t] == m.ESS_LT_PLAN_kW[t] + m.change[t]


m.ess_output = Constraint(m.T, rule=ess_output_rule)


def pcc_output_rule(m, t):
    "PCC output Calculation"
    return m.PCC_kW[t] == m.P_kW[t] + m.PV_kW[t] + m.ESS_kW[t]


m.pcc_output = Constraint(m.T, rule=pcc_output_rule)


def ess_SoC_rule(m, t):
    "ESS SOC Calculation"
    if t >= 1:
        return (
            m.ESS_SoC[t]
            == m.ESS_SoC[t - 1] + ((m.ESS_kW[t - 1] * kW_to_kWh) / ESS_eff_kWh) * 100
        )
    else:
        return m.ESS_SoC[t] == ESS_SOC_0


m.ess_SoC_const = Constraint(m.T, rule=ess_SoC_rule)
"""
def soc_end_target_rule(m):
    "End SoC value"
    return m.ESS_SoC[len(m.ESS_SoC)-1] + ((m.ESS_kW[len(m.ESS_SoC)-1]*kW_to_kWh)/ESS_eff_kWh)*100 == ESS_SOC_END

m.soc_end_target = Constraint(rule=soc_end_target_rule)
"""


def pcc_hi_limit_rule(m, t):
    "High PCC output limit"
    return m.PCC_kW[t] <= P_imp_lim_kW


m.pcc_power_import_limit = Constraint(m.T, rule=pcc_hi_limit_rule)


def pcc_lo_limit_rule(m, t):
    "Low PCC output limit"
    return m.PCC_kW[t] >= P_exp_lim_kW


m.pcc_power_export_limit = Constraint(m.T, rule=pcc_lo_limit_rule)


def ess_sim_c_d_restrict_rule(m, t):
    "Prohibit simultaneous ESS positive and negative correction"
    return m.ESS_C_z[t] + m.ESS_D_z[t] <= 1


m.ess_sim_c_d_restrict = Constraint(m.T, rule=ess_sim_c_d_restrict_rule)


def ess_d_limit_rule(m, t):
    "Restrict ESS P negative correction"
    return m.ESS_C_kW[t] <= 2 * ESS_kW * m.ESS_C_z[t]


m.ess_d_limit = Constraint(m.T, rule=ess_d_limit_rule)


def ess_c_limit_rule(m, t):
    "Restrict ESS P positive correction"
    return m.ESS_D_kW[t] >= -2 * ESS_kW * m.ESS_D_z[t]


m.ess_c_limit = Constraint(m.T, rule=ess_c_limit_rule)


def ess_corr_lo_limit_rule(m, t):
    "Restrict ESS discharging P"
    return m.ESS_kW[t] >= -ESS_kW


m.ess_corr_lo_limit = Constraint(m.T, rule=ess_corr_lo_limit_rule)


def ess_corr_hi_limit_rule(m, t):
    "Restrict ESS charging P"
    return m.ESS_kW[t] <= ESS_kW


m.ess_corr_hi_limit = Constraint(m.T, rule=ess_corr_hi_limit_rule)


def change_rule(m, t):
    "Calculate absolute ESS P correction"
    return m.change[t] == m.ESS_D_kW[t] + m.ESS_C_kW[t]


m.change_constraint = Constraint(m.T, rule=change_rule)

########################################################################
# Cost function and optimization objective
########################################################################
cost = sum((m.ESS_C_kW[t] - m.ESS_D_kW[t]) for t in m.T)
m.objective = Objective(expr=cost, sense=minimize)

########################################################################
# Solve
########################################################################
solver = SolverFactory("glpk", options={"tmlim": 300})
results = solver.solve(m)
results.write()

# debug_model(m)

if (results.solver.status == SolverStatus.ok) and (
    (results.solver.termination_condition == TerminationCondition.optimal)
    or (results.solver.termination_condition == TerminationCondition.feasible)
):
    # Format results as data frame
    results_df = model_to_df(m)
    
    if (len(results_df) == 0):
        # Use old prognosis, if it exists
        logger.warning(f"Optimization failed - empty result")
        essPowerPrognosisRaw = [r["value"] for r in query_utils.get_last_prognosis_readings(current_dp_id)]
    else:
        # use optimized results
        essPowerPrognosisRaw = results_df["ESS"].values * 1000

    essPowerPlanned =[]

    for i, value in enumerate(essPowerPrognosisRaw):
        reading_time = start_time + timedelta(seconds=i * interval)  
        essPowerPlanned.append({
            "time": reading_time.isoformat().replace('+00:00', 'Z'),
            "value": value
        })

    prognosis_payload = {
        "datapointId": result_dp_id,
        "time": start_time.isoformat().replace('+00:00', 'Z'),
        "readings":essPowerPlanned
    }

    response = query_utils.post_datapoint_prognosis(prognosis_payload)
    logger.info(f"datapoint prognosis was posted: {response}")
else: 
    logger.error(f"Solver failed - empty result")

logger.info("A1Runner finished")