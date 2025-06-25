# dsxos-app-A1

# Peak shaving Application

Consists of Two Applications

## Power Management Application

The Power Management Application ensures that power values at the Point of Common Coupling (PCC) do not exceed predefined limits. It determines the direction of power flow at the PCC and enforces the respective power limits. If PCC power exceeds the limit, system assets are utilized to adjust the power levels back within the limit. Conversely, if PCC power is within the limit, the application fine-tunes the output of system assets to maintain optimal power levels according to the scheduled energy.

## Energy Management Application

The Energy Management Application ensures sufficient energy is available to support the power management functions. It uses load and production forecasts to model PCC energy use over the outlook period. If the used energy exceeds the set limits, the application adjusts the energy schedule, making minimal changes to the existing schedule to ensure compliance and efficiency.

Peak shaving algorithm is described [here](https://github.com/DSxOS/platform/blob/main/docs/workermodules/Peak_Shaving_Description.pdf)

Files
- main.py
- debug.py
- logger.py
- query_utils.py
- Query.py
- Util.py
- requirements.txt
- example_config.yaml