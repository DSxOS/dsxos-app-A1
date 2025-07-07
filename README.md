# dsxos-app-A1 - Peak shaving Application

This repository contains a **Peak Shaving Application** consisting of two coordinated components:

    1. Power Management Application

    2. Energy Management Application

Together, they manage power and energy flows at the Point of Common Coupling (PCC) to ensure operational efficiency and constraint compliance.

## Power Management Application

The Power Management Application ensures that power values at the Point of Common Coupling (PCC) do not exceed predefined limits. It determines the direction of power flow at the PCC and enforces the respective power limits. If PCC power exceeds the limit, system assets are utilized to adjust the power levels back within the limit. Conversely, if PCC power is within the limit, the application fine-tunes the output of system assets to maintain optimal power levels according to the scheduled energy.

## Energy Management Application

The Energy Management Application ensures sufficient energy is available to support the power management functions. It uses load and production forecasts to model PCC energy use over the outlook period. If the used energy exceeds the set limits, the application adjusts the energy schedule, making minimal changes to the existing schedule to ensure compliance and efficiency.

Peak shaving algorithm is described in detail [here](https://github.com/DSxOS/platform/blob/main/docs/workermodules/Peak_Shaving_Description.pdf)

Files

- main.py - Main entry point for running the peak shaving application.
- debug.py - Debugging tool for diagnosing model infeasibilities, logging variable/constraint states, and saving diagnostics.
- logger.py - Logging utility.
- query_utils.py - Helper methods for interacting with the database: 'get_datapoint' , 'get_last_reading', 'get_last_reading_value', 'get_last_prognosis_reading', 'get_datapoint_prognosis', 'post_prognosis_readings', 'post_datapoint_prognosis'.
- Query.py - Abstractions for HTTP-based data access (GET, POST, PUT, DELETE).
- Util.py - Data processing utilities: 'calculate_count', 'validate_inputs', 'parse_time', 'generate_result_series', 'extract_prognosis_values', 'find_common_time_range', 'extract_values_only'.
- requirements.txt - Required Python packages.
- example_config.yaml - Example configuration file used by main.py

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Ensure example_config.yaml is configured, then run:

```bash
python main.py
```
