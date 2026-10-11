# 🛢️ TwinFlow AI
### Digital Twin Prototype for Well-to-Surface Optimization

TwinFlow AI is a **demo/prototype** that combines an XGBoost regression model, a differential-evolution optimizer, and a Streamlit dashboard to explore production predictions and operating scenarios.

> **Important:** The included CSV is synthetic demonstration data, not real field data. Model predictions and recommendations are illustrative only. Do not use this prototype to operate real oil-and-gas equipment or make safety-critical decisions.

## Features
- Production prediction using XGBoost regression
- Choke and pump-speed scenario optimization using SciPy Differential Evolution
- Streamlit dashboard with production metrics, charts, equipment status, and dataset statistics
- Synthetic sample dataset included so the project can be run immediately

## Project structure
```text
TwinFlow-AI/
├── app.py
├── ai_engine.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── well_production.csv
```

## Requirements
- Python 3.13(64-bit) or newer recommended
- pip

## Run locally
1. **Download or clone** this repository.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment (recommended):

   **Windows**
   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the dashboard:
   ```bash
   python -m streamlit run app.py
   ```
6. Streamlit will print a local URL (usually `http://localhost:8501`) to open in your browser.

## Dataset format
`data/well_production.csv` must contain these columns:

| Column | Meaning |
|---|---|
| `pressure` | Pressure (bar) |
| `temperature` | Temperature (°C) |
| `flow_rate` | Flow rate (m³/h) |
| `choke` | Choke opening (%) |
| `pump_speed` | Pump speed (%) |
| `water_cut` | Water cut (%) |
| `production` | Production target (bbl/day) |

Replace the included synthetic dataset with an appropriately licensed dataset before using the model for research. Keep the same column names.

## How it works
1. The app loads the CSV and trains an XGBoost regression model.
2. The model predicts production from pressure, temperature, flow rate, choke, pump speed, and water cut.
3. The optimizer searches choke and pump-speed ranges and applies the prototype's configured pressure and energy penalties.
4. The dashboard displays predictions, the selected scenario, and dataset charts.

## Limitations
- No live sensor or SCADA connection is implemented; sensor values are fixed demo inputs in the dashboard.
- Equipment health indicators are illustrative placeholders.
- Optimization constraints and the pressure estimate are simplified assumptions and must be validated by domain experts.
- Predictions depend on the quality and representativeness of the training data.

## Tech stack
Python · Streamlit · Pandas · NumPy · Plotly · XGBoost · SciPy

## License
This project is provided under the MIT License. See `LICENSE`.
