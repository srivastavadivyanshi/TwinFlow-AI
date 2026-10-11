from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from scipy.optimize import differential_evolution


# =========================================================
# 1. LOAD WELL PRODUCTION DATA
# =========================================================

def load_training_data():
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "well_production.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {data_path}. "
            "Ensure data/well_production.csv is included in the project."
        )
    df = pd.read_csv(data_path)
    return df


# =========================================================
# 2. TRAIN XGBOOST MODEL
# =========================================================

def train_model():

    df = load_training_data()

    features = [
        "pressure",
        "temperature",
        "flow_rate",
        "choke",
        "pump_speed",
        "water_cut"
    ]

    X = df[features]

    y = df["production"]

    model = XGBRegressor(

        n_estimators=400,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=42
    )

    model.fit(X, y)

    return model


# =========================================================
# 3. PRODUCTION PREDICTION
# =========================================================

def predict_production(
    model,
    pressure,
    temperature,
    flow_rate,
    choke,
    pump_speed,
    water_cut
):

    input_data = pd.DataFrame({

        "pressure": [pressure],

        "temperature": [temperature],

        "flow_rate": [flow_rate],

        "choke": [choke],

        "pump_speed": [pump_speed],

        "water_cut": [water_cut]
    })

    prediction = model.predict(
        input_data
    )[0]

    return float(prediction)


# =========================================================
# 4. OPTIMIZATION ENGINE
# =========================================================

def optimize_operation(

    model,

    pressure,

    temperature,

    flow_rate,

    water_cut
):

    """
    Finds choke and pump settings
    that maximize predicted production.

    Constraints:

    Choke:
        30% -> 90%

    Pump:
        40% -> 100%

    Pressure:
        Must remain below 280 bar
    """

    def objective(x):

        choke = x[0]

        pump_speed = x[1]

        production = predict_production(

            model,

            pressure,

            temperature,

            flow_rate,

            choke,

            pump_speed,

            water_cut
        )

        estimated_pressure = (

            pressure

            + 0.15 * choke

            + 0.08 * pump_speed

        )

        penalty = 0

        if estimated_pressure > 280:

            penalty += (

                estimated_pressure - 280
            ) * 100

        energy_penalty = (
            0.5 * pump_speed
        )

        score = (

            -production

            + penalty

            + energy_penalty
        )

        return score

    bounds = [

        (30, 90),
        (40, 100)
    ]

    result = differential_evolution(

        objective,

        bounds,

        seed=42,

        maxiter=50
    )

    optimal_choke = result.x[0]

    optimal_pump = result.x[1]

    optimal_production = predict_production(

        model,

        pressure,

        temperature,

        flow_rate,

        optimal_choke,

        optimal_pump,

        water_cut
    )

    return {

        "choke": round(
            optimal_choke, 2
        ),

        "pump_speed": round(
            optimal_pump, 2
        ),

        "production": round(
            optimal_production, 2
        )
    }


# =========================================================
# 5. DIGITAL TWIN SIMULATION
# =========================================================

def simulate_sensor_data():

    np.random.seed()

    return {

        "pressure": round(
            np.random.normal(245, 3), 2
        ),

        "temperature": round(
            np.random.normal(78, 2), 2
        ),

        "flow_rate": round(
            np.random.normal(126, 4), 2
        ),

        "water_cut": round(
            np.random.uniform(15, 35), 2
        ),

        "choke": round(
            np.random.uniform(55, 70), 2
        ),

        "pump_speed": round(
            np.random.uniform(65, 85), 2
        )
    }


# =========================================================
# 6. MAIN TEST
# =========================================================

if __name__ == "__main__":

    print(
        "\nTraining TwinFlow AI..."
    )

    model = train_model()

    sensors = simulate_sensor_data()

    print("\nCurrent Sensors")

    for key, value in sensors.items():

        print(
            f"{key}: {value}"
        )

    current_production = predict_production(

        model,

        sensors["pressure"],

        sensors["temperature"],

        sensors["flow_rate"],

        sensors["choke"],

        sensors["pump_speed"],

        sensors["water_cut"]
    )

    print(
        f"\nCurrent Production: "
        f"{current_production:.2f} bbl/day"
    )

    result = optimize_operation(

        model,

        sensors["pressure"],

        sensors["temperature"],

        sensors["flow_rate"],

        sensors["water_cut"]
    )

    print("\nAI OPTIMIZATION")

    print(
        f"Optimal Choke: "
        f"{result['choke']}%"
    )

    print(
        f"Optimal Pump Speed: "
        f"{result['pump_speed']}%"
    )

    print(
        f"Predicted Production: "
        f"{result['production']} bbl/day"
    )


# to run :python -m streamlit run app.py
#streamlit run yourscript.py