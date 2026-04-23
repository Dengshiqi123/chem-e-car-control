"""Interactive Streamlit app for the iodine clock predictor."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import streamlit as st

from predictor import (
    DEFAULT_STOCKS,
    Experiment,
    PredictorError,
    final_concentrations,
    final_concentrations_from_row,
    fit_model,
    load_params,
    plot_fit,
    prediction_diagnostics_from_concentrations,
    predict_time,
    read_calibration,
    save_params,
)


ROOT = Path(__file__).resolve().parent
CALIBRATION_PATH = ROOT / "calibration_current.csv"
PARAMS_PATH = ROOT / "fitted_params.json"
FIT_PLOT_PATH = ROOT / "fit_curve.png"


def load_or_fit_params() -> dict:
    """Load fitted params, falling back to fitting the calibration data."""

    if PARAMS_PATH.exists():
        return load_params(PARAMS_PATH)
    df = read_calibration(CALIBRATION_PATH)
    params = fit_model(df)
    save_params(params, PARAMS_PATH)
    plot_fit(df, params, FIT_PLOT_PATH)
    return params


def params_with_user_stocks(params: dict, stocks: dict[str, float]) -> dict:
    patched = copy.deepcopy(params)
    patched["stock_concentrations"] = stocks
    return patched


def positive_number(label: str, value: float, step: float, min_value: float = 0.0) -> float:
    return float(
        st.number_input(
            label,
            min_value=min_value,
            value=float(value),
            step=float(step),
            format="%.6g",
        )
    )


def concentration_inputs() -> tuple[Experiment, dict[str, float]]:
    st.subheader("Experiment input")
    st.caption("Enter stock concentration and added volume for each reagent.")

    rows = [
        ("Na2S2O3", "na2s2o3_m", "na2s2o3_ml", "mol/L", DEFAULT_STOCKS["na2s2o3_m"], 4.75),
        ("KI", "ki_m", "ki_ml", "mol/L", DEFAULT_STOCKS["ki_m"], 5.0),
        ("starch", "starch_g_l", "starch_ml", "g/L", DEFAULT_STOCKS["starch_g_l"], 4.0),
        ("FeCl3", "fecl3_m", "fecl3_ml", "mol/L", DEFAULT_STOCKS["fecl3_m"], 0.3),
    ]

    stocks: dict[str, float] = {}
    volumes: dict[str, float] = {}
    for reagent, stock_key, volume_key, unit, default_conc, default_volume in rows:
        st.markdown(f"**{reagent}**")
        conc_col, volume_col = st.columns(2)
        with conc_col:
            stocks[stock_key] = positive_number(
                f"{reagent} stock concentration ({unit})",
                default_conc,
                0.001 if unit == "mol/L" else 0.1,
            )
        with volume_col:
            volumes[volume_key] = positive_number(
                f"{reagent} volume (mL)",
                default_volume,
                0.05,
            )

    st.markdown("**water**")
    volumes["water_ml"] = positive_number("water volume (mL)", 1.25, 0.05)

    experiment = Experiment(
        na2s2o3_ml=volumes["na2s2o3_ml"],
        ki_ml=volumes["ki_ml"],
        starch_ml=volumes["starch_ml"],
        water_ml=volumes["water_ml"],
        fecl3_ml=volumes["fecl3_ml"],
    )
    return experiment, stocks


def render_prediction(params: dict, experiment: Experiment, stocks: dict[str, float]) -> None:
    try:
        user_params = params_with_user_stocks(params, stocks)
        prediction = predict_time(experiment, user_params)
        concentrations = final_concentrations(experiment, stocks)
        diagnostics = prediction_diagnostics_from_concentrations(concentrations, user_params)
    except PredictorError as exc:
        st.error(str(exc))
        return

    st.metric("Predicted color-change time", f"{prediction:.2f} s")
    if diagnostics["can_color"] < 0.5:
        st.warning(
            "S2O3^2- is at or above the Fe3+ oxidation capacity. "
            "The reaction may not turn blue within the measurement window."
        )

    st.write("Final mixture concentrations")
    st.dataframe(
        pd.DataFrame(
            [
                {"item": "total volume", "value": concentrations["total_volume_ml"], "unit": "mL"},
                {"item": "Na2S2O3", "value": concentrations["na2s2o3_m"], "unit": "mol/L"},
                {"item": "KI", "value": concentrations["ki_m"], "unit": "mol/L"},
                {"item": "FeCl3", "value": concentrations["fecl3_m"], "unit": "mol/L"},
                {"item": "starch", "value": concentrations["starch_g_l"], "unit": "g/L"},
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.write("Mechanistic diagnostics")
    st.dataframe(
        pd.DataFrame(
            [
                {"item": "ionic strength", "value": diagnostics["ionic_strength_m"], "unit": "mol/L"},
                {"item": "gamma(Fe3+)", "value": diagnostics["gamma_fe3"], "unit": "-"},
                {"item": "gamma(I-)", "value": diagnostics["gamma_i"], "unit": "-"},
                {"item": "gamma(S2O3^2-)", "value": diagnostics["gamma_s2o3"], "unit": "-"},
                {"item": "iodide term", "value": diagnostics["iodide_term"], "unit": "relative"},
                {"item": "Fe3+ decay rate", "value": diagnostics["ferric_decay_rate_s_inv"], "unit": "1/s"},
                {"item": "Fe3+ conversion", "value": diagnostics["ferric_conversion"], "unit": "-"},
                {"item": "Fe3+ remaining", "value": diagnostics["ferric_remaining_m"], "unit": "mol/L"},
                {"item": "depletion factor", "value": diagnostics["depletion_factor"], "unit": "relative"},
                {"item": "kinetic time", "value": diagnostics["kinetic_time_s"], "unit": "s"},
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    render_calibration_context(concentrations, stocks)


def render_calibration_context(concentrations: dict[str, float], stocks: dict[str, float]) -> None:
    try:
        df = read_calibration(CALIBRATION_PATH)
        calibration_conc = pd.DataFrame(
            [final_concentrations_from_row(row) for row in df.to_dict(orient="records")]
        )
    except PredictorError:
        return

    labels = {
        "na2s2o3_m": "Na2S2O3",
        "ki_m": "KI",
        "fecl3_m": "FeCl3",
        "starch_g_l": "starch",
    }

    outside = []
    for key, label in labels.items():
        lower = float(calibration_conc[key].min())
        upper = float(calibration_conc[key].max())
        value = float(concentrations[key])
        if value < lower * 0.8 or value > upper * 1.2:
            outside.append(f"{label}: {value:.4g} outside {lower:.4g}-{upper:.4g}")

    if outside:
        st.info("This input is outside the calibrated range: " + "; ".join(outside))

    ki_varies = (
        ("ki_stock_m" in df.columns and df["ki_stock_m"].nunique(dropna=True) > 1)
        or df["ki_ml"].nunique(dropna=True) > 1
    )
    if not ki_varies and abs(stocks["ki_m"] - DEFAULT_STOCKS["ki_m"]) / max(DEFAULT_STOCKS["ki_m"], 1e-12) > 0.2:
        st.info("The calibration data do not include a KI scan; KI effects are mechanism-based extrapolation.")


def render_fit_panel() -> None:
    st.subheader("Calibration fit")
    st.caption("Refit calibration_current.csv and regenerate fit_curve.png.")

    if st.button("Refit and generate fit curve", type="primary"):
        try:
            df = read_calibration(CALIBRATION_PATH)
            params = fit_model(df)
            save_params(params, PARAMS_PATH)
            plot_fit(df, params, FIT_PLOT_PATH)
            summary = params["training_summary"]
        except PredictorError as exc:
            st.error(str(exc))
            return

        st.success(
            f"Fit complete: rows={summary['rows']}, "
            f"RMSE={summary['rmse_s']:.3f} s, MAE={summary['mae_s']:.3f} s"
        )

    if FIT_PLOT_PATH.exists():
        st.image(str(FIT_PLOT_PATH), caption="fit_curve.png")
    else:
        st.info("No fit curve yet. Click the button above to generate one.")


def main() -> None:
    st.set_page_config(page_title="Iodine Clock Predictor", layout="wide")
    st.title("Iodine Clock Reaction Predictor")
    st.write("Predict color-change time from reagent concentrations and volumes.")

    try:
        params = load_or_fit_params()
    except PredictorError as exc:
        st.error(str(exc))
        return

    input_col, output_col = st.columns([0.95, 1.05])
    with input_col:
        experiment, stocks = concentration_inputs()
    with output_col:
        st.subheader("Prediction")
        render_prediction(params, experiment, stocks)

    st.divider()
    render_fit_panel()


if __name__ == "__main__":
    main()
