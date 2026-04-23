"""Iodine clock predictor with a mechanism-constrained kinetic model.

The key feature is ferric depletion. FeCl3 is not only a trigger; in this
system Fe3+ is the stoichiometric oxidant that generates iodine:

  2 Fe3+ + 2 I- -> I2 + 2 Fe2+
  I2 + 2 S2O3^2- -> 2 I- + S4O6^2-

Before the blue starch-iodine color appears, thiosulfate consumes iodine almost
immediately. Color appears when cumulative iodine production exceeds the
thiosulfate scavenging capacity. Since one mole of S2O3^2- consumes one mole of
Fe3+ equivalent on the path to iodine, the induction time becomes strongly
nonlinear as [S2O3^2-] approaches the available [Fe3+].
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - exercised only when scipy is missing.
    least_squares = None


SCHEMA_VERSION = 6
EPS = 1e-12

VOLUME_KEYS = ("na2s2o3_ml", "ki_ml", "starch_ml", "water_ml", "fecl3_ml")
STOCK_COLUMN_TO_KEY = {
    "na2s2o3_stock_m": "na2s2o3_m",
    "ki_stock_m": "ki_m",
    "fecl3_stock_m": "fecl3_m",
    "starch_stock_g_l": "starch_g_l",
}
STOCK_COLUMNS = tuple(STOCK_COLUMN_TO_KEY)
REQUIRED_COLUMNS = (*VOLUME_KEYS, "time_s")

BASELINE_VOLUMES_ML = {
    "na2s2o3_ml": 4.75,
    "ki_ml": 5.0,
    "starch_ml": 4.0,
    "water_ml": 1.25,
    "fecl3_ml": 0.3,
}

DEFAULT_STOCKS = {
    "na2s2o3_m": 0.015,
    "ki_m": 0.200,
    "fecl3_m": 0.300,
    "starch_g_l": 10.0,
}

DEFAULT_FIXED = {
    "iodide_order": 2.0,
    "stoich_fe_per_s2o3": 1.0,
    "visual_i2_threshold_m": 1.0e-6,
    "starch_detection_order": 0.20,
    "davies_a_25c": 0.51,
    "davies_max_ionic_strength_m": 0.50,
    "no_color_time_s": 3600.0,
}

DEFAULT_FIT_PARAMS = {
    "rate_constant": 2500.0,
    "lag_s": 5.0,
    "iodide_saturation_m_inv": 180.0,
    "ferric_depletion_alpha": 1.45,
}

COLUMN_ALIASES = {
    "na2s2o3_ml": {
        "na2s2o3_ml",
        "na2s2o3",
        "s2o3_ml",
        "thiosulfate_ml",
        "sodium_thiosulfate_ml",
    },
    "ki_ml": {"ki_ml", "ki", "iodide_ml", "potassium_iodide_ml"},
    "starch_ml": {"starch_ml", "starch"},
    "water_ml": {"water_ml", "water", "h2o_ml"},
    "fecl3_ml": {"fecl3_ml", "fecl3", "ferric_chloride_ml", "trigger_ml"},
    "time_s": {"time_s", "time", "seconds", "observed_time_s", "color_time_s"},
    "na2s2o3_stock_m": {
        "na2s2o3_stock_m",
        "na2s2o3_conc_m",
        "na2s2o3_m",
        "s2o3_stock_m",
    },
    "ki_stock_m": {"ki_stock_m", "ki_conc_m", "ki_m", "iodide_stock_m"},
    "fecl3_stock_m": {"fecl3_stock_m", "fecl3_conc_m", "fecl3_m"},
    "starch_stock_g_l": {
        "starch_stock_g_l",
        "starch_conc_g_l",
        "starch_g_l",
    },
}


class PredictorError(ValueError):
    """Raised for user-facing input and compatibility errors."""


@dataclass(frozen=True)
class Experiment:
    """Input volumes for one experiment, in mL."""

    na2s2o3_ml: float
    ki_ml: float
    starch_ml: float
    water_ml: float
    fecl3_ml: float

    def as_dict(self) -> dict[str, float]:
        return {key: float(getattr(self, key)) for key in VOLUME_KEYS}


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a valid number: {value}") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative number")
    return number


def normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize calibration column names while accepting common aliases."""

    rename: dict[str, str] = {}
    normalized_to_original = {normalize_column_name(col): col for col in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            original = normalized_to_original.get(normalize_column_name(alias))
            if original is not None:
                rename[original] = canonical
                break

    canonical_df = df.rename(columns=rename)
    missing = [column for column in REQUIRED_COLUMNS if column not in canonical_df.columns]
    if missing:
        raise PredictorError(
            "calibration file is missing required columns: " + ", ".join(missing)
        )
    return canonical_df


def read_calibration(path: str | Path) -> pd.DataFrame:
    """Read and validate the calibration CSV."""

    path = Path(path)
    if not path.exists():
        raise PredictorError(f"calibration file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise PredictorError(f"failed to read calibration CSV: {exc}") from exc

    df = canonicalize_columns(df)
    kept = {*REQUIRED_COLUMNS, *STOCK_COLUMNS, "run_id", "series", "note"}
    df = df.loc[:, [column for column in df.columns if column in kept]]

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in STOCK_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_rows = df[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    if invalid_rows.any():
        bad = ", ".join(str(index + 1) for index in df.index[invalid_rows].tolist())
        raise PredictorError(f"calibration file has non-numeric values on row(s): {bad}")

    for column in VOLUME_KEYS:
        if (df[column] < 0).any():
            raise PredictorError(f"{column} must be non-negative")
    for column in STOCK_COLUMNS:
        if column in df.columns and (df[column].dropna() < 0).any():
            raise PredictorError(f"{column} must be non-negative")

    if (df["time_s"] <= 0).any():
        raise PredictorError("time_s must be positive")

    total_volume = df[list(VOLUME_KEYS)].sum(axis=1)
    if (total_volume <= 0).any():
        raise PredictorError("total reaction volume must be positive")

    return df.reset_index(drop=True)


def experiment_from_args(args: argparse.Namespace) -> Experiment:
    return Experiment(
        na2s2o3_ml=args.na2s2o3,
        ki_ml=args.ki,
        starch_ml=args.starch,
        water_ml=args.water,
        fecl3_ml=args.fecl3,
    )


def stocks_from_args(args: argparse.Namespace, params: dict[str, Any]) -> dict[str, float]:
    stocks = {**DEFAULT_STOCKS, **params.get("stock_concentrations", {})}
    override_map = {
        "na2s2o3_conc": "na2s2o3_m",
        "ki_conc": "ki_m",
        "starch_conc": "starch_g_l",
        "fecl3_conc": "fecl3_m",
    }
    for attr, key in override_map.items():
        value = getattr(args, attr, None)
        if value is not None:
            stocks[key] = float(value)
    return stocks


def stock_concentrations_from_row(row: dict[str, Any] | pd.Series) -> dict[str, float]:
    stocks = DEFAULT_STOCKS.copy()
    for column, stock_key in STOCK_COLUMN_TO_KEY.items():
        if column in row and pd.notna(row[column]):
            stocks[stock_key] = float(row[column])
    return stocks


def final_concentrations(
    volumes: dict[str, float] | Experiment,
    stocks: dict[str, float] | None = None,
) -> dict[str, float]:
    """Convert added volumes and stock concentrations into final concentrations."""

    if isinstance(volumes, Experiment):
        volume_dict = volumes.as_dict()
    else:
        volume_dict = {key: float(volumes[key]) for key in VOLUME_KEYS}

    stocks = {**DEFAULT_STOCKS, **(stocks or {})}
    total_ml = sum(volume_dict.values())
    if total_ml <= 0:
        raise PredictorError("total reaction volume must be positive")

    return {
        "total_volume_ml": total_ml,
        "na2s2o3_m": stocks["na2s2o3_m"] * volume_dict["na2s2o3_ml"] / total_ml,
        "ki_m": stocks["ki_m"] * volume_dict["ki_ml"] / total_ml,
        "fecl3_m": stocks["fecl3_m"] * volume_dict["fecl3_ml"] / total_ml,
        "starch_g_l": stocks["starch_g_l"] * volume_dict["starch_ml"] / total_ml,
    }


def final_concentrations_from_row(row: dict[str, Any] | pd.Series) -> dict[str, float]:
    volumes = {key: float(row[key]) for key in VOLUME_KEYS}
    return final_concentrations(volumes, stock_concentrations_from_row(row))


def ionic_strength(conc: dict[str, float]) -> float:
    """Estimate ionic strength from Na2S2O3, KI, and FeCl3 ions."""

    s2o3 = max(conc["na2s2o3_m"], 0.0)
    iodide = max(conc["ki_m"], 0.0)
    ferric = max(conc["fecl3_m"], 0.0)
    return 3.0 * s2o3 + iodide + 6.0 * ferric


def davies_gamma(charge: int, ionic_strength_m: float, fixed: dict[str, float]) -> float:
    """Davies activity coefficient approximation at 25 C."""

    if charge == 0:
        return 1.0

    capped_i = min(max(ionic_strength_m, 0.0), fixed["davies_max_ionic_strength_m"])
    sqrt_i = math.sqrt(capped_i)
    term = sqrt_i / (1.0 + sqrt_i) - 0.30 * capped_i
    gamma = 10.0 ** (-fixed["davies_a_25c"] * (charge**2) * term)
    return float(np.clip(gamma, 1.0e-4, 1.0))


def activity_state_from_concentrations(
    conc: dict[str, float],
    params: dict[str, Any],
) -> dict[str, float]:
    fixed = {**DEFAULT_FIXED, **params.get("fixed_params", {})}
    i_strength = ionic_strength(conc)
    gamma_fe = davies_gamma(+3, i_strength, fixed)
    gamma_i = davies_gamma(-1, i_strength, fixed)
    gamma_s2o3 = davies_gamma(-2, i_strength, fixed)

    return {
        "ionic_strength_m": i_strength,
        "gamma_fe3": gamma_fe,
        "gamma_i": gamma_i,
        "gamma_s2o3": gamma_s2o3,
        "a_fe3": gamma_fe * max(conc["fecl3_m"], 0.0),
        "a_i": gamma_i * max(conc["ki_m"], 0.0),
        "a_s2o3": gamma_s2o3 * max(conc["na2s2o3_m"], 0.0),
    }


def _positive_reaction_terms(conc: dict[str, float]) -> None:
    if conc["ki_m"] <= 0:
        raise PredictorError("KI final concentration must be positive")
    if conc["fecl3_m"] <= 0:
        raise PredictorError("FeCl3 final concentration must be positive")
    if conc["starch_g_l"] <= 0:
        raise PredictorError("starch final concentration must be positive")


def _safe_ratio(value: float, reference: float) -> float:
    return max(value, EPS) / max(reference, EPS)


def ferric_depletion_factor(conversion: float, alpha: float) -> float:
    """Integrated ferric-depletion slowdown.

    alpha=0 gives the first-order Fe3+ depletion limit, -ln(1-X).
    alpha>0 gives stronger critical slowing as X approaches 1, which captures
    active-Fe loss, hydrolysis/complexation, and near-endpoint detection delay
    without breaking the stoichiometric Fe3+ capacity constraint.
    """

    margin = max(1.0 - conversion, 1.0e-9)
    if abs(alpha) < 1.0e-6:
        return -math.log(margin)
    return ((1.0 / margin) ** alpha - 1.0) / alpha


def prediction_diagnostics_from_concentrations(
    conc: dict[str, float],
    params: dict[str, Any],
) -> dict[str, float]:
    """Return mechanistic intermediates used by the prediction."""

    _positive_reaction_terms(conc)
    fit = {**DEFAULT_FIT_PARAMS, **params.get("fit_params", {})}
    fixed = {**DEFAULT_FIXED, **params.get("fixed_params", {})}
    reference = params.get("reference_concentrations")
    if not reference:
        reference = final_concentrations(BASELINE_VOLUMES_ML, DEFAULT_STOCKS)

    activities = activity_state_from_concentrations(conc, params)
    starch_ratio = _safe_ratio(conc["starch_g_l"], reference["starch_g_l"])

    iodide_term = (max(activities["a_i"], EPS) ** fixed["iodide_order"]) / (
        1.0 + fit["iodide_saturation_m_inv"] * max(activities["a_i"], 0.0)
    )
    fe0 = max(conc["fecl3_m"], EPS)
    s0 = max(conc["na2s2o3_m"], 0.0)
    visual_threshold = fixed["visual_i2_threshold_m"] * (
        1.0 / starch_ratio
    ) ** fixed["starch_detection_order"]

    fe_required = fixed["stoich_fe_per_s2o3"] * s0 + 2.0 * visual_threshold
    ferric_conversion = fe_required / fe0
    ferric_remaining_m = fe0 - fe_required
    can_color = ferric_conversion < 1.0

    if can_color:
        depletion_factor = ferric_depletion_factor(
            ferric_conversion,
            fit["ferric_depletion_alpha"],
        )
        ferric_decay_rate_s_inv = 2.0 * fit["rate_constant"] * iodide_term
        kinetic_time = depletion_factor / max(ferric_decay_rate_s_inv, EPS)
        predicted = fit["lag_s"] + kinetic_time
    else:
        depletion_factor = math.inf
        ferric_decay_rate_s_inv = 2.0 * fit["rate_constant"] * iodide_term
        kinetic_time = math.inf
        predicted = fixed["no_color_time_s"]

    if not math.isfinite(predicted) or predicted <= 0:
        raise PredictorError("model produced an invalid prediction")

    return {
        **activities,
        "iodide_term": iodide_term,
        "ferric_decay_rate_s_inv": ferric_decay_rate_s_inv,
        "ferric_conversion": ferric_conversion,
        "ferric_remaining_m": ferric_remaining_m,
        "fe_required_m": fe_required,
        "visual_threshold_m": visual_threshold,
        "depletion_factor": depletion_factor,
        "kinetic_time_s": kinetic_time,
        "can_color": float(can_color),
        "predicted_time_s": float(predicted),
    }


def predict_time_from_concentrations(
    conc: dict[str, float],
    params: dict[str, Any],
) -> float:
    return prediction_diagnostics_from_concentrations(conc, params)["predicted_time_s"]


def predict_time_from_dict(volumes: dict[str, float], params: dict[str, Any]) -> float:
    """Predict color-change time in seconds for one experiment."""

    stocks = {**DEFAULT_STOCKS, **params.get("stock_concentrations", {})}
    conc = final_concentrations(volumes, stocks)
    return predict_time_from_concentrations(conc, params)


def predict_time_from_row(row: dict[str, Any] | pd.Series, params: dict[str, Any]) -> float:
    return predict_time_from_concentrations(final_concentrations_from_row(row), params)


def predict_time(experiment: Experiment, params: dict[str, Any]) -> float:
    return predict_time_from_dict(experiment.as_dict(), params)


def has_ki_variation(df: pd.DataFrame) -> bool:
    unique_ki_volumes = df["ki_ml"].nunique(dropna=True)
    unique_ki_stock = df["ki_stock_m"].nunique(dropna=True) if "ki_stock_m" in df.columns else 0
    return unique_ki_volumes > 1 or unique_ki_stock > 1


def initial_params_from_data(df: pd.DataFrame) -> dict[str, Any]:
    reference_volumes = {key: float(df[key].median()) for key in VOLUME_KEYS}
    reference_stocks = DEFAULT_STOCKS.copy()
    for column, key in STOCK_COLUMN_TO_KEY.items():
        if column in df.columns and df[column].notna().any():
            reference_stocks[key] = float(df[column].dropna().median())

    reference = final_concentrations(reference_volumes, reference_stocks)
    seed_params = {
        "schema_version": SCHEMA_VERSION,
        "model": "iodine_clock_ferric_depletion_v6",
        "stock_concentrations": reference_stocks,
        "reference_volumes_ml": reference_volumes,
        "reference_concentrations": reference,
        "fixed_params": DEFAULT_FIXED.copy(),
        "fit_params": DEFAULT_FIT_PARAMS.copy(),
    }
    return seed_params


def fit_model(df: pd.DataFrame) -> dict[str, Any]:
    """Fit local kinetic parameters while keeping the mechanistic structure fixed."""

    params = initial_params_from_data(df)

    if len(df) < 2:
        raise PredictorError("at least two calibration rows are needed to fit")
    if least_squares is None:
        raise PredictorError("scipy is required for fitting; install requirements.txt")

    y = df["time_s"].to_numpy(dtype=float)
    rows = df.to_dict(orient="records")
    fit_ki_saturation = has_ki_variation(df)

    def unpack(theta: np.ndarray) -> dict[str, Any]:
        candidate = json.loads(json.dumps(params))
        fit_params = candidate["fit_params"]
        fit_params["rate_constant"] = float(math.exp(theta[0]))
        fit_params["lag_s"] = float(math.exp(theta[1]))
        fit_params["ferric_depletion_alpha"] = float(math.exp(theta[2]))
        if fit_ki_saturation:
            fit_params["iodide_saturation_m_inv"] = float(math.exp(theta[3]))
        return candidate

    def residuals(theta: np.ndarray) -> np.ndarray:
        candidate = unpack(theta)
        pred = np.array([predict_time_from_row(row, candidate) for row in rows], dtype=float)
        data_resid = np.log(pred) - np.log(y)

        fit = candidate["fit_params"]
        prior = [
            0.04 * math.log(max(fit["ferric_depletion_alpha"], EPS) / 1.0),
            0.03 * math.log(max(fit["lag_s"], EPS) / 5.0),
        ]
        if fit_ki_saturation:
            prior.append(
                0.04
                * math.log(
                    max(fit["iodide_saturation_m_inv"], EPS)
                    / DEFAULT_FIT_PARAMS["iodide_saturation_m_inv"]
                )
            )
        return np.concatenate([data_resid, np.asarray(prior, dtype=float)])

    start = [
        math.log(params["fit_params"]["rate_constant"]),
        math.log(params["fit_params"]["lag_s"]),
        math.log(params["fit_params"]["ferric_depletion_alpha"]),
    ]
    lower = [math.log(1.0e-6), math.log(0.01), math.log(0.05)]
    upper = [math.log(1.0e9), math.log(30.0), math.log(10.0)]
    if fit_ki_saturation:
        start.append(math.log(params["fit_params"]["iodide_saturation_m_inv"]))
        lower.append(math.log(1.0e-4))
        upper.append(math.log(1.0e5))

    result = least_squares(
        residuals,
        np.asarray(start, dtype=float),
        bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        max_nfev=20000,
    )
    if not result.success:
        raise PredictorError(f"fit did not converge: {result.message}")

    fitted = unpack(result.x)
    pred = np.array([predict_time_from_row(row, fitted) for row in rows], dtype=float)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))

    fitted["training_summary"] = {
        "rows": int(len(df)),
        "rmse_s": rmse,
        "mae_s": mae,
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "columns": [column for column in df.columns if column in {*REQUIRED_COLUMNS, *STOCK_COLUMNS}],
        "ki_saturation_fitted": fit_ki_saturation,
        "note": (
            "The model uses activity-corrected Fe3+/I- kinetics and explicit "
            "Fe3+ stoichiometric depletion. Add KI/FeCl3 scans to identify "
            "iodide saturation and ferric depletion outside this calibration set."
        ),
    }
    return fitted


def load_params(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise PredictorError(f"parameter file not found: {path}")
    try:
        params = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PredictorError(f"failed to read parameter JSON: {exc}") from exc

    params.setdefault("stock_concentrations", DEFAULT_STOCKS.copy())
    params.setdefault("fixed_params", DEFAULT_FIXED.copy())
    params.setdefault(
        "reference_concentrations",
        final_concentrations(BASELINE_VOLUMES_ML, params["stock_concentrations"]),
    )
    params.setdefault("fit_params", {})

    old_fit = params["fit_params"]
    if "rate_constant" not in old_fit:
        old_fit = {"rate_constant": DEFAULT_FIT_PARAMS["rate_constant"]}
    old_fit.setdefault("lag_s", old_fit.get("background_time_s", DEFAULT_FIT_PARAMS["lag_s"]))
    old_fit.setdefault("iodide_saturation_m_inv", DEFAULT_FIT_PARAMS["iodide_saturation_m_inv"])
    old_fit.setdefault("ferric_depletion_alpha", DEFAULT_FIT_PARAMS["ferric_depletion_alpha"])

    params["fit_params"] = {**DEFAULT_FIT_PARAMS, **old_fit}
    params["fixed_params"] = {**DEFAULT_FIXED, **params["fixed_params"]}
    params.setdefault("schema_version", SCHEMA_VERSION)
    params.setdefault("model", "iodine_clock_ferric_depletion_v6")
    return params


def save_params(params: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def plot_fit(df: pd.DataFrame, params: dict[str, Any], plot_out: str | Path) -> None:
    rows = df.to_dict(orient="records")
    pred = np.array([predict_time_from_row(row, params) for row in rows], dtype=float)
    observed = df["time_s"].to_numpy(dtype=float)

    order = np.argsort(df["na2s2o3_ml"].to_numpy(dtype=float))
    x = df["na2s2o3_ml"].to_numpy(dtype=float)[order]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(x, observed[order], "o", label="observed")
    axes[0].plot(x, pred[order], "-", label="fitted")
    axes[0].set_xlabel("Na2S2O3 volume (mL)")
    axes[0].set_ylabel("Color-change time (s)")
    axes[0].set_title("Calibration fit")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    limit_max = max(float(observed.max()), float(pred.max())) * 1.08
    limit_min = min(float(observed.min()), float(pred.min())) * 0.92
    axes[1].plot(observed, pred, "o")
    axes[1].plot([limit_min, limit_max], [limit_min, limit_max], "--", color="gray")
    axes[1].set_xlim(limit_min, limit_max)
    axes[1].set_ylim(limit_min, limit_max)
    axes[1].set_xlabel("Observed (s)")
    axes[1].set_ylabel("Predicted (s)")
    axes[1].set_title("Parity check")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_out, dpi=180)
    plt.close(fig)


def canonical_sweep_variable(variable: str) -> str:
    normalized = normalize_column_name(variable)
    if normalized in {"na2s2o3", "ki", "starch", "water", "fecl3"}:
        normalized = f"{normalized}_ml"
    if normalized not in VOLUME_KEYS:
        raise PredictorError("--variable must be one of: " + ", ".join(VOLUME_KEYS))
    return normalized


def plot_sweep(
    base: Experiment,
    variable: str,
    start: float,
    stop: float,
    points: int,
    params: dict[str, Any],
    plot_out: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    if points < 2:
        raise PredictorError("--points must be at least 2")
    if start < 0 or stop < 0:
        raise PredictorError("--start and --stop must be non-negative")

    variable = canonical_sweep_variable(variable)
    xs = np.linspace(start, stop, points)
    predictions = []
    base_dict = base.as_dict()
    for value in xs:
        row = {**base_dict, variable: float(value)}
        predictions.append(predict_time_from_dict(row, params))
    ys = np.asarray(predictions, dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(xs, ys, color="#1f77b4", linewidth=2)
    ax.scatter(
        [base_dict[variable]],
        [predict_time_from_dict(base_dict, params)],
        color="#d62728",
        zorder=3,
        label="base",
    )
    ax.set_xlabel(f"{variable} (mL)")
    ax.set_ylabel("Predicted color-change time (s)")
    ax.set_title("Prediction sweep")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_out, dpi=180)
    plt.close(fig)
    return xs, ys


def command_fit(args: argparse.Namespace) -> None:
    df = read_calibration(args.calibration)
    params = fit_model(df)
    save_params(params, args.params_out)
    plot_fit(df, params, args.plot_out)
    summary = params["training_summary"]
    print(f"fit complete: rows={summary['rows']} rmse_s={summary['rmse_s']:.3f} mae_s={summary['mae_s']:.3f}")
    print(f"params: {args.params_out}")
    print(f"plot: {args.plot_out}")


def command_predict(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    params["stock_concentrations"] = stocks_from_args(args, params)
    experiment = experiment_from_args(args)
    prediction = predict_time(experiment, params)
    print(f"predicted_time_s={prediction:.3f}")


def command_sweep(args: argparse.Namespace) -> None:
    params = load_params(args.params)
    params["stock_concentrations"] = stocks_from_args(args, params)
    base = experiment_from_args(args)
    xs, ys = plot_sweep(
        base,
        args.variable,
        args.start,
        args.stop,
        args.points,
        params,
        args.plot_out,
    )
    diffs = np.diff(ys)
    monotonic = bool(np.all(diffs >= -1e-9))
    print(f"sweep complete: points={len(xs)} monotonic_non_decreasing={monotonic}")
    print(f"start_prediction_s={ys[0]:.3f} stop_prediction_s={ys[-1]:.3f}")
    print(f"plot: {args.plot_out}")


def add_volume_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--na2s2o3", type=positive_float, required=True, help="Na2S2O3 volume in mL")
    parser.add_argument("--ki", type=positive_float, required=True, help="KI volume in mL")
    parser.add_argument("--starch", type=positive_float, required=True, help="starch volume in mL")
    parser.add_argument("--water", type=positive_float, required=True, help="water volume in mL")
    parser.add_argument("--fecl3", type=positive_float, required=True, help="FeCl3 trigger volume in mL")


def add_stock_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--na2s2o3-conc", type=positive_float, help="Na2S2O3 stock concentration in mol/L")
    parser.add_argument("--ki-conc", type=positive_float, help="KI stock concentration in mol/L")
    parser.add_argument("--starch-conc", type=positive_float, help="starch stock concentration in g/L")
    parser.add_argument("--fecl3-conc", type=positive_float, help="FeCl3 stock concentration in mol/L")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iodine clock reaction predictor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="fit parameters from a calibration CSV")
    fit_parser.add_argument("--calibration", required=True, help="calibration CSV path")
    fit_parser.add_argument("--params-out", required=True, help="output JSON parameter path")
    fit_parser.add_argument("--plot-out", required=True, help="output fit plot path")
    fit_parser.set_defaults(func=command_fit)

    predict_parser = subparsers.add_parser("predict", help="predict one experiment")
    add_volume_args(predict_parser)
    add_stock_args(predict_parser)
    predict_parser.add_argument("--params", required=True, help="fitted parameter JSON path")
    predict_parser.set_defaults(func=command_predict)

    sweep_parser = subparsers.add_parser("sweep", help="scan one volume variable and plot predictions")
    sweep_parser.add_argument("--variable", required=True, help="volume variable to scan, e.g. na2s2o3_ml")
    sweep_parser.add_argument("--start", type=positive_float, required=True, help="scan start in mL")
    sweep_parser.add_argument("--stop", type=positive_float, required=True, help="scan stop in mL")
    sweep_parser.add_argument("--points", type=int, default=200, help="number of scan points")
    add_volume_args(sweep_parser)
    add_stock_args(sweep_parser)
    sweep_parser.add_argument("--params", required=True, help="fitted parameter JSON path")
    sweep_parser.add_argument("--plot-out", required=True, help="output sweep plot path")
    sweep_parser.set_defaults(func=command_sweep)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except PredictorError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
