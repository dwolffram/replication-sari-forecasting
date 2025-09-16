from typing import Sequence

from darts.models import TSMixerModel
from darts.models.forecasting.lgbm import LightGBMModel
from tqdm import tqdm

from config import (
    ALLOWED_DATA_MODES,
    ALLOWED_MODELS,
    DATA_MODE_CONFIG,
    ENCODERS,
    FORECAST_DATES,
    HORIZON,
    QUANTILES,
    RANDOM_SEEDS,
    ROOT,
    SHARED_ARGS,
    DataMode,
    ModelName,
)
from src.realtime_utils import (
    load_realtime_training_data,
)
from src.tuning import exclude_covid_weights, get_best_parameters


def train_model(
    model: ModelName = "lightgbm",
    data_mode: DataMode = "all",
    forecast_dates: Sequence[str] = FORECAST_DATES,
    random_seeds: Sequence[int] = RANDOM_SEEDS,
) -> None:
    """
    Trains a forecasting model with the specified data mode and multiple random seeds.

    For each specified forecast date and random seed, this function:
        - Loads the best hyperparameters for the chosen model and data mode.
        - Prepares the training data according to the data mode (with or without covariates or sample weights).
        - Initializes the model (either LightGBMModel or TSMixerModel) with the selected parameters.
        - Trains the model on the prepared data.
        - Saves the trained model to a file whose name encodes the date, model, mode, and seed.

    Args:
        model (ModelName): The model family to train ("lightgbm" or "tsmixer").
        data_mode (DataMode): The data configuration to use (e.g. "all", "no_covid", "no_covariates").
        forecast_dates (Sequence[str]): Dates for which to train and save models.
        random_seeds (Sequence[int]): Random seeds to use for training runs.

    Returns:
        None
    """
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported model {model!r}. Allowed: {sorted(ALLOWED_MODELS)}")
    if data_mode not in ALLOWED_DATA_MODES:
        raise ValueError(f"Invalid data_mode: {data_mode!r}. Allowed: {sorted(ALLOWED_DATA_MODES)}")

    model_name = model if data_mode == "all" else f"{model}-{data_mode}"

    use_covariates, sample_weight = DATA_MODE_CONFIG[data_mode]

    # pick best hyperparams for this family (optionally filtered)
    params, wis = get_best_parameters(
        model, use_covariates=use_covariates, sample_weight=sample_weight, clean=True, return_score=True
    )
    use_encoders = params.pop("use_encoders")

    print(
        f"\n=== Training config ===\n"
        f"  model          : {model_name}\n"
        f"  use_covariates : {use_covariates}\n"
        f"  sample_weight  : {sample_weight}\n"
        f"  forecast_dates : {min(forecast_dates)} → {max(forecast_dates)} (n={len(forecast_dates)})\n"
        f"  seeds          : {min(random_seeds)} → {max(random_seeds)} (n={len(random_seeds)})\n"
        f"=======================\n"
        f"  Parameters:"
    )
    for key, value in params.items():
        print(f"    {key}: {value}")
    print(f"\n  Validation score : {wis:.3f}\n=======================\n")

    failed_dates: list[tuple[str, str]] = []  # (date, reason)

    for forecast_date in forecast_dates:
        try:
            path = ROOT / "models" / forecast_date
            path.mkdir(parents=True, exist_ok=True)

            targets, covariates = load_realtime_training_data(as_of=forecast_date)

            # If needed, generate custom weights based on targets
            weights = exclude_covid_weights(targets) if sample_weight == "no-covid" else sample_weight

            for seed in tqdm(random_seeds, desc=f"{forecast_date}", leave=False):
                model_path = path / f"{forecast_date}-{model_name}-{seed}.pt"
                if model == "lightgbm":
                    mdl = LightGBMModel(
                        **params,
                        output_chunk_length=HORIZON,
                        add_encoders=ENCODERS if use_encoders else None,
                        likelihood="quantile",
                        quantiles=QUANTILES,
                        verbose=-1,
                        random_state=seed,
                    )
                    mdl.fit(
                        targets,
                        past_covariates=covariates if use_covariates else None,
                        sample_weight=weights,
                    )

                elif model == "tsmixer":
                    mdl = TSMixerModel(
                        **params,
                        add_encoders=ENCODERS if use_encoders else None,
                        **SHARED_ARGS,
                        random_state=seed,
                    )
                    mdl.fit(
                        targets,
                        past_covariates=covariates if use_covariates else None,
                        sample_weight=weights,
                        dataloader_kwargs={"pin_memory": False},
                    )

                mdl.save(str(model_path))  # Darts .save only accepts str

        except Exception as e:
            failed_dates.append((forecast_date, f"{type(e).__name__}: {e}"))
            print(f"[{forecast_date}] ABORTED — {type(e).__name__}: {e}")
            continue  # proceed to next forecast_date

    if not failed_dates:  # no errors at all
        print("Training completed successfully.")
    else:  # at least one date failed
        print("Completed with errors — the following dates failed:")
        for d, reason in failed_dates:
            print(f"  {d}: {reason}")
