import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.models import TSMixerModel
from epiweeks import Week


def compute_validation_score(
    model,
    targets_train,
    targets_validation,
    covariates,
    horizon,
    num_samples,
    metric,
    metric_kwargs,
    enable_optimization=True,
    sample_weight=None,
):
    # torch model: add dataloader_kwargs
    if isinstance(model, TSMixerModel):
        model.fit(
            targets_train,
            past_covariates=covariates,
            sample_weight=sample_weight,
            dataloader_kwargs={"pin_memory": torch.cuda.is_available()},
        )

    else:
        model.fit(targets_train, past_covariates=covariates, sample_weight=sample_weight)

    if isinstance(targets_train, list):
        validation_start = targets_train[0].end_time() + targets_train[0].freq
    else:
        validation_start = targets_train.end_time() + targets_train.freq

    scores = model.backtest(
        series=targets_validation,
        past_covariates=covariates,
        start=validation_start,
        forecast_horizon=horizon,
        stride=1,
        last_points_only=False,
        retrain=False,
        verbose=False,
        num_samples=num_samples,
        metric=metric,
        metric_kwargs=metric_kwargs,
        enable_optimization=enable_optimization,
    )

    score = np.mean(scores)

    return score if not np.isnan(score) else float("inf")


def get_season_start(start_year):
    return pd.to_datetime(Week(start_year, 40, system="iso").enddate())


def get_season_end(start_year):
    return pd.to_datetime(Week(start_year + 1, 39, system="iso").enddate())


def train_validation_split(series, validation_year):
    validation_end = get_season_end(validation_year)
    train_end = get_season_end(validation_year - 1)

    ts_validation = series[:validation_end]
    ts_train = series[:train_end]

    return ts_train, ts_validation


def exclude_covid_weights(targets):
    exclusion_start = pd.Timestamp("2019-06-30")
    exclusion_end = pd.Timestamp("2023-07-03")

    # Linear weights for the entire time range
    total_duration = len(targets.time_index)
    weights = np.linspace(0, 1, total_duration)

    # Adjust for exclusion period: Set weights to 0 during the exclusion period
    weights = np.where(
        (targets.time_index >= exclusion_start) & (targets.time_index <= exclusion_end),
        0,  # Weight is 0 during the exclusion period
        weights,  # Linear increase otherwise
    )

    # Create TimeSeries object for weights
    ts_weights = TimeSeries.from_times_and_values(times=targets.time_index, values=weights)

    return ts_weights
