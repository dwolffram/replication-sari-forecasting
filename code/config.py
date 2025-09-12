from pathlib import Path
from typing import Literal, get_args

import pandas as pd
from darts.metrics.metrics import mql
from darts.utils.likelihood_models import NegativeBinomialLikelihood
from epiweeks import Week
from torch.optim import SGD, Adam, AdamW

ROOT = Path.cwd().parent

DataMode = Literal["all", "no_covid", "no_covariates"]
ModelName = Literal["lightgbm", "tsmixer"]

ALLOWED_DATA_MODES = set(get_args(DataMode))
ALLOWED_MODELS = set(get_args(ModelName))

# map data_mode → (use_covariates, sample_weight)
DATA_MODE_CONFIG = {
    "all": (True, "linear"),
    "no_covid": (True, "no-covid"),
    "no_covariates": (False, "linear"),
}

MODEL_NAMES = {
    "ensemble": "Ensemble",
    "lightgbm-coupling": "LightGBM",
    "lightgbm-no_covariates-coupling": "LightGBM-NoCovariates",
    "lightgbm-no_covid-coupling": "LightGBM-NoCovid",
    "lightgbm-oracle": "LightGBM-Oracle",
    "lightgbm-discard": "LightGBM-Discard",
    "lightgbm-naive": "LightGBM-Naive",
    "tsmixer-no_covariates-coupling": "TSMixer",
    "tsmixer-no_covariates-discard": "TSMixer-Discard",
    "tsmixer-no_covariates-naive": "TSMixer-Naive",
    "tsmixer-no_covariates-oracle": "TSMixer-Oracle",
    "tsmixer-no_covid-coupling": "TSMixer-NoCovid",
    "tsmixer-coupling": "TSMixer-Covariates",
    # "tsmixer-oracle": "TSMixer-Covariates-Oracle",
    # "tsmixer-discard": "TSMixer-Covariates-Discard",
    # "tsmixer-naive": "TSMixer-Covariates-Naive",
    "hhh4-no_covid-coupling": "hhh4-NoCovid",
    "hhh4-coupling": "hhh4",
    "hhh4-oracle": "hhh4-Oracle",
    "hhh4-discard": "hhh4-Discard",
    "hhh4-naive": "hhh4-Naive",
    "hhh4-shuffle": "hhh4-Shuffle",
    "hhh4-vincentization": "hhh4-Vincentization",
    "simple_nowcast": "Nowcast",
    "persistence": "Persistence",
    "historical": "Historical",
    "tscount-simple": "TSCount-Simple",
    "tscount-extended": "TSCount-Extended",
}

# TRAIN_END        = pd.Timestamp('2018-09-30')
# VALIDATION_START = pd.Timestamp('2018-10-07')
# VALIDATION_END   = pd.Timestamp('2019-09-29')
# TEST_START       = pd.Timestamp('2019-10-06')
# TEST_END         = pd.Timestamp('2020-09-27')
# EVAL_START       = pd.Timestamp('2023-01-01')

SEASON_DICT = {year: pd.to_datetime(Week(year + 1, 39, system="iso").enddate()) for year in range(2014, 2020)}

TARGETS = [
    "icosari-sari-DE",
    "icosari-sari-00-04",
    "icosari-sari-05-14",
    "icosari-sari-15-34",
    "icosari-sari-35-59",
    "icosari-sari-60-79",
    "icosari-sari-80+",
]

SOURCES = ["survstat", "icosari", "agi"]

SOURCE_DICT = {
    "sari": "icosari",
    "are": "agi",
    "influenza": "survstat",
    "rsv": "survstat",
}

QUANTILES = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]

METRIC = [mql for _ in QUANTILES]
METRIC_KWARGS = [{"q": q} for q in QUANTILES]


NUM_SAMPLES = 1000
HORIZON = 4

ENCODERS = {"datetime_attribute": {"future": ["month", "weekofyear"]}}

SHARED_ARGS = dict(
    output_chunk_length=HORIZON,
    likelihood=NegativeBinomialLikelihood(),
    pl_trainer_kwargs={
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "accelerator": "cpu",
        "logger": False,
    },
)

OPTIMIZER_DICT = {
    "Adam": Adam,
    "AdamW": AdamW,
    "SGD": SGD,
}

# FORECAST_DATES = pd.date_range("2023-11-16", "2024-09-12", freq="7D").strftime("%Y-%m-%d").tolist()
exclude = pd.to_datetime(["2023-12-28", "2024-01-04"])
FORECAST_DATES = pd.date_range("2023-11-16", "2024-09-12", freq="7D").difference(exclude).strftime("%Y-%m-%d").tolist()

RANDOM_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
