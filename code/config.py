from pathlib import Path

import pandas as pd
from darts.metrics.metrics import mql
from darts.utils.likelihood_models import NegativeBinomialLikelihood
from epiweeks import Week

ROOT = Path.cwd().parent

MODEL_NAMES = {
    "KIT-MeanEnsemble": "Ensemble",
    "lightgbm_new": "LightGBM",
    "lightgbm_noCovariates": "LightGBM-NoCovariates",
    "lightgbm_noCovid": "LightGBM-NoCovid",
    "lightgbm_oracle": "LightGBM-Oracle",
    "lightgbm_skip": "LightGBM-Skip",
    "lightgbm_uncorrected": "LightGBM-Uncorrected",
    "tsmixer_covariates": "TSMixer",
    "tsmixer": "TSMixer-NoCovariates",
    "tsmixer_noCovid": "TSMixer-NoCovid",
    "tsmixer_oracle": "TSMixer-Oracle",
    "tsmixer_skip": "TSMixer-Skip",
    "tsmixer_uncorrected": "TSMixer-Uncorrected",
    "KIT-hhh4": "hhh4-NoCovid",
    "KIT-hhh4_all_data": "hhh4",
    "KIT-hhh4_all_data_oracle": "hhh4-Oracle",
    "KIT-hhh4_all_data_skip": "hhh4-Skip",
    "KIT-hhh4_all_data_naive": "hhh4-Uncorrected",
    "KIT-simple_nowcast": "Nowcast",
    "KIT-persistence": "Persistence",
    "baseline": "Historical",
    "KIT-tscount_negbin_seas": "TSCount-NB-S",
    "KIT-tscount_pois": "TSCount-Pois",
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
    },
)

# OPTIMIZER_DICT = {
#     "Adam" : torch.optim.Adam,
#     "AdamW" : torch.optim.AdamW,
#     "SGD": torch.optim.SGD
# }


# FORECAST_DATES = pd.date_range("2023-11-16", "2024-09-12", freq="7D").strftime("%Y-%m-%d").tolist()
exclude = pd.to_datetime(["2023-12-28", "2024-01-04"])
FORECAST_DATES = pd.date_range("2023-11-16", "2024-09-12", freq="7D").difference(exclude).strftime("%Y-%m-%d").tolist()

RANDOM_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
