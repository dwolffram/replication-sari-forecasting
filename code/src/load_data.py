import os
import pandas as pd
import numpy as np
from pathlib import Path
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from config import ROOT, QUANTILES, MODEL_NAMES, SEASON_DICT

# Functions to name the components
# For state level forecasts we remove the age group,
# the location is removed from forecasts for age groups.


def contains_number(input_str):
    return any(char.isdigit() for char in input_str)


def rename_components(ts):
    names = ["-".join(x) for x in ts.static_covariates_values()]
    names = [n[:-4] if "00+" in n else n for n in names]
    names = [n.replace("-DE", "") if contains_number(n) else n for n in names]
    return ts.with_columns_renamed(ts.components, names)


# Add nans to the beginning of a series (to concatenate with another series of different length)
def prepend_nan(series, start):
    idx = pd.date_range(
        start=start, end=series.start_time() - series.freq, freq=series.freq
    )

    fill_values = np.full((len(idx), series.n_components), np.nan)
    new_series = series.prepend_values(fill_values)
    return new_series.with_columns_renamed(new_series.columns, series.columns)


def append_nan(series, end):
    idx = pd.date_range(
        start=series.end_time() + series.freq, end=end, freq=series.freq
    )

    fill_values = np.full((len(idx), series.n_components), np.nan)
    return series.append_values(fill_values)


def resize_timeseries(ts, start_date, end_date):
    """
    Ensure that a time series has a specified start and end time. It either cuts off or pads values.
    """
    if ts.start_time() < start_date:
        # cut off
        ts = ts.drop_before(
            start_date - pd.Timedelta(days=7)
        )  # drops everything before the given date (inclusive)
    elif ts.start_time() > start_date:
        # prepend nan
        ts = prepend_nan(ts, start_date)

    if ts.end_time() > end_date:
        # cut off
        ts = ts.drop_after(end_date + pd.Timedelta(days=7))
    elif ts.end_time() < end_date:
        # append nan
        ts = append_nan(ts, end_date)

    return ts


# URL = 'https://raw.githubusercontent.com/KITmetricslab/RESPINOW-Hub/main/'

TARGETS_DICT = {
    "survstat": ["influenza"],  # , 'rsv'],
    "icosari": ["sari"],
    "agi": ["are"],
    "cvn": ["influenza", "influenza-tests"],
    "nrz": ["influenza", "influenza-tests", "rsv", "rsv-tests"],
}

LOCATION_FILTERS = {"survstat-rsv": ["DE-SN"], "agi-are": ["DE"]}


# def load_data(sources=TARGETS_DICT.keys(),
#               start_date=pd.to_datetime('2014-10-05'),
#               end_date=pd.to_datetime('2024-09-01')):

#     targets_dict = {key: TARGETS_DICT[key] for key in sources} # only use a subset of the available sources
#     ts = []
#     for source in targets_dict.keys():
#         # print(source)
#         for target in targets_dict[source]:
#             # print('-', target)
#             df = pd.read_csv(URL + f'data/{source}/{target.split("-")[0]}/latest_data-{source}-{target}.csv', parse_dates=['date'])
#             if f'{source}-{target}' in LOCATION_FILTERS.keys():
#                 df = df[df.location.isin(LOCATION_FILTERS[f'{source}-{target}'])]
#             df['source'] = source
#             df['target'] = target
#             ts_temp = TimeSeries.from_group_dataframe(df, group_cols=['source', 'target', 'location', 'age_group'],
#                                              time_col='date', value_cols='value',
#                                              freq='7D', fillna_value=0)
#             ts_temp = concatenate(ts_temp, axis=1)
#             ts_temp = resize_timeseries(ts_temp, start_date, end_date)
#             ts_temp = rename_components(ts_temp)
#             ts.append(ts_temp)

#     ts = concatenate(ts, axis=1)

#     return ts


# def load_stringency(start, end, fill_missing=None):
#     df = pd.read_csv('https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/timeseries/stringency_index_avg.csv')
#     df = df[df.country_name == 'Germany'].iloc[:, 6:].T
#     df.columns = ['SI']
#     df.index = pd.to_datetime(df.index)
#     df = df.resample('W-SUN').mean()

#     si = TimeSeries.from_dataframe(df)

#     if start < si.start_time():
#         si = prepend_nan(si, start)
#     if end > si.end_time():
#         si = append_nan(si, end)
#     if fill_missing is not None:
#         si = fill_missing_values(si, fill_missing)

#     return si


def train_validation_test_split(ts, train_end, validation_end, test_end):
    train_end = pd.to_datetime(train_end)
    validation_end = pd.to_datetime(validation_end)
    test_end = pd.to_datetime(test_end)

    # cut off end of timeseries
    ts = ts.drop_after(test_end)

    # make test set
    train, test = ts.split_after(validation_end)

    # make train and validation set
    train, validation = train.split_after(train_end)

    return train, validation, test


def target_covariate_split(ts, targets):
    covariates = [t for t in ts.components if t not in targets]
    return ts[targets], ts[covariates]


def extract_info(row):
    """
    Splits the info of the 'strata' column into 'location' and 'age_group'.
    """
    if any(char.isdigit() for char in row["strata"]):
        location = "DE"
        age_group = row["strata"]
    else:
        location = row["strata"]
        age_group = "00+"
    return pd.Series([location, age_group])


def reshape_forecast(ts_forecast, nowcast=False, deterministic=False):
    """
    Transforms a forecast from TimeSeries format to Hub format.
    """

    if deterministic:
        df_temp = ts_forecast.pd_dataframe().reset_index().melt(id_vars="date")
    else:
        df_temp = ts_forecast.quantiles_df(quantiles=QUANTILES)
        df_temp = df_temp.reset_index().melt(id_vars="date")
        df_temp["quantile"] = df_temp.component.apply(lambda x: x.split("_")[-1])

    source = ts_forecast.components[0].split("-")[0]
    indicator = ts_forecast.components[0].split("-")[1]

    df_temp["strata"] = df_temp.component.apply(
        lambda x: x.replace(f"{source}-{indicator}-", "").split("_")[0]
    )
    df_temp[["location", "age_group"]] = df_temp.apply(extract_info, axis=1)

    df_temp["horizon"] = df_temp.date.rank(method="dense").astype(int)
    if nowcast:
        df_temp.horizon = df_temp.horizon - df_temp.horizon.max()
        df_temp["forecast_date"] = df_temp.date.max() + pd.Timedelta(days=4)
    else:
        df_temp["forecast_date"] = df_temp.date.min() - pd.Timedelta(days=3)

    if deterministic:
        df_temp = df_temp.loc[df_temp.index.repeat(len(QUANTILES))].reset_index(
            drop=True
        )
        df_temp["quantile"] = pd.Series(QUANTILES * len(df_temp)).astype("str")

    df_temp["type"] = "quantile"
    df_temp = df_temp.rename(columns={"date": "target_end_date"})

    return df_temp[
        [
            "location",
            "age_group",
            "forecast_date",
            "target_end_date",
            "horizon",
            "type",
            "quantile",
            "value",
        ]
    ]


def reshape_historical_forecasts(hfc, nowcast=False, deterministic=False):
    """
    Reshapes all forecasts in 'hfc' and stores them in one dataframe.
    """
    dfs = []
    for f in hfc:
        dfs.append(reshape_forecast(f, nowcast, deterministic))

    return pd.concat(dfs)


def reshape_hfc(hfc, nowcast=False, deterministic=False):
    """
    Reshapes all historical forecasts (hfc) and stores them in one dataframe.
    This function handles different input structures: a list of forecasts (hfc),
    a dictionary of hfcs, or a list hfcs.
    """
    dfs = []

    if isinstance(hfc, dict):
        # Handle dictionary input where one hfc per target is stored (univariate model)
        for key in hfc.keys():
            dfs.append(reshape_historical_forecasts(hfc[key], nowcast, deterministic))

    elif isinstance(hfc, list):
        if all(isinstance(item, list) for item in hfc):
            # Handle list of hfcs (global model)
            for sublist in hfc:
                dfs.append(
                    reshape_historical_forecasts(sublist, nowcast, deterministic)
                )
        else:
            # Handle hfc (multivariate model)
            dfs.append(reshape_historical_forecasts(hfc, nowcast, deterministic))
    else:
        raise ValueError(
            "Input data must be a hfc, a dictionary of hfcs, or a list of hfcs."
        )

    return pd.concat(dfs)


def export_forecasts(df, team, target="survstat-influenza"):
    """
    Splits a dataframe of multiple forecasts (in hub format) by forecast_date and saves them to csv-files.
    """

    folder = f"submissions/{team}/"
    os.makedirs(folder, exist_ok=True)

    for date in df["forecast_date"].unique():
        # Filter DataFrame for the current forecast date
        filtered_df = df[df["forecast_date"] == date]

        # Save filtered DataFrame to CSV file named with forecast date
        filename = f"{folder}/{date.strftime('%Y-%m-%d')}-{target}-{team}.csv"
        filtered_df.to_csv(filename, index=False)


def filter_by_level(df, level):
    if level == "national":
        df = df[(df["location"] == "DE") & (df["age_group"] == "00+")]
    elif level == "states":
        df = df[(df["location"] != "DE")]
    elif level == "age":
        df = df[(df["location"] == "DE") & (df["age_group"] != "00+")]
    return df


def add_median(df):
    df_median = df[df["quantile"] == 0.5].copy()
    df_median["type"] = "median"
    return pd.concat([df, df_median], ignore_index=True)


def add_truth(df, source="icosari", disease="sari", target=False):
    if target:
        df_truth = pd.read_csv(ROOT / f"data/target-{source}-{disease}.csv")
    else:
        df_truth = pd.read_csv(ROOT / f"data/latest_data-{source}-{disease}.csv")

    df_truth = df_truth.rename(columns={"value": "truth"})

    df = df.merge(
        df_truth,
        how="left",
        left_on=["location", "age_group", "target_end_date"],
        right_on=["location", "age_group", "date"],
    )
    return df


def load_predictions(
    models=None,
    start="2023-11-16",
    end="2024-09-12",
    exclude_christmas=True,
    include_median=True,
    include_truth=True,
    target=True,
):
    path_forecasts = Path.cwd().parent / "forecasts"
    files = list(path_forecasts.rglob("*.csv"))

    df = pd.concat(
        (pd.read_csv(f).assign(model=f.stem.split("-", 5)[-1]) for f in files),
        ignore_index=True,
    )

    if include_median:
        df = add_median(df)
    if include_truth:
        df = add_truth(df, source="icosari", disease="sari", target=target)
    if exclude_christmas:
        df = df[df.forecast_date != "2023-12-28"]

    df.model = df.model.replace(MODEL_NAMES)

    if models is None:
        models = MODEL_NAMES.values()
    df = df[df.model.isin(models)]

    return df[df.forecast_date.between(start, end)].reset_index(drop=True)


def load_nowcasts(
    start="2023-11-16",
    end="2024-09-12",
    include_truth=True,
    exclude_christmas=True,
    quantiles=None,
):
    path_nowcasts = Path.cwd().parent / "nowcasts" / "KIT-simple_nowcast"
    files = list(path_nowcasts.rglob("*.csv"))

    df = pd.concat(
        (pd.read_csv(f).assign(model="Nowcast") for f in files),
        ignore_index=True,
    )

    if exclude_christmas:
        df = df[df.forecast_date != "2023-12-28"]

    df = df[df.forecast_date.between(start, end)].reset_index(drop=True)

    if include_truth:
        df = add_truth(df, target=True)

    if quantiles is not None:
        df = df[df["quantile"].isin(quantiles)]

    return df


def encode_static_covariates(ts, ordinal=False):
    ts.static_covariates.drop(
        columns=["source", "target", "location"], inplace=True, errors="ignore"
    )

    # Use OneHotEncoder per default, use OrdinalEncoder if 'ordinal' is True
    scaler = StaticCovariatesTransformer(
        transformer_cat=OrdinalEncoder() if ordinal else OneHotEncoder()
    )
    ts = scaler.fit_transform(ts)
    return ts


# def load_data_split(train_end, validation_end, test_end,
#                     ordinal_encoding=False, multiple_series=False, targets_as_covariates=False):

#     ts = load_data(SOURCES)
#     ts = encode_static_covariates(ts, ordinal=ordinal_encoding)
#     targets, covariates = target_covariate_split(ts, TARGETS)

#     if multiple_series:
#         # to train on multiple series instead of a multivariate series
#         targets = [targets[col] for col in targets.columns]

#         targets_train      = [target[ : train_end] for target in targets]
#         targets_validation = [target[ : validation_end] for target in targets]
#         targets_test       = [target[ : test_end] for target in targets]
#         targets_eval       = [target[EVAL_START : ] for target in targets]

#         if targets_as_covariates:
#             covariates = [ts]*len(TARGETS)
#         else:
#             covariates = [covariates]*len(TARGETS)
#     else:
#         targets_train      = targets[ : train_end]
#         targets_validation = targets[ : validation_end]
#         targets_test       = targets[ : test_end]
#         targets_eval       = targets[EVAL_START : ]

#         if targets_as_covariates:
#             covariates = ts

#     return ts, targets, targets_train, targets_validation, targets_test, targets_eval, covariates


def compute_historical_forecasts(
    model, series, covariates, start, horizon, num_samples, retrain=False
):
    return model.historical_forecasts(
        series=series,
        start=start,
        past_covariates=covariates,
        forecast_horizon=horizon,
        num_samples=num_samples,
        stride=1,
        last_points_only=False,
        retrain=retrain,
        verbose=True,
    )


def get_split_dates(test_year):
    train_end = SEASON_DICT[test_year - 2]
    validation_end = SEASON_DICT[test_year - 1]
    test_end = SEASON_DICT[test_year]

    validation_start = train_end + pd.Timedelta(1, "W")
    test_start = validation_end + pd.Timedelta(1, "W")

    return train_end, validation_start, validation_end, test_start, test_end


def combine_component_and_feature_names(ts):
    return ts.with_columns_renamed(
        ts.columns, ts.static_covariates.component.unique() + "__" + ts.columns
    )


def load_features(lag=8, multiple_series=False):
    df_features = pd.read_csv(f"../data/features/features_icosari_{lag}w.csv")
    ts_features = TimeSeries.from_group_dataframe(
        df_features,
        group_cols=["component"],
        time_col="date",
        freq="7D",
        fillna_value=0,
    )
    ts_features = [
        combine_component_and_feature_names(ts_age) for ts_age in ts_features
    ]
    ts_features = [ts_age.with_static_covariates(None) for ts_age in ts_features]

    if not multiple_series:
        ts_features = concatenate(ts_features, axis="component")

    return ts_features


def add_features(covariates, lag=8):
    ts_features = load_features(lag=lag)
    cov = covariates.slice_intersect(
        ts_features
    )  # features start a bit later because of rolling window
    ts_features = ts_features.slice_intersect(
        cov
    )  # features are longer because they cover the whole period until now
    cov = concatenate([cov.with_static_covariates(None), ts_features], axis="component")
    return cov
