import pandas as pd
from darts import TimeSeries, concatenate

from config import ROOT
from src.load_data import extract_info

RT_PATH = {
    "sari": ROOT / "data/reporting_triangle-icosari-sari.csv",
    "influenza": "https://raw.githubusercontent.com/KITmetricslab/RESPINOW-Hub/main/data/survstat/influenza/reporting_triangle-survstat-influenza.csv",
}


def rename_columns(ts):
    names = [
        ("" if "value" in x else "cumsum_") + "_".join(y)
        for x, y in zip(ts.components, ts.static_covariates_values())
    ]
    names = [n.replace("DE-", "") for n in names]
    names = [n.replace("_00+", "") for n in names]
    names = [
        n.replace("DE_", "") if (len(n.replace("cumsum_", "").split("_")) >= 3) else n
        for n in names
    ]

    return ts.with_columns_renamed(ts.components, names)


def load_reporting_triangle(target="sari", add_cumsum=True, shift=True):
    path = RT_PATH[target]
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.loc[:, :"value_4w"]
    min_delay = -1 if "value_-1w" in df.columns else 0
    df.loc[:, "value_final"] = df.loc[:, f"value_{min_delay}w" : "value_4w"].sum(axis=1)

    if add_cumsum:
        # compute cumulative values
        for i in range(min_delay, 4):
            df[f"cumsum_{i}w"] = df.loc[:, f"value_{min_delay}w" : f"value_{i}w"].sum(
                axis=1
            )

        if shift:
            # shift cumsum columns
            for i in range(min_delay, 4):
                df[f"cumsum_{i}w"] = df.groupby(["age_group", "location"])[
                    f"cumsum_{i}w"
                ].shift(i - 4)

    if shift:
        # shift value columns
        for i in range(min_delay, 4):
            df[f"value_{i}w"] = df.groupby(["age_group", "location"])[
                f"value_{i}w"
            ].shift(i - 4)

    df = df.dropna()

    value_cols = ["value", "cumsum"] if add_cumsum else ["value"]

    df = pd.wide_to_long(
        df,
        stubnames=value_cols,
        sep="_",
        suffix=r".*",
        i=["location", "age_group", "year", "week", "date"],
        j="horizon",
    ).reset_index()

    ts_list = []

    for c in value_cols:
        if c == "cumsum":
            df_temp = df.dropna()
        else:
            df_temp = df

        ts = TimeSeries.from_group_dataframe(
            df_temp,
            group_cols=["location", "age_group", "horizon"],
            time_col="date",
            value_cols=[c],
            freq="7D",
            fillna_value=0,
        )
        ts = concatenate(ts, axis=1)
        ts_list.append(ts)

    ts = concatenate(ts_list, axis=1)
    ts = rename_columns(ts)

    return ts


def load_frozen_truth(horizon):
    rt = load_reporting_triangle("sari", shift=False)
    rt = rt[[c for c in rt.columns if "cumsum" in c]]

    frozen_cols = [c for c in rt.columns if f"{-1 * horizon}w" in c]

    df_frozen = rt[frozen_cols]
    df_frozen = df_frozen.pd_dataframe()
    df_frozen = df_frozen.reset_index().melt(id_vars="date")
    df_frozen["strata"] = df_frozen.component.apply(lambda x: x.split("_")[1])
    df_frozen[["location", "age_group"]] = df_frozen.apply(extract_info, axis=1)
    df_frozen = df_frozen.drop(columns="component")

    return df_frozen
