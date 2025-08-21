from src.load_data import *
import wandb
import warnings

def get_cv_chunks(ts):
    '''
    Split the given timeseries into chunks corresponding to different seasons.
    '''
    chunk_start = ts.start_time()
    chunks = []
    for t in list(SEASON_DICT.values()): # iterate season ends
        chunk = ts[chunk_start : t]
        chunks.append(chunk)
        chunk_start = chunk.end_time() + ts.freq 
    return chunks


# use start=1 for test split (so the test chunk and the previous chunk stay together)
# use start=0 for validation split
def get_cv_series(chunks, i, start=0): 
    '''
    Turns a list of chunks into a timeseries. The chunks before the i-th chunk (included) are moved to the back of the series.
    '''
    if i<start or i>=len(chunks):
        print(f'Please select a value i >= {start} and i < len(chunks).')
    elif i==len(chunks)-1:
        return concat_match_last(chunks)
    else:
        return concat_match_last(chunks[i+1:] + chunks[:i+1])
    
    
def concat_match_last(chunks):
    '''
    Concatenate a list of time series chunks. The resulting series will have a time index matching the last provided chunk.
    (The index of the previous chunks might therefore differ from the resulting series.)
    '''
    ts = chunks[-1]
    time_name = ts.time_index.name
    
    for c in chunks[-2::-1]: # backwards and skip the last one
        ts = ts.prepend_values(c.values())
        
    ts = TimeSeries.from_xarray(ts.data_array().rename({'time': time_name})) # rename time_index
        
    return ts


def create_validation_data(validation_chunks, start=0):
    '''
    Creates all possible train-validation-splits from a list of chunks and returns them as timeseries.
    The matching covariates are also returned.
    '''
    targets_train = []
    targets_validation = []
    covariates = []
    
    for j in range(start, len(validation_chunks)):
        ts_validation = get_cv_series(validation_chunks, j, start=start)
        train_end = validation_chunks[j].start_time() - validation_chunks[j].freq
        targets_val, cov = target_covariate_split(ts_validation, TARGETS)
        targets_trn = targets_val[:train_end]
        
        targets_train.append(targets_trn)
        targets_validation.append(targets_val)
        covariates.append(cov)
    
    return targets_train, targets_validation, covariates


def get_validation_data(test_year, sources=SOURCES, start=0): 
    
    if (test_year not in SEASON_DICT.keys()) or (test_year <= 2014):
        print("Error: invalid test_year.")
        return
    
    ts = load_data(sources)
    ts = encode_static_covariates(ts, ordinal=False)

    chunks = get_cv_chunks(ts)
    
    i = list(SEASON_DICT.keys()).index(test_year)
    # print(chunks[i].start_time().date())

    validation_chunks = chunks[i+1:] + chunks[:i]
    
    targets_train, targets_validation, covariates = create_validation_data(validation_chunks, start)
    
    return targets_train, targets_validation, covariates


def get_test_data(test_year, sources=SOURCES): 

    if (test_year not in SEASON_DICT.keys()) or (test_year <= 2014):
        print("Error: invalid test_year.")
        return 

    ts = load_data(sources)
    ts = encode_static_covariates(ts, ordinal=False)

    chunks = get_cv_chunks(ts)

    i = list(SEASON_DICT.keys()).index(test_year)

    ts_test = get_cv_series(chunks, i, start=1)

    train_end = chunks[i].start_time() - chunks[i].freq
    targets_test, covariates = target_covariate_split(ts_test, TARGETS)
    targets_train = targets_test[:train_end]
    
    return targets_train, targets_test, covariates


# def compute_validation_score(model, targets_train, targets_validation, covariates, horizon, num_samples, metric, metric_kwargs):
#     # model.reset_model()
#     model.fit(targets_train, past_covariates=covariates)
    
#     validation_start = targets_train.end_time() + targets_train.freq
    
#     scores = model.backtest(
#         series=targets_validation,
#         past_covariates=covariates,
#         start=validation_start,
#         forecast_horizon=horizon,
#         stride=1,
#         last_points_only=False,
#         retrain=False,
#         verbose=False,
#         num_samples=num_samples,
#         metric=metric, 
#         metric_kwargs=metric_kwargs
#     )
    
#     score = np.mean(scores) # WIS as mean of quantile scores
    
#     return score if score != np.nan else float("inf")

# def compute_validation_score(model, targets_train, targets_validation, covariates, 
#                              horizon, num_samples, metric, metric_kwargs, enable_optimization=True):
    
#     model.fit(targets_train, past_covariates=covariates)
    
#     if isinstance(targets_train, list):
#         validation_start = targets_train[0].end_time() + targets_train[0].freq
#     else:
#         validation_start = targets_train.end_time() + targets_train.freq

#     hfc = model.historical_forecasts(
#         series=targets_validation,
#         past_covariates=covariates,
#         start=validation_start,
#         forecast_horizon=horizon,
#         stride=1,
#         last_points_only=False,
#         retrain=False,
#         verbose=False,
#         num_samples=num_samples,
#         enable_optimization=enable_optimization
#     )


#     scores = model.backtest(
#         series=targets_validation,
#         past_covariates=covariates,
#         historical_forecasts=hfc,
#         start=validation_start,
#         forecast_horizon=horizon,
#         stride=1,
#         last_points_only=False,
#         retrain=False,
#         verbose=False,
#         num_samples=num_samples,
#         metric=metric, 
#         metric_kwargs=metric_kwargs
#     )

#     score = np.mean(scores)
    
#     return score if score != np.nan else float("inf")


def compute_validation_score(model, targets_train, targets_validation, covariates, 
                             horizon, num_samples, metric, metric_kwargs, enable_optimization=True, sample_weight=None):
    
    model.fit(targets_train, past_covariates=covariates, sample_weight=sample_weight)
    
    if isinstance(targets_train, list):
        validation_start = targets_train[0].end_time() + targets_train[0].freq
    else:
        validation_start = targets_train.end_time() + targets_train.freq
    
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
            category=UserWarning,
            module="sklearn.utils.validation",
        )
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
            enable_optimization=enable_optimization
        )

    score = np.mean(scores)
    
    return score if not np.isnan(score) else float("inf")


def get_best_parameters(project, model, target_metric="WIS", test_year=None, use_covariates=None, multiple_series=None, lags=None, sample_weight=None, sweep=None):
    api = wandb.Api()

    # Fetch all runs from a specific project
    runs = api.runs(f"dwolffram-karlsruhe-institute-of-technology/{project}",
                filters={"$and": [{"sweep_name" : sweep} if (sweep is not None) else None,
                                  {"config.model": model}, 
                                  {"config.test_year": test_year} if (test_year is not None) else None, 
                                  {"config.use_covariates": use_covariates} if (use_covariates is not None) else None,
                                  {"config.multiple_series": multiple_series} if (multiple_series is not None) else None,
                                  {"config.lags": lags} if (lags is not None) else None,
                                  {"config.sample_weight": sample_weight} if (sample_weight is not None) else None,
                                 ]})

    # Initialize variables to track the best run
    best_run = None
    best_metric = float('inf') 

    for run in runs:
        metric_value = run.summary.get(target_metric)
        if metric_value is not None and metric_value < best_metric:  
            best_metric = metric_value
            best_run = run
            
    print(f"{target_metric} of best run: {best_metric}")
    
    return best_run.config


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


def get_custom_weights(targets):

    len_before = len(targets[: get_season_end(2019)].time_index)

    len_after = len(targets[get_season_start(2020) :].time_index)

    weights = np.append(
        np.linspace(0.5, 0.5, len_before),
        np.linspace(0.5, 1, len_after)
    )

    ts_weights = TimeSeries.from_times_and_values(
        times=targets.time_index,
        values=weights
    )
    
    return ts_weights


def exclude_covid_weights(targets):

    exclusion_start = pd.Timestamp('2019-06-30')
    exclusion_end = pd.Timestamp('2023-07-03')

    # Linear weights for the entire time range
    total_duration = len(targets.time_index)
    weights = np.linspace(0, 1, total_duration)

    # Adjust for exclusion period: Set weights to 0 during the exclusion period
    weights = np.where(
        (targets.time_index >= exclusion_start) & (targets.time_index <= exclusion_end),
        0,  # Weight is 0 during the exclusion period
        weights  # Linear increase otherwise
    )

    # Create TimeSeries object for weights
    ts_weights = TimeSeries.from_times_and_values(
        times=targets.time_index,
        values=weights
    )
    
    return ts_weights
