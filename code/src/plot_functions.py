from src.load_data import *
from src.load_reporting_triangle import *
import plotnine
from plotnine import ggplot, aes, geom_line, facet_wrap, labs, theme_bw, geom_ribbon, geom_point, scale_x_date, element_line, theme, element_blank
import matplotlib.pyplot as plt

def plot_split(ts, train_end, validation_end, test_end=None, target='icosari-sari-DE'):
    ts[ : train_end][target].plot(label='train')
    ts[train_end : validation_end][target].plot(label='validation')
    
    if test_end:
        ts[validation_end : test_end][target].plot(label='test')
    plt.title(target)
    plt.xlabel("")

def reshape_truth(y):
    '''
    Reformat timeseries so prediciton bands can start at the last known value at each forecast date.
    '''
    source = y.components[0].split('-')[0]
    indicator = y.components[0].split('-')[1]
    
    y = y.pd_dataframe()
    y = y.reset_index().melt(id_vars='date')

    # y['strata']   = y.component.apply(lambda x: x.split('-', 2)[-1].split('_')[0])
    y['strata']   = y.component.apply(lambda x: x.replace(f'{source}-{indicator}-', '').split('_')[0])
    y[['location', 'age_group']] = y.apply(extract_info, axis=1)

    for q in ['quantile_0.025', 'quantile_0.25', 'quantile_0.5', 'quantile_0.75', 'quantile_0.975']:
        y[q] = y.value

    y['type'] = 'truth'
    y['horizon'] = 0
    y = y.rename(columns={'date' : 'target_end_date'})
    y['forecast_date'] = y.target_end_date + pd.Timedelta(days=4)
    y = y.drop(columns=['component', 'strata', 'value'])
    
    return y


def prepare_plot_data(df, y):
    '''
    Transform dataframe to wide format and add truth data for plotting.
    '''
    df_plot = df.pivot(index = ['location', 'age_group', 'forecast_date', 'target_end_date', 'type', 'horizon'],
                       columns = 'quantile', values = 'value')
    
    df_plot.columns = ['quantile_' + str(q) for q in df_plot.columns]
    df_plot = df_plot.reset_index()
    
    y = reshape_truth(y)
    
    return pd.concat([df_plot, y], ignore_index=True)


def plot_forecasts(plot_data, stratum='states', start=0, stride=5, horizon=None):
    
    if stratum == 'national':
        plotnine.options.figure_size = (6, 2.5)
        df_temp = plot_data[(plot_data.location == 'DE') & (plot_data.age_group == '00+')]
        facet = 'location'
        ncol = 1
    
    elif stratum == 'states':
        plotnine.options.figure_size = (12, 10)
        df_temp = plot_data[(plot_data.location != 'DE') & (plot_data.age_group == '00+')]
        facet = 'location'
        ncol = 3
        
    elif stratum == 'age':
        plotnine.options.figure_size = (12, 5)
        df_temp = plot_data[(plot_data.location == 'DE') & (plot_data.age_group != '00+')]
        facet = 'age_group'
        ncol = 3
    
    y_temp = df_temp[df_temp.type == 'truth']
    
    if horizon is not None:
        df_temp = df_temp[df_temp.horizon == horizon]
        g = {}
    else:
        df_temp = df_temp[df_temp.forecast_date.isin(df_temp.forecast_date.unique()[start::stride])]
        g = {'group' : 'forecast_date'} # required if we plot multiple horizons at once
    
    return (
        ggplot(df_temp, aes(x='target_end_date')) +
        facet_wrap(facet, ncol=ncol, scales='free_y') +
        geom_ribbon(aes(ymin='quantile_0.025', ymax='quantile_0.975', **g), 
                    fill='blue', alpha=0.3) +
        geom_line(aes(y='quantile_0.5', **g), color='blue') +
        geom_line(y_temp, aes(x='target_end_date', y='quantile_0.5')) +
        geom_point(y_temp, aes(x='target_end_date', y='quantile_0.5'), size=0.75) +
        labs(x='Date', y='', title=f'{stratum.title()}{(" - Horizon: " +  str(horizon)) if horizon else ""}') +
        theme_bw()
    )


def plot_importance_lgbm(model, age_group='00+', horizon=0, max_features=None, y_size=8):
    source = model.lagged_label_names[0].split('-')[0]
    indicator = model.lagged_label_names[0].split('-')[1]

    if age_group == '00+':
        age_group = 'DE'
    horizon = horizon -1
    label = f'{source}-{indicator}-{age_group}_target_hrz{horizon}'
    estimator = model.model.estimators_[model.lagged_label_names.index(label)]
    
    feature_importances = estimator.feature_importances_
    feature_names = model.lagged_feature_names

    # Create a DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    # Sort the DataFrame by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    if max_features:
        feature_importance_df = feature_importance_df.head(max_features)
    
    plt.figure(figsize=(10, y_size))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance\n{age_group}, horizon: {horizon + 1}')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
    plt.show()
    
    
def get_sundays(start_date, end_date):
    sundays = pd.date_range(start=start_date, end=end_date, freq='W-SUN')
    return sundays


def plot_nowcasts(plot_data, stratum='national', horizon=0):
    df_frozen = load_frozen_truth(horizon)

    if stratum == 'national':
        plotnine.options.figure_size = (6, 2.5)
        df_temp = plot_data[(plot_data.location == 'DE') & (plot_data.age_group == '00+')]
        frozen_temp = df_frozen[(df_frozen.location == 'DE') & (df_frozen.age_group == '00+')]
        facet = 'location'
        ncol = 1
        
    elif stratum == 'age':
        plotnine.options.figure_size = (12, 5)
        df_temp = plot_data[(plot_data.location == 'DE') & (plot_data.age_group != '00+')]
        frozen_temp = df_frozen[(df_frozen.location == 'DE') & (df_frozen.age_group != '00+')]
        facet = 'age_group'
        ncol = 3
    
    y_temp = df_temp[df_temp.type == 'truth']
    
    df_temp = df_temp[(df_temp.horizon == horizon) & (df_temp.type != 'truth')]
    y_temp = y_temp[y_temp.target_end_date.between(df_temp.target_end_date.min(), df_temp.target_end_date.max())]
    frozen_temp = frozen_temp[frozen_temp.date.between(df_temp.target_end_date.min(), df_temp.target_end_date.max())]
    # Get all Sundays within the range of the dataset
    sundays = get_sundays(df_temp['target_end_date'].min(), df_temp['target_end_date'].max())

    return (
        ggplot(df_temp, aes(x='target_end_date')) +
        facet_wrap(facet, ncol=ncol, scales='free_y') +
        geom_ribbon(aes(ymin='quantile_0.025', ymax='quantile_0.975'), 
                    fill='blue', alpha=0.3) +
        geom_line(aes(y='quantile_0.5'), color='blue') +
        geom_line(y_temp, aes(x='target_end_date', y='quantile_0.5'), color='black') +
        geom_line(frozen_temp, aes(x='date', y='value'), color='red') +
        # geom_point(y_temp, aes(x='target_end_date', y='quantile_0.5'), size=0.75) +
        scale_x_date(
            # date_breaks='2 months',
            breaks=[pd.Timestamp('2024-01-01'), pd.Timestamp('2024-03-01'), pd.Timestamp('2024-05-01')],
            minor_breaks=sundays  # Add minor ticks at every Sunday
        ) +
        labs(x='Date', y='', title=f'{stratum.title()} — Horizon: {horizon}') +
        theme_bw() +
        theme(
            panel_grid_major_x=element_blank(),
            panel_grid_major_y=element_blank(),
            panel_grid_minor_y=element_blank(),
            panel_grid_minor=element_line(color='grey', size=0.5)
        )
    )
