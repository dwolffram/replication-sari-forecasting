library(tidyverse)

path_submissions <- 'data/post-covid/submissions/'
models <- c('KIT-hhh4', 'lightgbm', 'tsmixer')

load_member_models <- function(models, forecast_date){
  df <- data.frame()
  for (model in models) {
    filepath <- paste0(path_submissions, model, '/', forecast_date, '-icosari-sari-', model, '.csv')
    df_temp <- read_csv(filepath) %>% 
      mutate(model = model) %>% 
      filter(type == 'quantile')
    
    df <- rbind(df, df_temp)
  }
  return(df)
}

compute_ensemble <- function(models, forecast_date){
  df <- load_member_models(models, forecast_date)
  
  df_ensemble <- df %>% 
    group_by(location, age_group, forecast_date, target_end_date, 
             horizon, type, quantile) %>%
    summarize(value = mean(value), .groups = "drop") %>%
    mutate(age_group = factor(age_group, levels = c("00+", "00-04", "05-14", "15-34", "35-59", "60-79", "80+"))) %>%
    arrange(age_group)
  
  return(df_ensemble)
}


# Example
forecast_date <- '2023-12-14'
df <- load_member_models(models, forecast_date)
df <- compute_ensemble(models, forecast_date)



# Compute all
forecast_dates <- as.character(seq(from = as.Date("2023-12-14"),
                      to = as.Date("2024-09-19"),
                      by = 7))

for (forecast_date in forecast_dates){
  df_ensemble <- compute_ensemble(models, forecast_date)
  ensemble_path <- paste0(path_submissions, 'KIT-MeanEnsemble/', forecast_date, '-icosari-sari-KIT-MeanEnsemble.csv')
  write_csv(df_ensemble, ensemble_path)
}
