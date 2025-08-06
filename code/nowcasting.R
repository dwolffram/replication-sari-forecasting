# Apply the KIT-simple_nowcast baseline model to age-stratified SARI data.
# Author: Johannes Bracher, johannes.bracher@kit.edu

# set language to English on a Linux machine
Sys.setlocale(category = "LC_TIME", locale = "en_US.UTF8")
# Windows command:
# Sys.setlocale("LC_ALL", "C")

# read in functions for the nowcasting method
source("functions_nowcasting.R")

# path of the repo. ### ADAPT THIS - will likely just be something like ".."
path_repo <- "/home/johannes/Documents/RESPINOW/RESPINOW-Hub/"
# this is just to pull in the data

# global settings:
# note: these are handled using lists so code can be adapted to other data sources

# define data sources and disease:
data_source <- "icosari"
disease <- "sari"
# the type of nowcast correction which is necessary:
type <- "additions"
# how borrowing of delays is done:
borrow_delays <- TRUE
# how borrowing of dispersion parameters is done
borrow_dispersion <- TRUE
# this just serves to suppress messages on subsetting which ar not relevant here
messages <- FALSE

# set the sizes of training data sets
# limited by number of observations (in the early part, not relevant anymore)
n_history_dispersion <- 15
n_history_expectations <- 15
max_delay <- 4
max_horizon <- 3
quantile_levels <- seq(2.5, 97.5, 2.5)/100

# dates for which to produce nowcasts (stored centrally):
forecast_dates <- as.Date(read.csv("forecast_dates.csv")$forecast_date)

# read in reporting triangle:
triangle <- read.csv(paste0(path_repo, "/data/", data_source, "/", disease, "/",
                            "reporting_triangle-", data_source, "-", disease, ".csv"),
                     colClasses = c("date" = "Date"), check.names = FALSE)
# note: we load the raw RT, preprocessing takes place inside compute_nowcast
# this ensures data are used as they were available at a given point in time

# run over forecast_dates for which to generate nowcasts:
for(i in seq_along(forecast_dates)){
  
  # select forecast_date from vector
  forecast_date <- forecast_dates[i]
  cat(as.character(forecast_dates[i]), "\n")
  
  # a place holder, to be overwritten by a data frame to store nowcasts
  all_nc <- NULL
  
  # identify age groups:
  ags <- sort(unique(triangle$age_group))
  
  # run through age groups to generate nowcasts:
  for(ag in ags){
    
    # compute nowcast.
    # note: pre-processing and subsetting of RT takes place inside
    nc <- compute_nowcast(observed = triangle,
                          location = "DE",
                          age_group = ag,
                          forecast_date = forecast_date,
                          observed2 = triangle,
                          location2 = "DE",
                          age_group2 = "00+",
                          type = type,
                          borrow_delays = borrow_delays,
                          borrow_dispersion = borrow_dispersion,
                          n_history_expectations = n_history_expectations,
                          n_history_dispersion = n_history_dispersion,
                          max_delay = max_delay,
                          quantile_levels = quantile_levels,
                          messages = messages)
    nc <- nc$result
    
    # store in all_nc:
    if(is.null(all_nc)){ # create if does not exist yet
      all_nc <- nc
    }else{
      all_nc <- rbind(all_nc, nc)
    }
  }
  
  # write out:
  write.csv(all_nc, file = paste0("KIT-baseline/", forecast_date, "-", data_source, "-",
                                  disease, "-KIT-baseline.csv"), row.names = FALSE)
}
