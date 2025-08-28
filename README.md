## Replication package 
# Integrating Nowcasts into an Ensemble of Data-Driven Forecasting Models for SARI Hospitalizations in Germany
Daniel Wolffram, Johannes Bracher

## Repository Structure

- `code/` — Python project (primary codebase)  
  - `src/` — reusable Python modules  
  - `*.ipynb` — Jupyter notebooks  
  - `pyproject.toml`, `uv.lock` — Python environment  

- `r/` — R project (separate renv environment)  
  - `R/` — R functions 
  - `models/` — model-specific scripts  
  - `renv.lock`, `.Rprofile` — R environment  

- `data/` — input datasets 
- `figures/` — generated plots  
- `forecasts/` — model forecast outputs  
- `nowcasts/` — nowcast outputs  

## Environments

### Python
Managed via [uv](https://github.com/astral-sh/uv).  
Install dependencies from the repo root:

```bash
uv sync
```

### R
Dependencies are managed with [renv](https://rstudio.github.io/renv/).  
From the `r/` folder:

```bash
R -e "install.packages('renv'); renv::restore()"
```

## Conventions

Python code lives in `code/`.
R code lives in `r/`, with its own environment.

Shared outputs (`data/`, `forecasts/`, `nowcasts/`, `figures/`) live at the repo root and are accessible from both Python and R.
