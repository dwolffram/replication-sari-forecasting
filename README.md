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
To ensure reproducibility, please use R 4.5.1. Dependencies are managed with [renv](https://rstudio.github.io/renv/). 
From the `r/` folder:

```bash
R -e "install.packages('renv'); renv::restore()"
```

Note: The repository includes `.Rprofile` files (at both the root and in `r/`) that  
automatically activate the correct `renv` environment and anchor the [`here`](https://here.r-lib.org/)  
package to the repository root. This ensures that paths like `here("data", ...)` always work  
consistently, whether you open the whole repo or just the R subproject.


## Conventions

Python code lives in `code/`.
R code lives in `r/`, with its own environment.

Shared outputs (`data/`, `forecasts/`, `nowcasts/`, `figures/`) live at the repo root and are accessible from both Python and R.
