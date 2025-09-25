## Replication package

# Integrating Nowcasts into an Ensemble of Data-Driven Forecasting Models for SARI Hospitalizations in Germany

Daniel Wolffram, Johannes Bracher

------------------------------------------------------------------------

## Repository Structure

-   `code/` — Python project (primary codebase)
    -   `src/` — reusable Python modules
    -   `*.ipynb` — Jupyter notebooks for tuning, training, evaluation, and plotting
    -   `pyproject.toml`, `uv.lock` — Python environment
-   `r/` — R project (separate renv environment)
    -   `hhh4/`, `tscount/`, `persistence/` — model-specific scripts
    -   `nowcasting/` — nowcast computation
    -   `illustrations/` — visualizations
    -   `renv.lock`, `.Rprofile` — R environment
-   `data/` — input datasets
-   `figures/` — generated plots
-   `forecasts/` — generated forecasts
-   `nowcasts/` — generated nowcasts
-   `results/` — generated results
    -   `scores/` — evaluation metrics
    -   `tuning/` — hyperparameter tuning results

------------------------------------------------------------------------

## Environments

Python code lives in `code/`. R code lives in `r/`, with its own environment. Shared inputs and outputs (`data/`, `forecasts/`, `nowcasts/`, `figures/`, `results/`) live at the repo root and are accessible from both Python and R.

### Python

The project uses [uv](https://github.com/astral-sh/uv) to manage the Python environment.\
Install `uv` on your system as follows:

-   **Linux / macOS**

``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

-   **Windows:**

``` bash
irm https://astral.sh/uv/install.ps1 | iex
```

*(In case of problems, please refer to the official [installation guide](https://docs.astral.sh/uv/getting-started/installation/).)*

------------------------------------------------------------------------

Once `uv` is installed, set up the environment from the repository root:

``` bash
uv sync
```

This will create a local `.venv/` and install all dependencies specified in `pyproject.toml` and `uv.lock`. It will also automatically install the required Python version if it is not already available on your system.

To run the notebooks with this environment, you must first register it as a Jupyter kernel:

``` bash
uv run -m ipykernel install --user --name=replication-sari
```

For interactive use, you can start **JupyterLab** inside the managed environment:

``` bash
uv run jupyter lab
```

This provides a browser-based interface, useful if you don't have a preferred IDE installed. After launching, select the kernel *replication-sari* when opening notebooks.

------------------------------------------------------------------------

### R

To ensure reproducibility, please use **R 4.5.1.** Dependencies are managed with [renv](https://rstudio.github.io/renv/). From the `r/` folder, restore the environment with:

``` bash
R -e "install.packages('renv'); renv::restore()"
```

This will restore all R package dependencies as specified in `renv.lock`. \
⚠️ Unlike `uv`, `renv` does not install R itself — you must install R 4.5.1 manually.

Note: The repository includes `.Rprofile` files (at both the root and in `r/`) that automatically activate the correct `renv` environment and anchor the [`here`](https://here.r-lib.org/) package to the repository root. This ensures that paths like `here("data", ...)` always work consistently, whether you open the whole repo or just the R subproject.

------------------------------------------------------------------------

## Running the Pipeline

The repository contains a helper script [`run_pipeline.py`](./code/run_pipeline.py) that orchestrates the execution of all notebooks and R scripts in a defined order. This ensures reproducibility of results and allows running the full pipeline or just selected parts of it.\
*(If preferred, you can also open and run the individual notebooks or R scripts manually.)*

### Pipeline structure

<!-- The pipeline runs through the following stages:

1.  **exploration**\
    Exploratory data analysis and visualization.
    -   `plot_sari.ipynb`: visualize SARI data
    -   `plot_ari.ipynb`: visualize ARI data
    -   `plot_delays.ipynb`: analyze reporting delays
    -   `autocorrelation.ipynb`: investigate correlation structure of time series
2.  **nowcasts**\
    Real-time estimation of current case counts.
    -   `nowcasting/compute_nowcasts.R`
3.  **tuning**\
    Hyperparameter tuning for machine learning models (⚠️ may take several days).
    -   `tuning_lightgbm.ipynb`
    -   `tuning_tsmixer.ipynb`
4.  **training**\
    Train final models with tuned hyperparameters.
    -   `train_models.ipynb`
5.  **forecasts**\
    Generate forecasts with different model variants.
    -   `baseline_historical.ipynb`: historical baseline model
    -   `compute_forecasts.ipynb`: compute ML-based forecasts
    -   `persistence/persistence.R`: persistence baseline
    -   `hhh4/hhh4_default.R`, `hhh4/hhh4_exclude_covid.R`, `hhh4/hhh4_naive.R`,\
        `hhh4/hhh4_oracle.R`, `hhh4/hhh4_shuffle.R`, `hhh4/hhh4_skip.R`,\
        `hhh4/hhh4_vincentization.R`: hhh4 model variants
    -   `tscount/tscount_extended.R`, `tscount/tscount_simple.R`: tscount models
6.  **ensemble**\
    Combine forecasts into an ensemble.
    -   `compute_ensemble.R`
7.  **scores**\
    Compute forecast evaluation scores.
    -   `compute_scores.ipynb`
8.  **evaluation**\
    Final visualization and evaluation of forecasts.
    -   `plot_nowcasts.ipynb`
    -   `plot_forecasts.ipynb`
    -   `evaluation.ipynb`
    -   `evaluation_quantiles.ipynb`
    -   `diebold_mariano.ipynb` -->


The pipeline runs through the following stages:

<details>

<summary><b>exploration</b>: Exploratory data analysis and visualization.</summary>

-   `plot_sari.ipynb`: visualize SARI data\
-   `plot_ari.ipynb`: visualize ARI data\
-   `plot_delays.ipynb`: analyze reporting delays\
-   `autocorrelation.ipynb`: investigate correlation structure of time series

</details>

<details>

<summary><b>nowcasts</b>: Real-time estimation of current case counts.</summary>

-   `nowcasting/compute_nowcasts.R`

</details>

<details>

<summary><b>tuning</b>: Hyperparameter tuning for machine learning models (⚠️ may take several days).</summary>

-   `tuning_lightgbm.ipynb`\
-   `tuning_tsmixer.ipynb`

</details>

<details>

<summary><b>training</b>: Train final models with tuned hyperparameters.</summary>

-   `train_models.ipynb`

</details>

<details>

<summary><b>forecasts</b>: Generate forecasts with different model variants.</summary>

-   `baseline_historical.ipynb`: historical baseline model\
-   `compute_forecasts.ipynb`: compute ML-based forecasts\
-   `persistence/persistence.R`: persistence baseline\
-   `hhh4/hhh4_default.R`, `hhh4/hhh4_exclude_covid.R`, `hhh4/hhh4_naive.R`,\
    `hhh4/hhh4_oracle.R`, `hhh4/hhh4_shuffle.R`, `hhh4/hhh4_skip.R`,\
    `hhh4/hhh4_vincentization.R`: hhh4 model variants\
-   `tscount/tscount_extended.R`, `tscount/tscount_simple.R`: tscount models

</details>

<details>

<summary><b>ensemble</b>: Combine forecasts into an ensemble.</summary>

-   `compute_ensemble.R`

</details>

<details>

<summary><b>scores</b>: Compute forecast evaluation scores.</summary>

-   `compute_scores.ipynb`

</details>

<details>

<summary><b>evaluation</b>: Final visualization and evaluation of forecasts.</summary>

-   `plot_nowcasts.ipynb`\
-   `plot_forecasts.ipynb`\
-   `evaluation.ipynb`\
-   `evaluation_quantiles.ipynb`\
-   `diebold_mariano.ipynb`

</details>

### Usage

The pipeline can be executed with different options from the repository root.\
(*We use `uv run` instead of `python` to ensure the script is executed inside the correct environment managed by `uv`.*)

-   Run the **entire pipeline**

    ``` bash
    uv run run_pipeline.py
    ```

-   Run a **single stage**

    ``` bash
    uv run run_pipeline.py --stage training
    ```

-   Run a **contiguous range of stages**

    ``` bash
    uv run run_pipeline.py --start forecasts --end scores
    ```

-   Run everything **except selected stages**

    ``` bash
    uv run run_pipeline.py --skip tuning
    ```

⚠️ **Note:** The `tuning` stage can take a very long time (several days). If you do not want to run it, use `--skip tuning`

#### Requirement: correct R version

When running the pipeline, make sure that the `Rscript` command points to the correct R version (**4.5.1**).\
On some systems, the default `Rscript` may refer to an older version of R.

You can check this with:

``` bash
Rscript --version
```
