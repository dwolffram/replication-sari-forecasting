import os
import warnings
import papermill as pm

warnings.filterwarnings(
    "ignore",
    message="the file is not specified with any extension : null",
    category=UserWarning,
    module="papermill.iorw" 
)

notebooks_to_run = [
    "plot_sari.ipynb",
    "plot_ari.ipynb",
    "plot_delays.ipynb",
    "autocorrelation.ipynb",
    "plot_nowcasts.ipynb",
    "plot_forecasts.ipynb",
    "evaluation.ipynb",
    "evaluation_quantiles.ipynb",
    "diebold_mariano.ipynb"    
]

# Determine the null device based on the operating system
null_device = "NUL" if os.name == "nt" else "/dev/null"

for notebook_path in notebooks_to_run:
    print(f"Executing {notebook_path}...")
    try:
        pm.execute_notebook(
            notebook_path,
            null_device, # direct output to the null device
            report_mode=True
        )
        print(f"Successfully executed {notebook_path}")
    except Exception as e:
        print(f"Error executing {notebook_path}: {e}")

print("All notebooks processed.")