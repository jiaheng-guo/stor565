# STOR 565 Group 13 Final Project

This repository contains the code and paper for the final project of STOR 565 at UNC Chapel Hill. The project focuses on benchmarking classical and modern boosting algorithms on various datasets spanning regression and classification tasks. The final paper can be found at `./paper/outputs/main.pdf`.

## Project Structure

- `src/data_configs.py`: dataset configuration objects and loader utilities.
- `src/modeling.py`: preprocessing pipelines plus AdaBoost/GBM/XGBoost/LightGBM definitions.
- `src/evaluation.py`: evaluation loop with train/validation splits and cross-validation metrics.
- `src/plotting.py`: helper to visualize ROC-AUC across datasets.
- `src/run_experiments.py`: command-line entry point tying everything together and regenerating plots.
- `src/main.ipynb`: lightweight notebook wrapper that imports the Python modules above.
- `paper/`: LaTeX sources for the final report.

## Reproducing Experiments

```bash
conda env create -f environment.yml
conda activate stor565

python -m src.run_experiments
```

The script prints progress for every dataset/algorithm pair, writes `outputs/results.csv`, dumps the classification reports to `outputs/reports.txt`, and refreshes `static/auroc.png`. Alternatively, open `src/main.ipynb` after activating the environment to run the exact same workflow interactively.
