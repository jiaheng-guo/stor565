# STOR 565 (Fall 2025) Group 13 Final Project

This repository contains the paper and codes for the final project of STOR 565 at UNC Chapel Hill. The project focuses on benchmarking classical and modern boosting algorithms on various datasets spanning regression and classification tasks. The final paper can be found at `./paper/main.pdf`.

**Authors**: Tony Luo, Will Kim, Sichen Li, Shang Peng, Jiaheng Guo

## Repo Structure

- `src/data_configs.py`: dataset configuration objects and loader utilities.
- `src/modeling.py`: preprocessing pipelines plus AdaBoost/GBM/XGBoost/LightGBM definitions.
- `src/evaluation.py`: evaluation loop with train/validation splits and cross-validation metrics.
- `src/utils.py`: helper to visualize experiment results.
- `src/run.py`: command-line entry point tying everything together and regenerating plots.
- `papers/`: PDF and LaTeX sources for the final report, the PDF for the project proposal, and the presentation slides.
- `static/`: static results, including the final AUC plots.

## Reproducing Experiments

```bash
conda env create -f environment.yml
conda activate stor565
python -m src.run
```

*Expected running time: On my CPU (Apple M4), the whole experiments took approximately 3 days. The Optuna hyperparameter tuning process was the most time-consuming part.*
