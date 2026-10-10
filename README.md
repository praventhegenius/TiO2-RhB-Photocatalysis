# TiO2-RhB-Photocatalysis-ML

Code and data for the machine-learning analysis in our manuscript on sunlight-driven photocatalytic degradation of Rhodamine B (RhB) using anatase TiO2 nanorods (ANR).

> **Paper:** *[Manuscript title]*, [Authors], [Journal], [Year]. DOI: [to be added]

The repository contains everything needed to reproduce the models, tables and figures reported in the paper and its Supplementary Information.

## Overview

The analysis is a linked two-stage model:

| Stage | Data | Inputs | Target |
|---|---|---|---|
| **Stage 1**: catalyst + reaction conditions | Sheet 1 (N = 312 training rows) | Calcination temperature, ramp rate and time (F1-F3), reaction time (F8), RhB concentration (F9), pH (F10), co-existing ions (F11), radical scavengers (F12) | Degradation efficiency (%) |
| **Stage 2**: outdoor sunlight deployment | Sheet 2a (N = 192 rows) | Measured sunlight intensity (lux), reaction time, ambient temperature | Degradation efficiency (%) |

Stage 1 tells us which synthesis and operating conditions matter. Stage 2 takes the best catalyst (ANR-650-8-4) outdoors and models how sunlight intensity and temperature affect its performance across seasons and times of day.

Crystal phase, crystallite size, crystallinity and morphology (F4-F7) are **not** used as model inputs. In this dataset they are fully determined by the calcination recipe (F1-F3), so they are reported separately as a characterisation table.

Each stage compares nine regressors: XGBoost, LightGBM, Extra Trees, Random Forest, Gradient Boosting, SVR (RBF), Gaussian Process, KNN and a stacked ensemble.

## Repository structure

```
TiO2-RhB-Photocatalysis-ML/
├── data/
│   └── ANR_TiO2_RhB_dataset.xlsx      # full experimental dataset (Table S2)
├── src/
│   ├── 01_linked_ml_pipeline.py       # data cleaning, Stage 1 + Stage 2 models, validation, SHAP
│   ├── 02_chemistry_figures.py        # chemistry figures built on the trained models
│   ├── 03_stage2_t50_analysis.py      # t50 (time to 50 % degradation) analysis of Stage 2 data
│   └── project_style.py               # shared plotting style
├── outputs/
│   ├── tables/                        # CSV / JSON results
│   ├── figures/                       # all generated figures (300 DPI PNG)
│   └── models/final_models_v3.pkl     # fitted models used in the paper
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset

`data/ANR_TiO2_RhB_dataset.xlsx` contains five sheets:

| Sheet | Contents |
|---|---|
| `0.Fixed-Env.Vairables` | Fixed environmental constants |
| `1. ANR_02-06-2026 (Vari-Long)` | Stage 1 data: synthesis parameters, reaction conditions and degradation over time, plus stability/reusability runs |
| `(2) Fixed-Rxn.Vari` | Fixed reaction conditions for the outdoor (Stage 2) runs |
| `2(a) ANR-23-05-2026 (SunInt)` | Stage 2 data: degradation vs time under natural sunlight with measured lux |
| `(3) Temperature` | Measured ambient temperature for each weather condition and hour |

The sheet names are used directly by the scripts, so please don't rename them.

## Installation

Tested with Python 3.12.

```bash
git clone https://github.com/<your-username>/TiO2-RhB-Photocatalysis-ML.git
cd TiO2-RhB-Photocatalysis-ML
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The versions in `requirements.txt` are pinned. Other versions will probably work, but matplotlib needs to be 3.10 or newer.

## Usage

Run the scripts in order from the repository root:

```bash
python src/01_linked_ml_pipeline.py      # trains models, writes tables/figures and outputs/models/*.pkl
python src/02_chemistry_figures.py       # needs the .pkl from step 1
python src/03_stage2_t50_analysis.py     # independent, reads the dataset directly
```

All outputs are written to `outputs/`. The full run takes about 5 minutes on a normal laptop, mostly because of the repeated cross-validation and the Gaussian process model.

All random seeds are fixed (`SEED = 42`), so the numbers in `outputs/tables` should be reproduced exactly with the pinned package versions.

## Main outputs

**Tables** (`outputs/tables/`)

| File | Description |
|---|---|
| `01_data_cleaning_log.csv` | Rows removed or modified during cleaning |
| `02_stage1_metrics.csv`, `03_stage2_metrics.csv` | CV metrics for all nine models (both CV schemes) |
| `04_stability_RSD.csv` | Reproducibility (RSD) of the stability runs |
| `05_holdout_validation.csv` | Repeated 80/20 random and group hold-out tests |
| `06_shap_stage1.csv`, `06_shap_stage2.csv` | Mean absolute SHAP values per feature |
| `07_catalyst_characterisation.csv` | Synthesis recipe vs measured catalyst properties (F4-F7) |
| `08`-`13` | Catalyst ranking, interference, kinetic rate constants, linking validation, counterfactual stress test, season attribution |
| `14_t50_dataset.csv`, `15_t50_model_comparison.csv` | t50 analysis |

**Figures** (`outputs/figures/`): parity plots, model comparisons, residual diagnostics, learning curves, SHAP summaries and dependence plots, PCA, correlation heatmaps, kinetic fits, stability, operating guide heatmaps, response surfaces and the season analysis.

## Results summary

We report two cross-validation schemes for every model:

- **Repeated K-fold (RKF)**: random 5-fold splits, repeated. This is the primary estimate.
- **GroupKFold (GKF)**: whole experiments are held out together. This is a harder test of how well the model handles an experiment it has never seen, and the scores are much lower.

Selected model (XGBoost):

| | RKF R² | RKF MAE (%) | GKF R² | GKF MAE (%) |
|---|---|---|---|---|
| Stage 1 | 0.951 | 4.63 | 0.448 | 14.04 |
| Stage 2 | 0.967 | 3.88 | 0.754 | 11.04 |

The full comparison for all models is in `02_stage1_metrics.csv` and `03_stage2_metrics.csv`.

## A note on SHAP

SHAP values in this work describe how the trained model uses each input. They are a model attribution, not a mechanistic explanation of the photocatalysis, and should be read alongside the experimental evidence in the paper.

## Citation

If you use this code or data, please cite:

```bibtex
@article{[key],
  title   = {[Manuscript title]},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]},
  doi     = {[DOI]}
}
```

## License

The code is released under the MIT License (see `LICENSE`). If you reuse the dataset, please cite the paper above.

## Contact

Pravenraam Shankar, School of Computer Science and Engineering, VIT Vellore
For questions about the code, please open an issue on this repository.
