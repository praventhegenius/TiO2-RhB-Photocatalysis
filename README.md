# TiO2-RhB-Photocatalysis

Code and data for the paper:

> **Explainable Machine Learning for Prediction and Optimization of Photocatalytic Rhodamine B Degradation over Anatase TiO₂ Nanorods under Diurnal Sunlight Irradiation**
> Prathima Tanguturi Venkata, Pravenraam Shankar, Sanjhana Anandan, Anil Kumar Reddy Police, Chinna Gangi Reddy Nallagondu, Mamatha Kumari Murikinati, Vairamuthu Subbiah, Selva Rani Balasubramaniam, Sungjun Bae, Shankar Muthukonda Venkatakrishnan
> *Journal of Cleaner Production* (under review, manuscript JCLEPRO-D-26-27397)

The repository contains the full experimental dataset (Table S2) and the Python scripts (Table S4) used to train the models and produce the tables and figures in the paper and its Supplementary Information.

<img width="1470" height="565" alt="GA updated final" src="https://github.com/user-attachments/assets/98960655-3cb5-4413-ae37-d02db861ac49" />

## Overview

RhB decolorization over anatase TiO₂ nanorods was measured outdoors under natural sunlight at Yogi Vemana University, Kadapa, India. The data were collected in two stages and modelled separately:

| | Stage 1 | Stage 2 |
|---|---|---|
| Purpose | Catalyst synthesis and reaction conditions | Diurnal, seasonal and weather effects on the optimized catalyst (ANR-650-8-4) |
| Runs × time points | 39 runs × 8 | 24 runs × 8 |
| Observations | 312 | 192 |
| Model inputs | F1-F3, F8-F12 (8 inputs) | F8, F9, F10 (3 inputs) |
| Target | Degradation (decolorization) efficiency, % | Degradation (decolorization) efficiency, % |

In total that is 504 observations from 63 runs. A further 80 observations from 10 repeatability, reproducibility and reusability runs (rows 8a-10d in Sheet 1) are held out from training and used only for external validation.

Nine regression models are compared for each stage: XGBoost, LightGBM, Gradient Boosting, Random Forest, Extra Trees, SVR (RBF), Gaussian Process, KNN and a stacked ensemble (Ridge meta-learner on out-of-fold predictions of the eight base models).

### Feature codes

Feature codes follow Table S3 of the paper. Note that F9 and F10 mean different things in the two stages. The scripts use descriptive column names instead:

| Code (paper) | Description | Column name in code |
|---|---|---|
| **Stage 1** | | |
| F1 | Calcination temperature (°C) | `F1_CalcTemp` |
| F2 | Calcination ramp (°C min⁻¹) | `F2_CalcRamp` |
| F3 | Calcination time (h) | `F3_CalcTime` |
| F8 | Irradiation time (min) | `F8_TimeStep` |
| F9 | RhB concentration (×10⁻⁵ M) | `F9_RhBConc` |
| F10 | Solution pH | `F10_pH` |
| F11 | Co-existing ions (ppm) | `F11_Ions` |
| F12 | Radical scavenger (coded) | `F12_Scavengers` |
| **Stage 2** | | |
| F8 | Irradiation time (min) | `F8_TimeStep` |
| F9 | Average sunlight intensity (lux) | `SunlightLux` |
| F10 | Ambient temperature (°C) | `Temperature_C` |

F4-F7 (crystal phase, crystallite size, crystallinity, morphology) are kept in the dataset for reference but are not model inputs. They are written out separately as `07_catalyst_characterisation.csv`.

## Repository structure

```
TiO2-RhB-Photocatalysis-ML/
├── data/
│   └── ANR_TiO2_RhB_dataset.xlsx      # full experimental dataset (Table S2)
├── src/
│   ├── 01_linked_ml_pipeline.py       # cleaning, Stage 1 + Stage 2 models, validation, SHAP
│   ├── 02_chemistry_figures.py        # DE optimization, sensitivity, nomogram and other model-based figures
│   ├── 03_stage2_t50_analysis.py      # t50 analysis and counterfactual attribution (Stage 2)
│   └── project_style.py               # shared plotting style
├── docs/
│   └── graphical_abstract.png
├── outputs/
│   ├── tables/                        # CSV / JSON results
│   ├── figures/                       # generated figures (300 DPI PNG)
│   └── models/final_models_v3.pkl     # fitted models
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset

`data/ANR_TiO2_RhB_dataset.xlsx` is the Table S2 workbook. It has five sheets:

| Sheet | Contents |
|---|---|
| `0.Fixed-Env.Vairables` | Fixed experimental constants |
| `1. ANR_02-06-2026 (Vari-Long)` | Stage 1 runs plus the repeatability / reproducibility / reusability runs |
| `(2) Fixed-Rxn.Vari` | Fixed reaction conditions for the Stage 2 runs |
| `2(a) ANR-23-05-2026 (SunInt)` | Stage 2 runs with measured sunlight intensity |
| `(3) Temperature` | Hourly ambient temperature for each Stage 2 day |

The scripts read the sheets by name, so please don't rename them. The 0 min rows are dropped during cleaning (see `01_data_cleaning_log.csv`).

Efficiency is decolorization at 554 nm, Efficiency (%) = (C₀ − Cₜ)/C₀ × 100. TOC and COD were not measured.

## Installation

Tested with Python 3.12.

```bash
git clone https://github.com/<your-username>/TiO2-RhB-Photocatalysis-ML.git
cd TiO2-RhB-Photocatalysis-ML
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Package versions are pinned in `requirements.txt`. Other versions will probably work, but matplotlib needs to be 3.10 or newer.

## Usage

Run the scripts in order from the repository root:

```bash
python src/01_linked_ml_pipeline.py      # trains models, writes tables/figures and outputs/models/*.pkl
python src/02_chemistry_figures.py       # needs the .pkl from step 1
python src/03_stage2_t50_analysis.py     # independent, reads the dataset directly
```

Everything is written to `outputs/`. A full run takes about 5 minutes on a normal laptop. All random seeds are fixed (`SEED = 42`).

## Where each paper figure comes from

| Paper figure | Output file(s) in `outputs/figures/` | Script |
|---|---|---|
| Fig. 2 (a-d) model comparison | `fig_03a_model_comparison_stage1.png`, `fig_03b_model_comparison_stage2.png` | 01 |
| Fig. 2 (e,f) train-validation gap | `fig_08_train_test_gap.png` | 01 |
| Fig. 2 (g) learning curves | `fig_09_learning_curves.png` | 01 |
| Fig. 3 SHAP importance | `fig_06a_shap_attribution_stage1.png`, `fig_06b_shap_attribution_stage2.png` | 01 |
| Fig. 4 ion / scavenger response | `fig_N_counterfactual_stress.png` | 02 |
| Fig. 5 synthesis design space | `fig_I_synthesis_surface.png`, `fig_B_temperature_response.png` | 02 |
| Fig. 6 (a-d) residuals | `fig_07_residual_diagnostics.png` | 01 |
| Fig. 6 (e,f) DE optimization | `fig_J_cross_stage_bridge.png` | 02 |
| Fig. 7 (a-c) repeatability / reproducibility / reusability | `fig_G_stability_suite.png` | 02 |
| Fig. 7 (d-k) OVAT sensitivity | `fig_K_OAT_response_surfaces.png` | 02 |
| Fig. 8 (a) Stage 2 deployment | `fig_05_stage2_lux_temp.png` | 01 |
| Fig. 8 (b,c) operational nomogram | `fig_M_time_to_target_nomogram.png` | 02 |
| Fig. 9 t₅₀ by time of day, season, weather | `fig_T_t50_strata.png` | 03 |
| Fig. 10 counterfactual attribution | `fig_V_caseAB_comparison.png` | 03 |
| Fig. S1, S2 parity plots | `fig_01_stage1_parity.png`, `fig_02_stage2_parity.png` | 01 |
| Fig. S3 Pearson correlation | `fig_R_pearson_heatmap.png`, `fig_R2_pearson_heatmap_stage2.png` | 02 |
| Fig. S4 SHAP dependence | `fig_X_shap_dependence.png` | 02 |

The scripts also write some extra diagnostic figures that are not in the paper (PCA, OAT surfaces at other temperatures, season attribution and so on).

## Citation

If you use this code or data, please cite the paper. Full citation details will be added once it is published.

```bibtex
@article{tanguturivenkata2026anr,
  title   = {Explainable Machine Learning for Prediction and Optimization of Photocatalytic Rhodamine B Degradation over Anatase TiO2 Nanorods under Diurnal Sunlight Irradiation},
  author  = {Tanguturi Venkata, Prathima and Shankar, Pravenraam and Anandan, Sanjhana and Police, Anil Kumar Reddy and Nallagondu, Chinna Gangi Reddy and Murikinati, Mamatha Kumari and Subbiah, Vairamuthu and Balasubramaniam, Selva Rani and Bae, Sungjun and Muthukonda Venkatakrishnan, Shankar},
  journal = {Journal of Cleaner Production},
  year    = {2026},
  note    = {Under review}
}
```

## License

The code is released under the MIT License (see `LICENSE`). If you reuse the dataset, please cite the paper.

## Contact

- Code: Pravenraam Shankar, School of Computer Science and Engineering, VIT Vellore. Please open an issue on this repository.
- Paper: Shankar Muthukonda Venkatakrishnan (corresponding author), shankar@yogivemanauniversity.ac.in
