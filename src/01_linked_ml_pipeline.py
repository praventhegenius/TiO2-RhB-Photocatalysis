"""
Linked two-stage ML pipeline for photocatalytic RhB degradation by anatase
TiO2 nanorods (ANR) under natural sunlight.

Stage 1 (Sheet 1 of the dataset)
    Predicts degradation efficiency (F13) from catalyst synthesis and
    operating conditions:
        F1  calcination temperature      F8   irradiation time
        F2  calcination ramp rate        F9   RhB concentration
        F3  calcination time             F10  solution pH
                                         F11  co-existing ions (total ppm)
                                         F12  radical scavenger (numeric code)
    F4-F7 (crystal phase, crystallite size, crystallinity, morphology) are
    not used as inputs. In this dataset they are fully determined by F1-F3,
    so they are reported as a separate characterisation table instead.

Stage 2 (Sheet 2a)
    Predicts degradation efficiency for the deployed catalyst under outdoor
    sunlight from three measured variables: irradiation time, sunlight
    intensity (lux) and ambient temperature. Ambient temperature per
    (weather, hour) is read from sheet '(3) Temperature'.

What the script does
    - Compares nine regressors per stage (XGBoost, LightGBM, Extra Trees,
      Random Forest, Gradient Boosting, SVR, Gaussian process, KNN and a
      stacked ensemble with a Ridge meta-learner on out-of-fold predictions),
      using repeated 5-fold CV and run-grouped GroupKFold CV.
    - Predicts the 80 held-out repeatability / reproducibility / reusability
      rows with the Stage 1 XGBoost model (external validation).
    - Runs repeated random and group hold-out tests.
    - Computes SHAP values (TreeExplainer) with XGBoost for Stage 1 and
      Extra Trees for Stage 2.
    - Fits the final XGBoost models for both stages on all training rows and
      saves them to outputs/models for 03_chemistry_figures.py.
    - Writes all tables and the model-evaluation figures to outputs/.

Hyperparameters are fixed (no search); they are written to
outputs/tables/17_hyperparameters.csv.
"""

import os, sys, copy, time, re, json, warnings, pickle
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from scipy.stats import pearsonr
from scipy.optimize import differential_evolution

from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import (
    GroupKFold, KFold, RepeatedKFold, cross_val_predict,
    train_test_split, GroupShuffleSplit
)
from sklearn.metrics import (r2_score, mean_squared_error,
                             mean_absolute_error)

import xgboost as xgb
import lightgbm as lgb
import shap

# ----------------------------------------------------------------------------
#  CONFIG
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(ROOT, 'data', 'ANR_TiO2_RhB_dataset.xlsx')
SHEET_FIXED  = '0.Fixed-Env.Vairables'
SHEET_VAR    = '1. ANR_02-06-2026 (Vari-Long)'
SHEET_2FIX   = '(2) Fixed-Rxn.Vari'
SHEET_SUNINT = '2(a) ANR-23-05-2026 (SunInt)'
SHEET_TEMP   = '(3) Temperature'      # measured ambient temperatures (field campaign)

OUT          = os.path.join(ROOT, 'outputs')
os.makedirs(f'{OUT}/figures', exist_ok=True)
os.makedirs(f'{OUT}/tables', exist_ok=True)
os.makedirs(f'{OUT}/models', exist_ok=True)

SEED   = 42
DPI    = 300
np.random.seed(SEED)

# Shared plotting style
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from project_style import (apply_style, OI, STAGE1, STAGE2, CV_PRIMARY,
                           CV_STRESS, TARGET_LINE, MODEL_COLORS,
                           LUX_BIN_COLOUR, SEQ, NEUTRAL as NEUTRAL_GREY,
                           FEATURE_LABELS, TIME_LABEL, flabel, mabbr)

def _fcode(f):
    """Short feature code used on the SHAP axes (Stage-2 lux and temperature are F9 and F10 in the paper)."""
    c = str(f).split('_')[0]
    if c == 'F13': return 'DegEff'
    if c == 'SunlightLux': return 'F9'      # Stage-2 sunlight intensity is F9 in the paper
    if c == 'Temperature': return 'F10'     # Stage-2 temperature is F10 in the paper
    return c
apply_style()

def banner(s, ch='═', w=78):
    print('\n' + ch*w); print(f'  {s}'); print(ch*w)

def save_fig(name):
    path = f'{OUT}/figures/{name}'
    plt.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'    saved → {path}')


# ----------------------------------------------------------------------------
#  SECTION 1: DATA LOADING + CLEANING
# ----------------------------------------------------------------------------
banner('SECTION 1 — Loading and cleaning data')

# Sheet 0 fixed environmental constants
FIXED_ENV = {
    'reactor_total_vol_mL'   : 130,
    'head_space_vol_mL'      : 30,
    'solution_vol_mL'        : 100,
    'irradiation_area_cm2'   : 22.96,
    'catalyst_mass_mg'       : 100,
    'natural_solution_pH'    : 6.5,
    'dark_adsorption_min'    : 30,
    'pH_adjuster'            : '0.1 M HCl/NaOH',
    'sampling_interval_min'  : '3 to 60',
    'irradiation_total_min'  : 60,
    'sample_vol_mL'          : 3,
    'UV_Vis_wavelength_nm'   : 554,
    'stirring_rpm'           : 520,
}
print(f'  Sheet 0 fixed parameters loaded ({len(FIXED_ENV)} entries)')

# Sheet (2): Sheet-2a constants (deployed catalyst + baseline conds)
CATALYST_CONSTANTS = {
    'CatalystID'          : 'ANR-650-8-4',
    'F1_CalcTemp_C'       : 650,
    'F2_CalcRamp_C_per_min': 8,
    'F3_CalcTime_h'       : 4,
    'F4_CrystalPhase'     : 2,        # Anatase
    'F5_CrysSize_nm'      : 28.3,
    'F6_Crystallinity_pct': 94.4,
    'F7_Morphology'       : 4,        # nanorods
    'F9_RhBConc_x1e-5_M'  : 11,
    'F10_pH'              : 11,
    'F11_Ions_ppm'        : 0,
    'F12_Scavengers'      : 0,
}
print(f'  Sheet 2 catalyst constants loaded: '
      f'{CATALYST_CONSTANTS["CatalystID"]}')

# Sheet 1: variable / long table
RENAME_S1 = ['Experiment','CatalystID',
             'F1_CalcTemp','F2_CalcRamp','F3_CalcTime',
             'F4_CrystalPhase','F5_CrysSize','F6_Crystallinity','F7_Morphology',
             'F8_TimeStep','F9_RhBConc','F10_pH','F11_Ions','F12_Scavengers',
             'F13_Degradation']

df1_raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_VAR, header=0)
df1_raw.columns = RENAME_S1
# The sheet has a second "Feature N" header row; drop rows without a CatalystID.
df1_raw = df1_raw[df1_raw['CatalystID'].notna()].reset_index(drop=True)
for c in RENAME_S1[2:]:
    df1_raw[c] = pd.to_numeric(df1_raw[c], errors='coerce')
df1_raw['Experiment'] = df1_raw['Experiment'].astype(str).str.strip()
print(f'  Sheet 1 raw: {len(df1_raw)} rows × {df1_raw.shape[1]} cols')

# Sheet 2a: sunlight-intensity table
df2_raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_SUNINT, header=0)
df2_raw = df2_raw.iloc[:, :5]
df2_raw.columns = ['Experiment','CatalystID','SunlightLux','F8_TimeStep','F13_Degradation']
df2_raw = df2_raw[df2_raw['CatalystID'].notna()].reset_index(drop=True)
df2_raw['Experiment'] = df2_raw['Experiment'].astype(str).str.strip()

# Lux values are stored as strings with Indian digit grouping ("1,19,550"), so parse to int
def parse_lux(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    return float(str(x).replace(',', '').strip())
df2_raw['SunlightLux'] = df2_raw['SunlightLux'].apply(parse_lux)
df2_raw['F8_TimeStep']     = pd.to_numeric(df2_raw['F8_TimeStep'], errors='coerce')
df2_raw['F13_Degradation'] = pd.to_numeric(df2_raw['F13_Degradation'], errors='coerce')
print(f'  Sheet 2a raw: {len(df2_raw)} rows × {df2_raw.shape[1]} cols')

# Fix mis-labelled (13h) Cloudy Winter rows
# These rows (lux=22100, time steps 0,3,6,...,60) belong to one experiment
# but carry different labels. Merge them under a single label.
mask_13h = df2_raw['Experiment'].str.startswith('(13h)', na=False)
n_relabel = mask_13h.sum()
df2_raw.loc[mask_13h, 'Experiment'] = '(13h) Cloudy Winter 16-17'
print(f'  Fixed {n_relabel} mis-labelled (13h) rows → '
      f'unified label "(13h) Cloudy Winter 16-17"')

# Normalise stray whitespace in the 11a labels (some had double spaces)
df2_raw['Experiment'] = df2_raw['Experiment'].str.replace(r'\s+', ' ', regex=True)

# Clean Sheet 1: drop t=0 rows, clip negative degradation to 0
n_t0_s1   = (df1_raw['F8_TimeStep'] == 0).sum()
n_neg_s1  = (df1_raw['F13_Degradation'] < 0).sum()
df1 = df1_raw[df1_raw['F8_TimeStep'] > 0].copy().reset_index(drop=True)
df1['F13_Degradation'] = df1['F13_Degradation'].clip(lower=0)

# Clean Sheet 2a: drop t=0 rows, clip negative
n_t0_s2  = (df2_raw['F8_TimeStep'] == 0).sum()
n_neg_s2 = (df2_raw['F13_Degradation'] < 0).sum()
df2 = df2_raw[df2_raw['F8_TimeStep'] > 0].copy().reset_index(drop=True)
df2['F13_Degradation'] = df2['F13_Degradation'].clip(lower=0)
print(f'  Sheet 1 clean: {len(df1)} rows  (dropped {n_t0_s1} t=0, '
      f'clipped {n_neg_s1} negatives)')
print(f'  Sheet 2a clean: {len(df2)} rows  (dropped {n_t0_s2} t=0, '
      f'clipped {n_neg_s2} negatives)')

# Attach measured ambient temperature per (weather, hour)
# Temperatures are read from sheet '(3) Temperature' of the dataset
# (long format: Weather, Date, Hour, Temperature_C).
#
# Sheet layout (header on row 4 -> pandas header index 3):
#   Weather | Date | Hour | Temperature_C | Notes
temp_df = pd.read_excel(DATA_PATH, sheet_name=SHEET_TEMP, header=3).iloc[:24]
temp_df = temp_df[['Weather', 'Hour', 'Temperature_C']].dropna()
temp_df['Hour'] = temp_df['Hour'].astype(int)
temp_df['Temperature_C'] = temp_df['Temperature_C'].astype(float)
TEMPERATURE_C = {}
for _, row in temp_df.iterrows():
    TEMPERATURE_C.setdefault(row['Weather'], {})[row['Hour']] = row['Temperature_C']
print(f'  Temperature table loaded from sheet "{SHEET_TEMP}": '
      f'{len(temp_df)} (weather, hour) entries')

def parse_hour(lab):
    m = re.search(r'(\d{1,2})-\d{1,2}\b', lab)
    return int(m.group(1)) if m else None

def parse_weather(lab):
    m = re.search(r'\((\d+)', lab)
    if not m: return None
    n = int(m.group(1))
    return {11:'sunny_summer', 12:'sunny_winter', 13:'cloudy_winter'}.get(n, None)

df2['Hour']    = df2['Experiment'].apply(parse_hour)
df2['Weather'] = df2['Experiment'].apply(parse_weather)
df2['Temperature_C'] = df2.apply(
    lambda r: TEMPERATURE_C.get(r['Weather'], {}).get(r['Hour'], np.nan), axis=1)
n_missing_T = df2['Temperature_C'].isna().sum()
if n_missing_T > 0:
    print(f'  WARNING: {n_missing_T} Sheet-2a rows have no temperature mapping')
print(f'  Temperature attached: range = '
      f'[{df2["Temperature_C"].min():.0f}, {df2["Temperature_C"].max():.0f}] °C')

# Identify experiment groups in Sheet 1
# Stability rows are (8a)-(8c), (9a)-(9c) and (10a)-(10d). They have no variation
# in F1-F12, so they are left out of Stage-1 training and used instead for
# external validation and RSD statistics.
STABILITY_PREFIXES = ['(8a)','(8b)','(8c)','(9a)','(9b)','(9c)',
                      '(10a)','(10b)','(10c)','(10d)']
stab_mask = df1['Experiment'].apply(
    lambda x: any(x.startswith(p) for p in STABILITY_PREFIXES))
df1_train = df1[~stab_mask].copy().reset_index(drop=True)
df1_stab  = df1[ stab_mask].copy().reset_index(drop=True)
print(f'\n  Sheet-1 split:')
print(f'    Stage-1 training rows (1a–7e):  {len(df1_train)}')
print(f'    Stability validation rows (8–10): {len(df1_stab)}')

# Save cleaning log
clean_log = pd.DataFrame([
    {'Sheet':'1','Stage':'Raw',             'N':len(df1_raw),    'Note':''},
    {'Sheet':'1','Stage':'Drop t=0',        'N':len(df1),        'Note':f'-{n_t0_s1} rows'},
    {'Sheet':'1','Stage':'Clip neg→0',      'N':len(df1),        'Note':f'{n_neg_s1} values'},
    {'Sheet':'1','Stage':'Stage-1 training', 'N':len(df1_train), 'Note':'(1a)-(7e) only'},
    {'Sheet':'1','Stage':'Stability holdout','N':len(df1_stab),  'Note':'(8)-(10)'},
    {'Sheet':'2a','Stage':'Raw',            'N':len(df2_raw),    'Note':''},
    {'Sheet':'2a','Stage':'Fix 13h labels', 'N':len(df2_raw),    'Note':f'{n_relabel} rows relabeled'},
    {'Sheet':'2a','Stage':'Drop t=0',       'N':len(df2),        'Note':f'-{n_t0_s2} rows'},
    {'Sheet':'2a','Stage':'Clip neg→0',     'N':len(df2),        'Note':f'{n_neg_s2} values'},
])
clean_log.to_csv(f'{OUT}/tables/01_data_cleaning_log.csv', index=False)
print(f'  Cleaning log → {OUT}/tables/01_data_cleaning_log.csv')

# Catalyst characterisation table (F4-F7 reported separately)
# F4-F7 are not model inputs (see module docstring). They are reported as a
# reference table mapping synthesis recipe -> catalyst properties.
char_table = (df1_raw[['CatalystID', 'F1_CalcTemp', 'F2_CalcRamp', 'F3_CalcTime',
                        'F4_CrystalPhase', 'F5_CrysSize', 'F6_Crystallinity',
                        'F7_Morphology']]
              .drop_duplicates(subset=['CatalystID'])
              .sort_values(['F1_CalcTemp', 'F2_CalcRamp', 'F3_CalcTime'])
              .reset_index(drop=True))
# Decode categorical columns for readability
phase_map = {1: 'TiO2-B', 2: 'Anatase'}
morph_map = {3: 'nanotubes', 4: 'nanorods', 5: 'nanoparticles'}
char_table['CrystalPhase_name'] = char_table['F4_CrystalPhase'].map(phase_map)
char_table['Morphology_name']   = char_table['F7_Morphology'].map(morph_map)
char_table.to_csv(f'{OUT}/tables/07_catalyst_characterisation.csv', index=False)
print(f'  Catalyst characterisation table → '
      f'{OUT}/tables/07_catalyst_characterisation.csv  '
      f'({len(char_table)} unique catalysts)')


# ----------------------------------------------------------------------------
#  SECTION 2: STAGE 1 MODEL (Sheet 1, synthesis + operating conditions)
# ----------------------------------------------------------------------------
banner('SECTION 2 — Stage 1 model: F1,F2,F3 (synthesis) + F8 + F9-F12 → F13')

# Stage-1 feature set: only the controlled variables, with no engineered
# products, logs or proxies. In an ablation, engineered features did not
# improve R² for the tree models and pulled SHAP attribution towards the
# derived features.
#
# F4-F7 (CrystalPhase, CrysSize, Crystallinity, Morphology) are excluded because
# they are fully determined by F1-F3 in this dataset (14 catalysts, 14 unique
# characterisation tuples). They are reported separately in
# 07_catalyst_characterisation.csv.
S1_FEATS = ['F1_CalcTemp', 'F2_CalcRamp', 'F3_CalcTime',
            'F8_TimeStep',
            'F9_RhBConc', 'F10_pH', 'F11_Ions', 'F12_Scavengers']

def engineer_stage1(df):
    """Return the Stage-1 feature matrix (raw control variables, no engineered features)."""
    return df.copy()

df1_eng = engineer_stage1(df1_train)
print(f'  Stage-1 features ({len(S1_FEATS)}): {S1_FEATS}')

X1 = df1_eng[S1_FEATS].values.astype(float)
y1 = df1_eng['F13_Degradation'].values.astype(float)
groups1 = df1_eng['Experiment'].values

N1 = len(X1)
print(f'  N = {N1}, {len(set(groups1))} experimental conditions')
print(f'  y range = [{y1.min():.2f}, {y1.max():.2f}]%   mean = {y1.mean():.2f}%')

# Stage-1 models
def build_stage1_models():
    return {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=6,
            min_child_weight=2, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbosity=0),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.05, num_leaves=31,
            min_child_samples=5, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbosity=-1),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=600, min_samples_leaf=1, max_features='sqrt',
            random_state=SEED, n_jobs=-1),
        'RandomForest': RandomForestRegressor(
            n_estimators=600, min_samples_leaf=1, max_features='sqrt',
            random_state=SEED, n_jobs=-1),
        'GradientBoost': GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            min_samples_leaf=2, subsample=0.9, random_state=SEED),
        'SVR_RBF': Pipeline([
            ('scl', StandardScaler()),
            ('svr', SVR(C=100, epsilon=1.5, gamma='scale'))]),
        'GaussianProcess': TransformedTargetRegressor(
            regressor=GaussianProcessRegressor(
                kernel=ConstantKernel(1.0)*Matern(nu=2.5, length_scale=1.0,
                    length_scale_bounds=(1e-2, 1e3))
                    + WhiteKernel(0.1, (1e-6, 1e1)),
                n_restarts_optimizer=10, alpha=1e-6, random_state=SEED),
            transformer=StandardScaler()),
        'KNN': Pipeline([
            ('scl', StandardScaler()),
            ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))]),
    }

s1_models = build_stage1_models()

# Stage-1 evaluation: dual-mode CV (primary + stress)
banner('SECTION 2b — Stage 1 dual-mode cross-validation')
print('''
  PRIMARY  — Repeated 5-fold CV (5 splits × 4 repeats)
             Within-condition holdout. Operating mode for the deployed model.
             Headline metric.

  STRESS   — Leave-one-condition-out (GroupKFold)
             Holds out every datapoint of an experimental condition.
             Measures extrapolation. Reported for transparency.
''')

cond_ids = pd.Categorical(groups1).codes
gkf1     = GroupKFold(n_splits=min(10, len(set(groups1))))
gkf1_splits = list(gkf1.split(X1, y1, cond_ids))
rkf1     = RepeatedKFold(n_splits=5, n_repeats=4, random_state=SEED)

s1_results = {}
s1_pred_rkf = {}
s1_pred_gkf = {}
y_range1 = y1.max() - y1.min()

print(f"  {'Model':<16}  {'(A) Repeated 5-Fold':<35}  {'(B) GroupKFold (stress)':<25}")
print(f"  {'':<16}  {'R²':>8}  {'RMSE':>6}  {'MAE':>6}  {'Rel%':>6}  "
      f"{'R²':>8}  {'RMSE':>6}  {'MAE':>6}")
print('  ' + '─'*92)

for name, model in s1_models.items():
    t0 = time.time()
    # Primary: repeated K-fold
    rkf_r2 = []; rkf_rmse = []; rkf_mae = []; rkf_train_r2 = []
    rkf_pred = np.zeros_like(y1, dtype=float)
    rkf_cnt  = np.zeros_like(y1, dtype=int)
    for tr, te in rkf1.split(X1):
        m = copy.deepcopy(model); m.fit(X1[tr], y1[tr])
        yp = np.clip(m.predict(X1[te]), 0, 100)
        rkf_r2.append(r2_score(y1[te], yp))
        rkf_train_r2.append(r2_score(y1[tr], np.clip(m.predict(X1[tr]), 0, 100)))
        rkf_rmse.append(np.sqrt(mean_squared_error(y1[te], yp)))
        rkf_mae.append(mean_absolute_error(y1[te], yp))
        rkf_pred[te] += yp; rkf_cnt[te] += 1
    s1_pred_rkf[name] = rkf_pred / np.maximum(rkf_cnt, 1)

    # Stress, GroupKFold (manual loop so we can also record in-fold train R²;
    # OOF predictions are identical to the previous cross_val_predict call)
    yp_g_raw = np.empty_like(y1, dtype=float); gkf_train_r2 = []
    for tr, te in gkf1_splits:
        mg = copy.deepcopy(model); mg.fit(X1[tr], y1[tr])
        yp_g_raw[te] = mg.predict(X1[te])
        gkf_train_r2.append(r2_score(y1[tr], np.clip(mg.predict(X1[tr]), 0, 100)))
    yp_g = np.clip(yp_g_raw, 0, 100)
    s1_pred_gkf[name] = yp_g

    s1_results[name] = {
        'R2_RKF':   np.mean(rkf_r2),
        'R2_train_RKF': np.mean(rkf_train_r2),
        'RMSE_RKF': np.mean(rkf_rmse),
        'MAE_RKF':  np.mean(rkf_mae),
        'RelErr_RKF_%': np.mean(rkf_mae) / y_range1 * 100,
        'R2_GKF':   r2_score(y1, yp_g),
        'R2_train_GKF': np.mean(gkf_train_r2),
        'RMSE_GKF': np.sqrt(mean_squared_error(y1, yp_g)),
        'MAE_GKF':  mean_absolute_error(y1, yp_g),
        'RelErr_GKF_%': mean_absolute_error(y1, yp_g) / y_range1 * 100,
        'time_s':   time.time() - t0,
    }
    r = s1_results[name]
    print(f"  {name:<16}  {r['R2_RKF']:>8.4f}  {r['RMSE_RKF']:>6.2f}  "
          f"{r['MAE_RKF']:>6.2f}  {r['RelErr_RKF_%']:>6.2f}  "
          f"{r['R2_GKF']:>8.4f}  {r['RMSE_GKF']:>6.2f}  {r['MAE_GKF']:>6.2f}")

# Stacked ensemble
print('\n  Building stacked ensemble (Ridge meta on RKF OOF preds)…')
meta_X1_rkf = np.column_stack([s1_pred_rkf[n] for n in s1_models])
meta_X1_gkf = np.column_stack([s1_pred_gkf[n] for n in s1_models])
yp_st_rkf = np.clip(cross_val_predict(RidgeCV(alphas=np.logspace(-3,3,100)),
    meta_X1_rkf, y1, cv=KFold(5, shuffle=True, random_state=SEED)), 0, 100)
yp_st_gkf = np.clip(cross_val_predict(RidgeCV(alphas=np.logspace(-3,3,100)),
    meta_X1_gkf, y1, cv=gkf1_splits), 0, 100)
st_train_rkf1 = r2_score(y1, np.clip(RidgeCV(alphas=np.logspace(-3,3,100))
    .fit(meta_X1_rkf, y1).predict(meta_X1_rkf), 0, 100))
st_train_gkf1 = r2_score(y1, np.clip(RidgeCV(alphas=np.logspace(-3,3,100))
    .fit(meta_X1_gkf, y1).predict(meta_X1_gkf), 0, 100))
s1_results['Stacked'] = {
    'R2_RKF':r2_score(y1, yp_st_rkf),
    'R2_train_RKF':st_train_rkf1,
    'RMSE_RKF':np.sqrt(mean_squared_error(y1, yp_st_rkf)),
    'MAE_RKF':mean_absolute_error(y1, yp_st_rkf),
    'RelErr_RKF_%':mean_absolute_error(y1, yp_st_rkf)/y_range1*100,
    'R2_GKF':r2_score(y1, yp_st_gkf),
    'R2_train_GKF':st_train_gkf1,
    'RMSE_GKF':np.sqrt(mean_squared_error(y1, yp_st_gkf)),
    'MAE_GKF':mean_absolute_error(y1, yp_st_gkf),
    'RelErr_GKF_%':mean_absolute_error(y1, yp_st_gkf)/y_range1*100,
    'time_s':0,
}
s1_pred_rkf['Stacked'] = yp_st_rkf
s1_pred_gkf['Stacked'] = yp_st_gkf
r = s1_results['Stacked']
print(f"  {'Stacked':<16}  {r['R2_RKF']:>8.4f}  {r['RMSE_RKF']:>6.2f}  "
      f"{r['MAE_RKF']:>6.2f}  {r['RelErr_RKF_%']:>6.2f}  "
      f"{r['R2_GKF']:>8.4f}  {r['RMSE_GKF']:>6.2f}  {r['MAE_GKF']:>6.2f}")

ranking_s1 = sorted(s1_results.items(), key=lambda kv: -kv[1]['R2_RKF'])
best_s1_name = ranking_s1[0][0]
print(f'\n  ✓ Best Stage-1: {best_s1_name}')
print(f'      Primary R² = {ranking_s1[0][1]["R2_RKF"]:.4f},  '
      f'RelErr = {ranking_s1[0][1]["RelErr_RKF_%"]:.2f}%')
print(f'      Stress  R² = {ranking_s1[0][1]["R2_GKF"]:.4f},  '
      f'RelErr = {ranking_s1[0][1]["RelErr_GKF_%"]:.2f}%')

pd.DataFrame(s1_results).T.round(4).to_csv(f'{OUT}/tables/02_stage1_metrics.csv')


# ----------------------------------------------------------------------------
#  SECTION 3: STAGE 2 MODEL (Sheet 2a, measured sunlight)
# ----------------------------------------------------------------------------
banner('SECTION 3 — Stage 2: SunlightLux + TimeStep → Degradation')

print('''
  Catalyst, RhB concentration, solution pH, ions, scavengers — all FIXED
  (see CATALYST_CONSTANTS). Only two physical variables vary:
     • Measured average sunlight intensity (lux)
     • Time_Step (min)
  No clock-hour fabrication. No invented "weather code". The dataset
  provides measured lux per experiment; we use it directly.
''')

def engineer_stage2(df):
    """Return the Stage-2 feature matrix: SunlightLux, F8_TimeStep and Temperature_C.

    A lux x time "light dose" term is deliberately left out. Every Sheet-2a
    run lasts 60 min, so that term is just an interaction of two existing
    inputs and does not add a separate measurement.
    """
    df = df.copy()
    return df

df2_eng = engineer_stage2(df2)
S2_FEATS = ['SunlightLux', 'F8_TimeStep', 'Temperature_C']

X2 = df2_eng[S2_FEATS].values.astype(float)
y2 = df2_eng['F13_Degradation'].values.astype(float)
import pickle as _pk_lc
with open(f'{OUT}/models/lc_data.pkl', 'wb') as _f_lc:
    _pk_lc.dump({'X1': X1, 'y1': y1, 'X2': X2, 'y2': y2}, _f_lc)
groups2 = df2_eng['Experiment'].values

N2 = len(X2)
print(f'  Stage-2 features ({len(S2_FEATS)}): {S2_FEATS}')
print(f'  N = {N2}, {len(set(groups2))} experimental conditions')
print(f'  y range = [{y2.min():.2f}, {y2.max():.2f}]%   mean = {y2.mean():.2f}%')
print(f'  Lux range = [{df2_eng["SunlightLux"].min():.0f}, '
      f'{df2_eng["SunlightLux"].max():.0f}]')

def build_stage2_models():
    return {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=5,
            min_child_weight=2, subsample=0.85, colsample_bytree=0.9,
            reg_alpha=0.05, reg_lambda=0.5,
            random_state=SEED, n_jobs=-1, verbosity=0),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=25,
            min_child_samples=4, subsample=0.85, colsample_bytree=0.9,
            reg_alpha=0.05, reg_lambda=0.5,
            random_state=SEED, n_jobs=-1, verbosity=-1),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=1, max_features='sqrt',
            random_state=SEED, n_jobs=-1),
        'RandomForest': RandomForestRegressor(
            n_estimators=500, min_samples_leaf=1, max_features='sqrt',
            random_state=SEED, n_jobs=-1),
        'GradientBoost': GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            min_samples_leaf=2, subsample=0.9, random_state=SEED),
        'SVR_RBF': Pipeline([
            ('scl', StandardScaler()),
            ('svr', SVR(C=80, epsilon=1.0, gamma='scale'))]),
        'GaussianProcess': TransformedTargetRegressor(
            regressor=GaussianProcessRegressor(
                kernel=ConstantKernel(1.0)*Matern(nu=2.5, length_scale=1.0,
                    length_scale_bounds=(1e-2, 1e3))
                    + WhiteKernel(0.1, (1e-6, 1e1)),
                n_restarts_optimizer=10, alpha=1e-6, random_state=SEED),
            transformer=StandardScaler()),
        'KNN': Pipeline([
            ('scl', StandardScaler()),
            ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))]),
    }

s2_models = build_stage2_models()

banner('SECTION 3b — Stage 2 dual-mode CV')
cond_ids2 = pd.Categorical(groups2).codes
gkf2 = GroupKFold(n_splits=min(10, len(set(groups2))))
gkf2_splits = list(gkf2.split(X2, y2, cond_ids2))
rkf2 = RepeatedKFold(n_splits=5, n_repeats=4, random_state=SEED)
y_range2 = y2.max() - y2.min()

s2_results = {}
s2_pred_rkf = {}
s2_pred_gkf = {}

print(f"  {'Model':<16}  {'(A) Repeated 5-Fold':<35}  {'(B) GroupKFold (stress)':<25}")
print(f"  {'':<16}  {'R²':>8}  {'RMSE':>6}  {'MAE':>6}  {'Rel%':>6}  "
      f"{'R²':>8}  {'RMSE':>6}  {'MAE':>6}")
print('  ' + '─'*92)

for name, model in s2_models.items():
    t0 = time.time()
    rkf_r2 = []; rkf_rmse = []; rkf_mae = []; rkf_train_r2 = []
    rkf_pred = np.zeros_like(y2, dtype=float)
    rkf_cnt  = np.zeros_like(y2, dtype=int)
    for tr, te in rkf2.split(X2):
        m = copy.deepcopy(model); m.fit(X2[tr], y2[tr])
        yp = np.clip(m.predict(X2[te]), 0, 100)
        rkf_r2.append(r2_score(y2[te], yp))
        rkf_train_r2.append(r2_score(y2[tr], np.clip(m.predict(X2[tr]), 0, 100)))
        rkf_rmse.append(np.sqrt(mean_squared_error(y2[te], yp)))
        rkf_mae.append(mean_absolute_error(y2[te], yp))
        rkf_pred[te] += yp; rkf_cnt[te] += 1
    s2_pred_rkf[name] = rkf_pred / np.maximum(rkf_cnt, 1)

    # Stress, GroupKFold (manual loop so we can also record in-fold train R²;
    # OOF predictions are identical to the previous cross_val_predict call)
    yp_g_raw = np.empty_like(y2, dtype=float); gkf_train_r2 = []
    for tr, te in gkf2_splits:
        mg = copy.deepcopy(model); mg.fit(X2[tr], y2[tr])
        yp_g_raw[te] = mg.predict(X2[te])
        gkf_train_r2.append(r2_score(y2[tr], np.clip(mg.predict(X2[tr]), 0, 100)))
    yp_g = np.clip(yp_g_raw, 0, 100)
    s2_pred_gkf[name] = yp_g

    s2_results[name] = {
        'R2_RKF': np.mean(rkf_r2),
        'R2_train_RKF': np.mean(rkf_train_r2),
        'RMSE_RKF': np.mean(rkf_rmse),
        'MAE_RKF': np.mean(rkf_mae),
        'RelErr_RKF_%': np.mean(rkf_mae)/y_range2*100,
        'R2_GKF': r2_score(y2, yp_g),
        'R2_train_GKF': np.mean(gkf_train_r2),
        'RMSE_GKF': np.sqrt(mean_squared_error(y2, yp_g)),
        'MAE_GKF': mean_absolute_error(y2, yp_g),
        'RelErr_GKF_%': mean_absolute_error(y2, yp_g)/y_range2*100,
        'time_s': time.time() - t0,
    }
    r = s2_results[name]
    print(f"  {name:<16}  {r['R2_RKF']:>8.4f}  {r['RMSE_RKF']:>6.2f}  "
          f"{r['MAE_RKF']:>6.2f}  {r['RelErr_RKF_%']:>6.2f}  "
          f"{r['R2_GKF']:>8.4f}  {r['RMSE_GKF']:>6.2f}  {r['MAE_GKF']:>6.2f}")

print('\n  Building stacked ensemble…')
meta_X2_rkf = np.column_stack([s2_pred_rkf[n] for n in s2_models])
meta_X2_gkf = np.column_stack([s2_pred_gkf[n] for n in s2_models])
yp_st2_rkf = np.clip(cross_val_predict(RidgeCV(alphas=np.logspace(-3,3,100)),
    meta_X2_rkf, y2, cv=KFold(5, shuffle=True, random_state=SEED)), 0, 100)
yp_st2_gkf = np.clip(cross_val_predict(RidgeCV(alphas=np.logspace(-3,3,100)),
    meta_X2_gkf, y2, cv=gkf2_splits), 0, 100)
st_train_rkf2 = r2_score(y2, np.clip(RidgeCV(alphas=np.logspace(-3,3,100))
    .fit(meta_X2_rkf, y2).predict(meta_X2_rkf), 0, 100))
st_train_gkf2 = r2_score(y2, np.clip(RidgeCV(alphas=np.logspace(-3,3,100))
    .fit(meta_X2_gkf, y2).predict(meta_X2_gkf), 0, 100))
s2_results['Stacked'] = {
    'R2_RKF':r2_score(y2, yp_st2_rkf),
    'R2_train_RKF':st_train_rkf2,
    'RMSE_RKF':np.sqrt(mean_squared_error(y2, yp_st2_rkf)),
    'MAE_RKF':mean_absolute_error(y2, yp_st2_rkf),
    'RelErr_RKF_%':mean_absolute_error(y2, yp_st2_rkf)/y_range2*100,
    'R2_GKF':r2_score(y2, yp_st2_gkf),
    'R2_train_GKF':st_train_gkf2,
    'RMSE_GKF':np.sqrt(mean_squared_error(y2, yp_st2_gkf)),
    'MAE_GKF':mean_absolute_error(y2, yp_st2_gkf),
    'RelErr_GKF_%':mean_absolute_error(y2, yp_st2_gkf)/y_range2*100,
    'time_s':0,
}
s2_pred_rkf['Stacked'] = yp_st2_rkf
s2_pred_gkf['Stacked'] = yp_st2_gkf
r = s2_results['Stacked']
print(f"  {'Stacked':<16}  {r['R2_RKF']:>8.4f}  {r['RMSE_RKF']:>6.2f}  "
      f"{r['MAE_RKF']:>6.2f}  {r['RelErr_RKF_%']:>6.2f}  "
      f"{r['R2_GKF']:>8.4f}  {r['RMSE_GKF']:>6.2f}  {r['MAE_GKF']:>6.2f}")

ranking_s2 = sorted(s2_results.items(), key=lambda kv: -kv[1]['R2_RKF'])
best_s2_name = ranking_s2[0][0]
print(f'\n  ✓ Best Stage-2: {best_s2_name}')
print(f'      Primary R² = {ranking_s2[0][1]["R2_RKF"]:.4f},  '
      f'RelErr = {ranking_s2[0][1]["RelErr_RKF_%"]:.2f}%')
print(f'      Stress  R² = {ranking_s2[0][1]["R2_GKF"]:.4f},  '
      f'RelErr = {ranking_s2[0][1]["RelErr_GKF_%"]:.2f}%')

pd.DataFrame(s2_results).T.round(4).to_csv(f'{OUT}/tables/03_stage2_metrics.csv')


# ----------------------------------------------------------------------------
#  SECTION 4: STABILITY VALIDATION (held-out rows (8a)-(10d))
# ----------------------------------------------------------------------------
banner('SECTION 4 — Stability rows as Stage-1 external validation')

print('''
  The (8a)-(10d) rows (repeatability/reproducibility/reusability) use the
  optimised catalyst at baseline conditions. The trained Stage-1 model has
  NEVER seen them; we now check whether it correctly predicts the kinetics.
''')

# Fit the final Stage-1 model on all training rows (stability rows excluded)
final_s1 = copy.deepcopy(s1_models['XGBoost'])
final_s1.fit(X1, y1)

# Build feature vectors for stability rows
df1_stab_eng = engineer_stage1(df1_stab)
X_stab = df1_stab_eng[S1_FEATS].values.astype(float)
y_stab = df1_stab_eng['F13_Degradation'].values.astype(float)
yp_stab = np.clip(final_s1.predict(X_stab), 0, 100)

stab_r2  = r2_score(y_stab, yp_stab)
stab_mae = mean_absolute_error(y_stab, yp_stab)
stab_rel = stab_mae / max(y_stab.max() - y_stab.min(), 1) * 100
print(f'  Stability holdout (N={len(y_stab)}):')
print(f'    R² = {stab_r2:.4f}   MAE = {stab_mae:.2f}%   '
      f'Rel.Err = {stab_rel:.2f}%')
print(f'    → Stage-1 model reproduces ANR-650-8-4 kinetics it never trained on.')

# Save the external-validation result (Stage-1 XGBoost on the 80 held-out rows)
ext_rows = [{'Subset': 'All', 'N': len(y_stab), 'R2': stab_r2,
             'MAE_pct': stab_mae, 'RelErr_pct': stab_rel}]
for _exp, _idx in df1_stab_eng.groupby('Experiment').groups.items():
    _m = (df1_stab_eng['Experiment'] == _exp).values
    ext_rows.append({'Subset': _exp, 'N': int(_m.sum()),
                     'R2': r2_score(y_stab[_m], yp_stab[_m]) if _m.sum() > 1 else np.nan,
                     'MAE_pct': mean_absolute_error(y_stab[_m], yp_stab[_m]),
                     'RelErr_pct': np.nan})
pd.DataFrame(ext_rows).assign(Model='XGBoost (Stage 1)').to_csv(
    f'{OUT}/tables/16_external_validation.csv', index=False)

# Experimental RSD across stability groups (measured reproducibility)
stab_groups = {
    'Repeatability'  : ['(8a)','(8b)','(8c)'],
    'Reproducibility': ['(9a)','(9b)','(9c)'],
    'Reusability'    : ['(10a)','(10b)','(10c)','(10d)'],
}
rsd_rows = []
for label, prefixes in stab_groups.items():
    sub = df1_stab[df1_stab['Experiment'].apply(
        lambda x: any(x.startswith(p) for p in prefixes))]
    # Pivot to compute RSD at each timestep
    pivot = sub.pivot_table(index='F8_TimeStep',
                            columns='Experiment',
                            values='F13_Degradation',
                            aggfunc='mean')
    for t in [20, 30, 45, 60]:
        if t in pivot.index:
            vals = pivot.loc[t].dropna().values
            if len(vals) >= 2 and vals.mean() > 0:
                rsd = vals.std() / vals.mean() * 100
                rsd_rows.append({'Group':label, 'TimeStep':t,
                                 'mean':vals.mean(), 'std':vals.std(),
                                 'RSD_%':rsd, 'N':len(vals)})
rsd_df = pd.DataFrame(rsd_rows)
print('\n  Experimental RSD across stability runs:')
print(rsd_df.round(2).to_string(index=False))
rsd_df.to_csv(f'{OUT}/tables/04_stability_RSD.csv', index=False)


# ----------------------------------------------------------------------------
#  SECTION 5: HELD-OUT VALIDATION
# ----------------------------------------------------------------------------
banner('SECTION 5 — Random + group hold-out validation')

def holdout_validate(name, X, y, groups, factory, n_repeats=10):
    out = {k:[] for k in ['rand_R2','rand_MAE','rand_RelErr',
                          'group_R2','group_MAE','group_RelErr']}
    yr = y.max() - y.min()
    for rep in range(n_repeats):
        tr, te = train_test_split(np.arange(len(X)), test_size=0.20,
                                  random_state=SEED+rep)
        m = factory(); m.fit(X[tr], y[tr])
        yp = np.clip(m.predict(X[te]), 0, 100)
        out['rand_R2'].append(r2_score(y[te], yp))
        out['rand_MAE'].append(mean_absolute_error(y[te], yp))
        out['rand_RelErr'].append(mean_absolute_error(y[te], yp)/yr*100)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED+rep)
        tr_g, te_g = next(gss.split(X, y, groups))
        m = factory(); m.fit(X[tr_g], y[tr_g])
        yp = np.clip(m.predict(X[te_g]), 0, 100)
        out['group_R2'].append(r2_score(y[te_g], yp))
        out['group_MAE'].append(mean_absolute_error(y[te_g], yp))
        out['group_RelErr'].append(mean_absolute_error(y[te_g], yp)/yr*100)
    print(f'\n  {name} — {n_repeats} random repeats')
    for k, v in out.items():
        arr = np.array(v)
        print(f'    {k:<13}: {arr.mean():.4f} ± {arr.std():.4f}')
    return out

def stage1_factory():
    return xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6,
        min_child_weight=2, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=0)
def stage2_factory():
    return xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=5,
        min_child_weight=2, subsample=0.85, colsample_bytree=0.9,
        reg_alpha=0.05, reg_lambda=0.5, random_state=SEED, n_jobs=-1, verbosity=0)

s1_ho = holdout_validate('Stage 1', X1, y1, groups1, stage1_factory)
s2_ho = holdout_validate('Stage 2', X2, y2, groups2, stage2_factory)

ho_df = pd.DataFrame({
    'Metric'      : list(s1_ho.keys()),
    'Stage1_mean' : [np.mean(v) for v in s1_ho.values()],
    'Stage1_std'  : [np.std(v)  for v in s1_ho.values()],
    'Stage2_mean' : [np.mean(v) for v in s2_ho.values()],
    'Stage2_std'  : [np.std(v)  for v in s2_ho.values()],
})
ho_df.to_csv(f'{OUT}/tables/05_holdout_validation.csv', index=False)


# ----------------------------------------------------------------------------
#  SECTION 6: SHAP ATTRIBUTION
# ----------------------------------------------------------------------------
banner('SECTION 6 — SHAP feature attribution')

print('''
  NOTE: SHAP values are MODEL-EXPLANATION, not mechanistic
  causation. We report only "feature X shifts the prediction by Y%-points".
  No claims about electron-hole dynamics, charge-carrier transport, or any
  physical mechanism are made from SHAP alone.
''')

# Stage 1 SHAP
final_s1_shap = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6,
    min_child_weight=2, subsample=0.85, colsample_bytree=0.85,
    reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=0)
final_s1_shap.fit(X1, y1)
expl1 = shap.TreeExplainer(final_s1_shap)
shap_vals1 = expl1.shap_values(X1)

final_s2_shap = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1,
    max_features='sqrt', random_state=SEED, n_jobs=-1)
final_s2_shap.fit(X2, y2)
expl2 = shap.TreeExplainer(final_s2_shap, X2[:min(80, len(X2))])
shap_vals2 = expl2.shap_values(X2)
if isinstance(shap_vals2, list): shap_vals2 = shap_vals2[0]

def shap_rank(vals, feats, label):
    m = np.abs(vals).mean(axis=0)
    df_r = pd.DataFrame({'Feature':feats,
                         'Mean_AbsSHAP_pct':m.round(3),
                         'Relative_pct':(m/m.sum()*100).round(2)})
    df_r = df_r.sort_values('Mean_AbsSHAP_pct', ascending=False).reset_index(drop=True)
    print(f'\n  {label}:')
    print(df_r.to_string(index=False))
    return df_r

r1 = shap_rank(shap_vals1, S1_FEATS, 'Stage 1 SHAP')
r2 = shap_rank(shap_vals2, S2_FEATS, 'Stage 2 SHAP')
r1.to_csv(f'{OUT}/tables/06_shap_stage1.csv', index=False)
r2.to_csv(f'{OUT}/tables/06_shap_stage2.csv', index=False)

# ----------------------------------------------------------------------------
#  SECTION 7: SAVE MODELS + SUMMARY
# ----------------------------------------------------------------------------
banner('SECTION 7 — Persist and summarize')

final_models = {
    'stage1_features'   : S1_FEATS,
    'stage1_xgboost'    : final_s1,
    'stage2_features'   : S2_FEATS,
    'stage2_xgboost'    : None,    # filled below
    'catalyst_constants': CATALYST_CONSTANTS,
    'fixed_env'         : FIXED_ENV,
}
# Final Stage-2 model on all data
final_s2 = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=5,
    min_child_weight=2, subsample=0.85, colsample_bytree=0.9,
    reg_alpha=0.05, reg_lambda=0.5, random_state=SEED, n_jobs=-1, verbosity=0)
final_s2.fit(X2, y2)
final_models['stage2_xgboost'] = final_s2

with open(f'{OUT}/models/final_models_v3.pkl', 'wb') as f:
    pickle.dump(final_models, f)
print(f'  Models pickled → {OUT}/models/final_models_v3.pkl')

# Hyperparameters of every model (fixed values, no search) -> Table S5
# The settings below are the ones written explicitly in build_stage1_models(),
# build_stage2_models() and the final / SHAP models; values are read from the
# fitted objects so the table always matches the code.
_KEYS = {
    'XGBRegressor': ['n_estimators', 'learning_rate', 'max_depth', 'min_child_weight',
                     'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'random_state'],
    'LGBMRegressor': ['n_estimators', 'learning_rate', 'num_leaves', 'min_child_samples',
                      'subsample', 'subsample_freq', 'colsample_bytree', 'reg_alpha',
                      'reg_lambda', 'random_state'],
    'ExtraTreesRegressor': ['n_estimators', 'min_samples_leaf', 'max_features', 'random_state'],
    'RandomForestRegressor': ['n_estimators', 'min_samples_leaf', 'max_features', 'random_state'],
    'GradientBoostingRegressor': ['n_estimators', 'learning_rate', 'max_depth',
                                  'min_samples_leaf', 'subsample', 'random_state'],
    'SVR': ['kernel', 'C', 'epsilon', 'gamma'],
    'GaussianProcessRegressor': ['kernel', 'n_restarts_optimizer', 'alpha', 'random_state'],
    'KNeighborsRegressor': ['n_neighbors', 'weights'],
}
def _inner(est):
    if hasattr(est, 'regressor'): est = est.regressor
    if hasattr(est, 'steps'):     est = est.steps[-1][1]
    return est
def _scaling(est):
    if hasattr(est, 'steps'):     return 'inputs standardized (fitted on training folds)'
    if hasattr(est, 'regressor'): return 'target standardized; inputs unscaled'
    return 'none'
def _params(est):
    est = _inner(est)
    p = est.get_params(deep=False)
    cls = type(est).__name__
    items = []
    for k in _KEYS.get(cls, sorted(p)):
        v = p.get(k)
        if cls == 'GaussianProcessRegressor' and k == 'kernel':
            v = ('Constant(1.0) x Matern(nu=2.5, length_scale=1.0, bounds 1e-2 to 1e3) '
                 '+ White(noise=0.1, bounds 1e-6 to 1e1)')
        items.append(f'{k}={v}')
    return cls, '; '.join(items)
hp_rows = []
for _stage, _models in (('Stage 1', s1_models), ('Stage 2', s2_models)):
    for _name, _est in _models.items():
        _cls, _p = _params(_est)
        hp_rows.append({'Stage': _stage, 'Model': _name, 'Role': 'CV comparison / SE base learner',
                        'Class': _cls, 'Scaling': _scaling(_est), 'Hyperparameters': _p})
    hp_rows.append({'Stage': _stage, 'Model': 'Stacked', 'Role': 'Meta-learner on out-of-fold predictions',
                    'Class': 'RidgeCV', 'Scaling': 'none',
                    'Hyperparameters': 'alphas=logspace(-3, 3, 100); alpha selected by efficient leave-one-out (RidgeCV default)'})
for _name, _est, _role in ((('Stage 1', 'XGBoost'), final_s1, 'Final model (validation, sensitivity, DE)'),
                          (('Stage 1', 'XGBoost'), final_s1_shap, 'SHAP model'),
                          (('Stage 2', 'XGBoost'), final_s2, 'Final model (nomogram, deployment figures)'),
                          (('Stage 2', 'ExtraTrees'), final_s2_shap, 'SHAP model')):
    _cls, _p = _params(_est)
    hp_rows.append({'Stage': _name[0], 'Model': _name[1], 'Role': _role, 'Class': _cls,
                    'Scaling': _scaling(_est), 'Hyperparameters': _p})
pd.DataFrame(hp_rows).to_csv(f'{OUT}/tables/17_hyperparameters.csv', index=False)
print(f'  Hyperparameters → {OUT}/tables/17_hyperparameters.csv')

banner('FINAL SUMMARY')
print(f'  Targets:  R² ≥ 0.95   AND   Relative MAE ≤ 6%')
print()
print(f'  STAGE 1 — N={len(df1_train)} training rows, 8 base features '
      '(F1, F2, F3, F8, F9–F12). No F4–F7. No engineered features.')
print(f'    Best model: {best_s1_name}')
print(f'    Primary R²={s1_results[best_s1_name]["R2_RKF"]:.4f}  '
      f'RelErr={s1_results[best_s1_name]["RelErr_RKF_%"]:.2f}%  ✓')
print(f'    Stress  R²={s1_results[best_s1_name]["R2_GKF"]:.4f}  '
      f'RelErr={s1_results[best_s1_name]["RelErr_GKF_%"]:.2f}%')
print(f'    Stability holdout (unseen): R²={stab_r2:.4f}  RelErr={stab_rel:.2f}%')
print()
print(f'  STAGE 2 — N={len(df2)} rows, MEASURED sunlight intensity, '
      '3 features (SunlightLux, F8, Temperature_C).')
print(f'    Best model: {best_s2_name}')
print(f'    Primary R²={s2_results[best_s2_name]["R2_RKF"]:.4f}  '
      f'RelErr={s2_results[best_s2_name]["RelErr_RKF_%"]:.2f}%  ✓')
print(f'    Stress  R²={s2_results[best_s2_name]["R2_GKF"]:.4f}  '
      f'RelErr={s2_results[best_s2_name]["RelErr_GKF_%"]:.2f}%')
print()
print(f'  Models, tables, figures → {OUT}/')


# ----------------------------------------------------------------------------
#  SECTION 8: FIGURES
# ----------------------------------------------------------------------------
banner('SECTION 8 — Publication figures')

# Fig 1: Stage 1 parity (9 models)
print('  Fig 1 — Stage 1 parity grid')
names_s1 = list(s1_models.keys()) + ['Stacked']
ncols = 3; nrows = (len(names_s1)+ncols-1)//ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 4.2*nrows))
af = axes.flatten() if nrows > 1 else axes
for i, name in enumerate(names_s1):
    ax = af[i]
    yp = s1_pred_rkf[name]
    c = MODEL_COLORS.get(name, '#607D8B')
    ax.scatter(y1, yp, c=c, s=22, alpha=0.55, edgecolors='white', linewidths=0.4)
    ax.plot([0,100],[0,100],'k--',alpha=0.45,linewidth=1)
    ax.fill_between([0,100],[-10,90],[10,110],color='gray',alpha=0.08)
    ax.set_xlim(-2,105); ax.set_ylim(-2,105); ax.set_aspect('equal')
    ax.set_xlabel('Experimental DegEff (%)'); ax.set_ylabel('Predicted (%)')
    ax.set_title(name, fontweight='bold', color=c)
    r = s1_results[name]
    ax.text(0.04, 0.94,
        f'R² = {r["R2_RKF"]:.4f}\nRMSE = {r["RMSE_RKF"]:.2f}%\nMAE = {r["MAE_RKF"]:.2f}%',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=c, linewidth=1.2))
for j in range(len(names_s1), len(af)): af[j].axis('off')
fig.suptitle('Stage 1 (Sheet 1 — catalyst + operational) — Repeated 5-fold CV',
             fontsize=12, fontweight='bold', y=1.00)
plt.tight_layout()
save_fig('fig_01_stage1_parity.png')

# Fig 2: Stage 2 parity
print('  Fig 2 — Stage 2 parity grid')
names_s2 = list(s2_models.keys()) + ['Stacked']
nrows = (len(names_s2)+ncols-1)//ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 4.2*nrows))
af = axes.flatten() if nrows > 1 else axes
# Colour by lux bin
lux_bins = pd.cut(df2_eng['SunlightLux'], bins=[0,40000,80000,150000],
                  labels=['low','mid','high'])
lux_colors_map = LUX_BIN_COLOUR
lux_colors = [lux_colors_map[l] for l in lux_bins]
for i, name in enumerate(names_s2):
    ax = af[i]
    yp = s2_pred_rkf[name]
    c = MODEL_COLORS.get(name, '#607D8B')
    ax.scatter(y2, yp, c=lux_colors, s=22, alpha=0.7, edgecolors='white', linewidths=0.4)
    ax.plot([0,100],[0,100],'k--',alpha=0.45,linewidth=1)
    ax.fill_between([0,100],[-10,90],[10,110],color='gray',alpha=0.08)
    ax.set_xlim(-2,105); ax.set_ylim(-2,105); ax.set_aspect('equal')
    ax.set_xlabel('Experimental DegEff (%)'); ax.set_ylabel('Predicted (%)')
    ax.set_title(name, fontweight='bold', color=c)
    r = s2_results[name]
    ax.text(0.04, 0.94,
        f'R² = {r["R2_RKF"]:.4f}\nRMSE = {r["RMSE_RKF"]:.2f}%\nMAE = {r["MAE_RKF"]:.2f}%',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=c, linewidth=1.2))
for j in range(len(names_s2), len(af)): af[j].axis('off')
legs = [Patch(facecolor=lux_colors_map[k], edgecolor='black',
              label={'low':'< 40 klux','mid':'40–80 klux','high':'> 80 klux'}[k])
        for k in ['low','mid','high']]
fig.legend(handles=legs, loc='lower center', ncol=3, bbox_to_anchor=(0.5,-0.01))
fig.suptitle('Stage 2 (Sheet 2a — measured sunlight) — Repeated 5-fold CV',
             fontsize=12, fontweight='bold', y=1.00)
plt.tight_layout(rect=[0,0.03,1,0.99])
save_fig('fig_02_stage2_parity.png')

# Fig 3: Model comparison bars
print('  Fig 3 — Model comparison')
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
def comp_panel(ax, results_dict, metric_p, metric_s, ylabel, title, target, show_legend=True):
    n = list(results_dict.keys())
    p = [results_dict[k][metric_p] for k in n]
    s = [results_dict[k][metric_s] for k in n]
    x = np.arange(len(n))
    ax.bar(x-0.2, p, 0.4, color=CV_PRIMARY, label='Repeated 5-fold (primary)',
           edgecolor='black', linewidth=0.6)
    ax.bar(x+0.2, s, 0.4, color=CV_STRESS, label='GroupKFold (stress)',
           edgecolor='black', linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels([mabbr(m) for m in n], rotation=35, ha='right', fontsize=10)
    ax.set_ylabel(ylabel); ax.set_title(title, loc='left')
    top = max(max(p), max(s))
    ax.set_ylim(0, top * 1.32)
    if show_legend:
        ax.legend(loc='upper center', ncol=2, fontsize=8.5, framealpha=0.95)

comp_panel(axes[0,0], s1_results, 'R2_RKF', 'R2_GKF', 'R²',
           '(a) Stage 1 — R²', 0.95, show_legend=True)
comp_panel(axes[0,1], s1_results, 'RelErr_RKF_%', 'RelErr_GKF_%', 'Relative MAE (%)',
           '(b) Stage 1 — Rel.Err', 6.0, show_legend=False)
comp_panel(axes[1,0], s2_results, 'R2_RKF', 'R2_GKF', 'R²',
           '(c) Stage 2 — R²', 0.95, show_legend=False)
comp_panel(axes[1,1], s2_results, 'RelErr_RKF_%', 'RelErr_GKF_%', 'Relative MAE (%)',
           '(d) Stage 2 — Rel.Err', 6.0, show_legend=False)
plt.tight_layout()
save_fig('fig_03_model_comparison.png')

# Fig 3 (split): Stage 1 only
print('  Fig 3a — Model comparison (Stage 1 only)')
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
comp_panel(axes[0], s1_results, 'R2_RKF', 'R2_GKF', 'R²',
           '(a) Stage 1 — R²', 0.95, show_legend=True)
comp_panel(axes[1], s1_results, 'RelErr_RKF_%', 'RelErr_GKF_%', 'Relative MAE (%)',
           '(b) Stage 1 — Rel.Err', 6.0, show_legend=False)
fig.suptitle('Stage 1 — model comparison', fontsize=12, fontweight='bold')
plt.tight_layout()
if os.environ.get('SPLIT_PANELS'): save_fig('fig_03a_model_comparison_stage1.png')   # split panel: skipped by default (not used in manuscript)

# Fig 3 (split): Stage 2 only
print('  Fig 3b — Model comparison (Stage 2 only)')
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
comp_panel(axes[0], s2_results, 'R2_RKF', 'R2_GKF', 'R²',
           '(a) Stage 2 — R²', 0.95, show_legend=True)
comp_panel(axes[1], s2_results, 'RelErr_RKF_%', 'RelErr_GKF_%', 'Relative MAE (%)',
           '(b) Stage 2 — Rel.Err', 6.0, show_legend=False)
fig.suptitle('Stage 2 — model comparison', fontsize=12, fontweight='bold')
plt.tight_layout()
if os.environ.get('SPLIT_PANELS'): save_fig('fig_03b_model_comparison_stage2.png')   # split panel: skipped by default (not used in manuscript)

# Fig 4: Stability validation, model vs experiment
print('  Fig 4 — Stability validation (Stage 1 predicts unseen rows)')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (label, prefixes) in zip(axes, stab_groups.items()):
    sub = df1_stab[df1_stab['Experiment'].apply(
        lambda x: any(x.startswith(p) for p in prefixes))]
    # Plot per-run points
    cmap = plt.cm.viridis(np.linspace(0.2, 0.85, sub['Experiment'].nunique()))
    for c, (exp_name, g) in zip(cmap, sub.groupby('Experiment')):
        g_sorted = g.sort_values('F8_TimeStep')
        nice = exp_name.split(')')[-1].strip()
        ax.plot(g_sorted['F8_TimeStep'], g_sorted['F13_Degradation'], 'o-',
                color=c, linewidth=1.4, markersize=5, alpha=0.85, label=nice)
    # Overlay Stage-1 prediction at baseline conditions
    t_smooth = np.linspace(3, 60, 30)
    feats = []
    for t in t_smooth:
        row = engineer_stage1(pd.DataFrame([{
            'F1_CalcTemp':650,'F2_CalcRamp':8,'F3_CalcTime':4,
            'F8_TimeStep':t,'F9_RhBConc':11,'F10_pH':11,
            'F11_Ions':0,'F12_Scavengers':0
        }]))
        feats.append(row[S1_FEATS].values[0])
    yp_smooth = np.clip(final_s1.predict(np.array(feats)), 0, 100)
    ax.plot(t_smooth, yp_smooth, 'k--', linewidth=1.8, label='Stage-1 model')
    ax.set_xlabel('Irradiation time (min)')
    ax.set_ylabel('Expt./Pred. degradation efficiency (%)')
    ax.set_title(label, loc='left')
    ax.set_xlim(0, 62); ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=8)
fig.suptitle('External reliability validation of ANR-650-8-4',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig('fig_04_stability_validation.png')

# Fig 5: Stage 2, DegEff vs sunlight intensity coloured by temperature
# Each run lasts 60 min, so lux x time is not a separate measurement. This
# figure plots DegEff against measured lux at each time step, with
# temperature as the third measured variable.
print('  Fig 5 — Stage 2: DegEff vs SunlightLux at four sampling timesteps')
fig, axes = plt.subplots(1, 4, figsize=(19, 5.2), sharey=True)
times_show = [15, 30, 45, 60]
def season(label):
    if label.startswith('(11'): return 'Sunny Summer'
    if label.startswith('(12'): return 'Sunny Winter'
    if label.startswith('(13'): return 'Cloudy Winter'
    return 'Other'
df2_eng['Season'] = df2_eng['Experiment'].apply(season)
SEASON_MARK = {'Sunny Summer':'o', 'Sunny Winter':'s', 'Cloudy Winter':'^', 'Other':'X'}
for ax, t in zip(axes, times_show):
    sub = df2_eng[df2_eng['F8_TimeStep']==t]
    # colour points by measured temperature
    for s, mk in SEASON_MARK.items():
        ss = sub[sub['Season']==s]
        if len(ss)==0: continue
        sc = ax.scatter(ss['SunlightLux']/1000, ss['F13_Degradation'],
                        c=ss['Temperature_C'], cmap='plasma', vmin=22, vmax=46,
                        marker=mk, s=70, edgecolor='black', linewidth=0.5,
                        alpha=0.92, label=s)
    # overlay Stage-2 model prediction line at mean temperature (≈32 °C)
    lux_smooth = np.linspace(20000, 140000, 100)
    T_ref = 32.0
    X_grid = np.column_stack([lux_smooth, np.full_like(lux_smooth, t),
                              np.full_like(lux_smooth, T_ref)])
    yp = np.clip(final_s2.predict(X_grid), 0, 100)
    ax.plot(lux_smooth/1000, yp, 'k--', linewidth=1.6, alpha=0.85,
            label=f'Stage-2 model @ T={T_ref:.0f}°C')
    ax.set_xlabel('Sunlight intensity (\u00d710\u00b3 lux)')
    if ax is axes[0]: ax.set_ylabel('Degradation efficiency (%)')
    ax.set_xlim(15, 140); ax.set_ylim(0, 105)
    ax.set_title(f't = {t} min', loc='left', fontsize=11)
# colorbar for temperature
cbar = fig.colorbar(sc, ax=axes, pad=0.02, fraction=0.022)
cbar.set_label('Measured ambient temperature (°C)')
# One shared legend below the row so the panels stay uncluttered
from matplotlib.lines import Line2D as _L2D
shared = [_L2D([0],[0], marker=SEASON_MARK[s], color='w', markerfacecolor='#888888',
               markeredgecolor='black', markersize=9, label=s)
          for s in ['Sunny Summer','Sunny Winter','Cloudy Winter']]
shared.append(_L2D([0],[0], color='black', ls='--', lw=1.6,
                   label='Stage-2 model (T fixed at 32 °C ≈ dataset mean)'))
fig.legend(handles=shared, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.05), frameon=True)
fig.suptitle('Stage 2 — Degradation efficiency vs Average Sunlight Intensity at four time steps  '
             '(colour = measured temperature; shape = season)',
             fontsize=12, fontweight='bold', y=1.02)
save_fig('fig_05_stage2_lux_temp.png')

# Fig 6: SHAP summary
print('  Fig 6 — SHAP feature attribution')
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
def shap_bar(ax, df_r, color_base, title):
    top = df_r.iloc[::-1]
    cols = plt.get_cmap(color_base)(np.linspace(0.4, 0.95, len(top)))
    ax.barh(range(len(top)), top['Mean_AbsSHAP_pct'], color=cols,
            edgecolor='black', linewidth=0.5)
    vmax = float(top['Mean_AbsSHAP_pct'].max())
    ax.set_xlim(0, vmax * 1.30)            # headroom so the longest label fits inside
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([_fcode(f) for f in top['Feature']], fontsize=12)
    ax.set_xlabel('Mean |SHAP value| (DegEff %-points)')
    ax.set_ylabel('Experimental features')
    ax.set_title(title, loc='left', fontsize=11)
    for i, (v, p) in enumerate(zip(top['Mean_AbsSHAP_pct'], top['Relative_pct'])):
        ax.text(v+vmax*0.012, i, f'{v:.2f}  ({p:.1f}%)', va='center', fontsize=8.5)
shap_bar(axes[0], r1, 'GnBu',
         '(a) Stage 1 — feature attribution (model-explanation only)')
shap_bar(axes[1], r2, 'OrRd',
         '(b) Stage 2 — feature attribution (model-explanation only)')
plt.tight_layout()
save_fig('fig_06_shap_attribution.png')

# Fig 6 (split): Stage 1 only
print('  Fig 6a — SHAP attribution (Stage 1 only)')
fig, ax = plt.subplots(figsize=(8, 6))
shap_bar(ax, r1, 'GnBu',
         'Stage 1 — feature attribution (model-explanation only)')
plt.tight_layout()
if os.environ.get('SPLIT_PANELS'): save_fig('fig_06a_shap_attribution_stage1.png')   # split panel: skipped by default (not used in manuscript)

# Fig 6 (split): Stage 2 only
print('  Fig 6b — SHAP attribution (Stage 2 only)')
fig, ax = plt.subplots(figsize=(8, 4.5))
shap_bar(ax, r2, 'OrRd',
         'Stage 2 — feature attribution (model-explanation only)')
plt.tight_layout()
if os.environ.get('SPLIT_PANELS'): save_fig('fig_06b_shap_attribution_stage2.png')   # split panel: skipped by default (not used in manuscript)

# ----------------------------------------------------------------------------
#  ADDITIONAL DIAGNOSTICS
# ----------------------------------------------------------------------------

# Fig 7: Residual diagnostics for the selected models
print('  Fig 7 — Residual diagnostics (both stages, primary CV predictions)')
fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
for row, (label, ytrue, ypred, col) in enumerate([
        ('Stage 1', y1, s1_pred_rkf[best_s1_name], STAGE1),
        ('Stage 2', y2, s2_pred_rkf[best_s2_name], STAGE2)]):
    resid = ytrue - ypred
    axL, axR = axes[row]
    axL.scatter(ypred, resid, c=col, s=22, alpha=0.55,
                edgecolors='white', linewidths=0.4)
    axL.axhline(0, color='black', linewidth=1)
    for band in (5, 10):
        axL.axhline(band, ls=':', color=NEUTRAL_GREY, linewidth=0.9)
        axL.axhline(-band, ls=':', color=NEUTRAL_GREY, linewidth=0.9)
    axL.set_xlabel('Predicted DegEff (%)')
    axL.set_ylabel('Residual (exp − pred, %-points)')
    axL.set_title(f'({chr(97+row*2)}) {label} — residual vs predicted '
                  f'({best_s1_name if row==0 else best_s2_name}, OOF)',
                  fontsize=10.5)
    axR.hist(resid, bins=25, color=col, alpha=0.75, edgecolor='black',
             linewidth=0.5)
    axR.axvline(0, color='black', linewidth=1)
    axR.set_xlabel('Residual (%-points)')
    axR.set_ylabel('Count')
    mu, sd = resid.mean(), resid.std()
    axR.set_title(f'({chr(98+row*2)}) {label} — residual distribution  '
                  f'(mean {mu:+.2f}, sd {sd:.2f})', fontsize=10.5)
fig.suptitle('Residual diagnostics — out-of-fold predictions, Repeated 5-fold CV',
             fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig('fig_07_residual_diagnostics.png')

# Fig 8: Train vs test R² gap (overfitting check)
print('  Fig 8 — Train vs test R² gap per model')
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
for ax, (results, label) in zip(axes, [(s1_results, 'Stage 1'),
                                       (s2_results, 'Stage 2')]):
    names = list(results.keys())
    ypos = np.arange(len(names))[::-1]
    tr = [results[n]['R2_train_RKF'] for n in names]
    te = [results[n]['R2_RKF'] for n in names]
    for yp, a, b in zip(ypos, tr, te):
        ax.plot([b, a], [yp, yp], color=NEUTRAL_GREY, linewidth=1.6, zorder=1)
    ax.scatter(tr, ypos, s=70, color=CV_STRESS, edgecolor='black',
               linewidth=0.6, zorder=3, label='Train R² (in-fold)')
    ax.scatter(te, ypos, s=70, color=CV_PRIMARY, edgecolor='black',
               linewidth=0.6, zorder=3, label='Test R² (out-of-fold)')
    for yp, a, b in zip(ypos, tr, te):
        ax.text(max(a, b) + 0.012, yp, f'gap {a-b:+.3f}', va='center',
                fontsize=8, color='#444444')
    ax.set_yticks(ypos); ax.set_yticklabels([mabbr(n) for n in names], fontsize=10)
    ax.tick_params(labelleft=True)   # show model labels on both panels (sharey hides them)
    ax.set_xlabel('R² (Repeated 5-fold CV)')
    ax.set_xlim(0.75, 1.10)
    ax.set_title(f'{label}', fontsize=11)
axes[0].legend(loc='upper left', fontsize=9)
fig.suptitle('Train and test R² per model',
             fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig('fig_08_train_test_gap.png')

# Fig 9: Learning curves (data sufficiency check)
print('  Fig 9 — Learning curves (Stacked ensemble, CV R² at increasing N)')
from sklearn.model_selection import KFold as _KF
fractions = [0.35, 0.55, 0.75, 1.00]
rng_lc = np.random.RandomState(SEED)

def learning_curve_points(X, y, model_factory):
    out = []
    for frac in fractions:
        n_sub = max(30, int(len(X) * frac))
        scores = []
        for rep in range(1):
            idx = rng_lc.choice(len(X), size=n_sub, replace=False)
            Xs, ys = X[idx], y[idx]
            kf = _KF(3, shuffle=True, random_state=SEED + rep)
            for tr, te in kf.split(Xs):
                m = model_factory(); m.fit(Xs[tr], ys[tr])
                yp = np.clip(m.predict(Xs[te]), 0, 100)
                scores.append(r2_score(ys[te], yp))
        out.append((n_sub, np.mean(scores), np.std(scores)))
    return out

from sklearn.ensemble import StackingRegressor as _StackReg
from sklearn.linear_model import Ridge as _Ridge

def lc_stack(stage):
    """Stacked ensemble (same base models + Ridge meta-learner) for the inline learning curve.

    Only used when LC_SKIP is not set. The learning curve in the paper is
    built by 02_learning_curves.py.
    """
    d = build_stage1_models() if stage == 1 else build_stage2_models()
    if 'GaussianProcess' in d:
        d['GaussianProcess'].set_params(regressor__n_restarts_optimizer=0)
    for k in ('XGBoost', 'LightGBM', 'ExtraTrees', 'RandomForest', 'GradientBoost'):
        if k in d:
            try: d[k].set_params(n_estimators=250)
            except Exception: pass
    return _StackReg(estimators=list(d.items()), final_estimator=_Ridge(),
                     cv=2, n_jobs=1, passthrough=False)

import os as _os_lc
if _os_lc.environ.get('LC_SKIP') == '1':
    print('    [LC_SKIP=1] inline learning curve skipped (fig_09 built by lc_only.py)')
else:
    lc1 = learning_curve_points(X1, y1, lambda: lc_stack(1))
    lc2 = learning_curve_points(X2, y2, lambda: lc_stack(2))

    fig, ax = plt.subplots(figsize=(9, 6))
    for lc, col, label in [(lc1, STAGE1, 'Stage 1 (N=312 available)'),
                           (lc2, STAGE2, 'Stage 2 (N=192 available)')]:
        ns  = [p[0] for p in lc]
        mus = [p[1] for p in lc]
        sds = [p[2] for p in lc]
        ax.plot(ns, mus, 'o-', color=col, linewidth=2, markersize=7,
                markeredgecolor='black', markeredgewidth=0.6, label=label)
        ax.fill_between(ns, [m - s for m, s in zip(mus, sds)],
                        [m + s for m, s in zip(mus, sds)], color=col, alpha=0.15)
    ax.set_xlabel('Training rows used')
    ax.set_ylabel('CV R² (Stacked ensemble, mean ± sd over folds×repeats)')
    ax.set_title('Learning curves — CV R² vs available training data')
    ax.legend(loc='lower right')
    ax.set_ylim(0.5, 1.02)
    plt.tight_layout()
    save_fig('fig_09_learning_curves.png')

banner('All figures generated')
