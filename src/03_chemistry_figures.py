"""
Model-based and chemistry figures.

Loads the dataset and the XGBoost models saved by 01_linked_ml_pipeline.py
(no retraining) and builds the figures below at 300 DPI.

Built by default (used in the paper):
    B   calcination temperature response (Stage 1 model)
    D   measured ion / scavenger interference
    I   calcination temperature x time design map (Stage 1 model)
    J   differential evolution on the Stage 1 model
    K   one-at-a-time sensitivity at 20 min (Stage 1 model)
    M   time-to-target nomogram (Stage 2 model, monotonic envelope)
    M2  nomogram bar chart at typical sunlight levels
    N   counterfactual stress scenarios (Stage 1 model)
    R   Pearson correlation, Stage 1 inputs
    R2  Pearson correlation, Stage 2 inputs
    X   SHAP dependence plots (Stage 1)
    Y   Stage 2 response surfaces at three temperatures

Other figures (A, C, E, G, H, L, P, Q, S) are exploratory and are built only
with ONLY=ALL. A single figure can be built with e.g. ONLY="fig_K".
Figures F and O (cumulative light dose) are not produced.

Run 01_linked_ml_pipeline.py first.
"""

import os, re, json, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, differential_evolution

# ----------------------------------------------------------------------------
#  CONFIG
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'ANR_TiO2_RhB_dataset.xlsx')
SHEET_VAR    = '1. ANR_02-06-2026 (Vari-Long)'
SHEET_SUNINT = '2(a) ANR-23-05-2026 (SunInt)'

OUT    = os.path.join(ROOT, 'outputs')
FIGDIR = f'{OUT}/figures'
TBLDIR = f'{OUT}/tables'

SEED = 42; DPI = 300
np.random.seed(SEED)

# Shared plotting style
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from project_style import (apply_style, OI, STAGE1, STAGE2, CAT_COLORS,
                           WEATHER_COLOUR as _WC, SEQ, TARGET_LINE,
                           GOOD, BAD, NEUTRAL, FEATURE_LABELS, TIME_LABEL, flabel)
apply_style()

def _pcode(f):
    """Short feature code used on the correlation-matrix axes."""
    code = f.split('_')[0]
    if code == 'F13': return 'DegEff'
    if code == 'SunlightLux': return 'Lux'
    if code == 'Temperature': return 'Temp'
    return code

WEATHER_COLORS = {'sunny_summer': _WC['sunny_summer'],
                  'sunny_winter': _WC['sunny_winter'],
                  'cloudy_winter': _WC['cloudy_winter'],
                  'stability': OI['black']}
PHASE_COLORS  = {1: NEUTRAL, 2: GOOD}
MORPH_MARKERS = {3:'^', 4:'o', 5:'s'}
MORPH_NAMES   = {3:'nanotubes', 4:'nanorods', 5:'nanoparticles'}

def banner(s, ch='═', w=78):
    print('\n' + ch*w); print(f'  {s}'); print(ch*w)

def save_fig(name):
    path = f'{FIGDIR}/{name}'
    plt.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(); print(f'    saved → {path}')

# ----------------------------------------------------------------------------
#  LOAD DATA + MODELS
# ----------------------------------------------------------------------------
banner('Loading data, models, and rebuilding feature helpers')

RENAME_S1 = ['Experiment','CatalystID',
             'F1_CalcTemp','F2_CalcRamp','F3_CalcTime',
             'F4_CrystalPhase','F5_CrysSize','F6_Crystallinity','F7_Morphology',
             'F8_TimeStep','F9_RhBConc','F10_pH','F11_Ions','F12_Scavengers',
             'F13_Degradation']

df1_raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_VAR, header=0)
df1_raw.columns = RENAME_S1
df1_raw = df1_raw[df1_raw['CatalystID'].notna()].reset_index(drop=True)
for c in RENAME_S1[2:]:
    df1_raw[c] = pd.to_numeric(df1_raw[c], errors='coerce')
df1_raw['Experiment'] = df1_raw['Experiment'].astype(str).str.strip()
df1_raw['F13_Degradation'] = df1_raw['F13_Degradation'].clip(lower=0)
# Clean Stage-1 training subset (no t=0, no stability rows)
STAB_PFX = ['(8a)','(8b)','(8c)','(9a)','(9b)','(9c)','(10a)','(10b)','(10c)','(10d)']
df1 = df1_raw[df1_raw['F8_TimeStep'] > 0].copy().reset_index(drop=True)
df1_train = df1[~df1['Experiment'].apply(
    lambda x: any(x.startswith(p) for p in STAB_PFX))].reset_index(drop=True)
df1_stab  = df1[df1['Experiment'].apply(
    lambda x: any(x.startswith(p) for p in STAB_PFX))].reset_index(drop=True)

# Sheet 2a: measured sunlight
df2_raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_SUNINT, header=0).iloc[:, :5]
df2_raw.columns = ['Experiment','CatalystID','SunlightLux','F8_TimeStep','F13_Degradation']
df2_raw = df2_raw[df2_raw['CatalystID'].notna()].reset_index(drop=True)
df2_raw['Experiment'] = df2_raw['Experiment'].astype(str).str.strip()

def parse_lux(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    return float(str(x).replace(',', '').strip())
df2_raw['SunlightLux']    = df2_raw['SunlightLux'].apply(parse_lux)
df2_raw['F8_TimeStep']    = pd.to_numeric(df2_raw['F8_TimeStep'], errors='coerce')
df2_raw['F13_Degradation']= pd.to_numeric(df2_raw['F13_Degradation'], errors='coerce')
# Fix mislabelled (13h) rows
df2_raw.loc[df2_raw['Experiment'].str.startswith('(13h)', na=False),
            'Experiment'] = '(13h) Cloudy Winter 16-17'
df2_raw['Experiment'] = df2_raw['Experiment'].str.replace(r'\s+', ' ', regex=True)
df2_raw['F13_Degradation'] = df2_raw['F13_Degradation'].clip(lower=0)
df2 = df2_raw[df2_raw['F8_TimeStep'] > 0].copy().reset_index(drop=True)

# Attach Temperature_C to df2 from sheet '(3) Temperature'
# The sheet holds a long-format (weather, hour) table with the header on row 4.
def _parse_hour(lab):
    m = re.search(r'(\d{1,2})-\d{1,2}\b', lab)
    return int(m.group(1)) if m else None

def _parse_weather(lab):
    m = re.search(r'\((\d+)', lab)
    if not m: return None
    n = int(m.group(1))
    return {11:'sunny_summer', 12:'sunny_winter', 13:'cloudy_winter'}.get(n, None)

try:
    _tdf = pd.read_excel(DATA_PATH, sheet_name='(3) Temperature', header=3).iloc[:24]
    _tdf = _tdf[['Weather', 'Hour', 'Temperature_C']].dropna()
    _tdf['Hour']          = _tdf['Hour'].astype(int)
    _tdf['Temperature_C'] = _tdf['Temperature_C'].astype(float)
    TEMP_LOOKUP = {}
    for _, r in _tdf.iterrows():
        TEMP_LOOKUP.setdefault(r['Weather'], {})[r['Hour']] = r['Temperature_C']
    df2['Hour']    = df2['Experiment'].apply(_parse_hour)
    df2['Weather'] = df2['Experiment'].apply(_parse_weather)
    df2['Temperature_C'] = df2.apply(
        lambda r: TEMP_LOOKUP.get(r['Weather'], {}).get(r['Hour'], np.nan), axis=1)
    print(f'  Temperature attached to df2 from sheet "(3) Temperature": '
          f'range [{df2["Temperature_C"].min():.0f}, '
          f'{df2["Temperature_C"].max():.0f}] °C')
except Exception as e:
    print(f'  WARNING: could not load temperatures from Excel: {e}')

# Trained models saved by 01_linked_ml_pipeline.py
with open(f'{OUT}/models/final_models_v3.pkl', 'rb') as f:
    MODELS = pickle.load(f)
stage1_mdl = MODELS['stage1_xgboost']
stage2_mdl = MODELS['stage2_xgboost']
S1_FEATS   = MODELS['stage1_features']
S2_FEATS   = MODELS['stage2_features']
CAT_CONST  = MODELS['catalyst_constants']

print(f'  Stage-1 features ({len(S1_FEATS)}): {S1_FEATS}')
print(f'  Stage-2 features ({len(S2_FEATS)}): {S2_FEATS}')
print(f'  Training rows: Stage 1 = {len(df1_train)}, Stage 2 = {len(df2)}')
print(f'  Stability validation rows (unseen): {len(df1_stab)}')

# Catalyst characterisation table
char_table = (df1_raw[['CatalystID','F1_CalcTemp','F2_CalcRamp','F3_CalcTime',
                        'F4_CrystalPhase','F5_CrysSize','F6_Crystallinity','F7_Morphology']]
              .drop_duplicates(subset=['CatalystID'])
              .sort_values(['F1_CalcTemp','F2_CalcRamp','F3_CalcTime'])
              .reset_index(drop=True))

# Prediction helpers
# Stage 1 inputs: F1, F2, F3, F8, F9, F10, F11, F12, in this order.
def s1_predict(F1, F2, F3, t, F9=11, F10=11, F11=0, F12=0):
    x = np.array([[F1, F2, F3, t, F9, F10, F11, F12]], dtype=float)
    return float(np.clip(stage1_mdl.predict(x)[0], 0, 100))

def s1_predict_vec(F1, F2, F3, times, F9=11, F10=11, F11=0, F12=0):
    return np.array([s1_predict(F1, F2, F3, t, F9, F10, F11, F12) for t in times])

# Stage 2: SunlightLux, F8_TimeStep, Temperature_C
# Temperature defaults to 32 °C (approx. mid-range of measured temperatures).
def s2_predict(lux, t, temp_c=32.0):
    x = np.array([[lux, t, temp_c]], dtype=float)
    return float(np.clip(stage2_mdl.predict(x)[0], 0, 100))

# Sanity check
print(f"\n  Sanity-check predictions:")
print(f"    ANR-650-8-4 @ t=60, baseline: {s1_predict(650,8,4,60):.2f}%")
print(f"    Sunny summer noon (134k lux, 42°C) @ t=20: {s2_predict(134000, 20, 42):.2f}%")
print(f"    Sunny winter noon (120k lux, 32°C) @ t=20: {s2_predict(120000, 20, 32):.2f}%")
print(f"    Cloudy winter (22k lux, 23°C) @ t=45:      {s2_predict(22000, 45, 23):.2f}%")


# ----------------------------------------------------------------------------
#  Per-catalyst empirical performance (used by figs A, B, J)
# ----------------------------------------------------------------------------
# Restrict to baseline operational conditions (Ions=0, Scavengers=0) at t=60
# for an apples-to-apples comparison across catalysts.
baseline_60 = df1_train[(df1_train['F11_Ions']==0) &
                        (df1_train['F12_Scavengers']==0) &
                        (df1_train['F8_TimeStep']==60)]
rank = baseline_60.groupby('CatalystID').agg(
    F1_CalcTemp=('F1_CalcTemp','first'),
    F2_CalcRamp=('F2_CalcRamp','first'),
    F3_CalcTime=('F3_CalcTime','first'),
    F4_CrystalPhase=('F4_CrystalPhase','first'),
    F5_CrysSize=('F5_CrysSize','first'),
    F6_Crystallinity=('F6_Crystallinity','first'),
    F7_Morphology=('F7_Morphology','first'),
    MeanDegEff_60min=('F13_Degradation','mean'),
    StdDegEff_60min=('F13_Degradation','std'),
    N_observations=('F13_Degradation','size'),
).round(3).reset_index()
rank = rank.sort_values('MeanDegEff_60min', ascending=False).reset_index(drop=True)
rank.to_csv(f'{TBLDIR}/08_catalyst_empirical_ranking.csv', index=False)

# ----------------------------------------------------------------------------
#  FIGURE SELECTION
#  By default only the figures used in the paper are built.
#  Override with env, e.g.  ONLY="fig_K"  or  ONLY="fig_R fig_R2"  to build a
#  single figure (or a few) fast. Use ONLY="ALL" to force every figure.
#  Shared computations (per-catalyst rank, SHAP, stars) live ABOVE this gate so
#  any single-figure build still has what it needs.
# ----------------------------------------------------------------------------
_USED_FIGS = {'B','D','I','J','K','M','M2','N','R','R2','X','Y'}
_ALL_FIGS  = {'A','B','C','D','E','G','H','I','J','K','L','M','M2','N','P','Q','R','R2','S','X','Y'}
_only = os.environ.get('ONLY','').strip()
if not _only:
    BUILD = set(_USED_FIGS)
elif _only.upper() == 'ALL':
    BUILD = set(_ALL_FIGS)
else:
    _req = [t.strip() for t in _only.replace(',', ' ').split() if t.strip()]
    BUILD = set(t.replace('fig_', '').replace('.png', '') for t in _req)
def want(tag):
    return tag in BUILD
print(f'  [gate] building figures: {sorted(BUILD)}')

# Imports shared by several figures
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401 (registers the 3D projection)
from scipy.stats import pearsonr as pcc


# Stage-1 SHAP values (shared by fig_P and fig_X)
import shap as shap_lib

X1 = df1_train[S1_FEATS].values.astype(float)
explainer1 = shap_lib.TreeExplainer(stage1_mdl)
shap_vals1 = explainer1.shap_values(X1)
print(f'  Computed SHAP for Stage 1: {shap_vals1.shape}')

# Feature display names (cleaner labels for the figure)
S1_LABELS = {
    'F1_CalcTemp':   'Calcination temp (°C)',
    'F2_CalcRamp':   'Calcination ramp (°C/min)',
    'F3_CalcTime':   'Calcination time (h)',
    'F8_TimeStep':   TIME_LABEL,
    'F9_RhBConc':    'RhB concentration (×10⁻⁵ M)',
    'F10_pH':        'Solution pH',
    'F11_Ions':      'Co-existing ions (code)',
    'F12_Scavengers':'Radical scavengers (code)',
}
labels = [S1_LABELS[f] for f in S1_FEATS]

# Significance-star helper (shared by fig_R and fig_R2)
def stars(p):
    if p <= 0.001: return '***'
    if p <= 0.01:  return '**'
    if p <= 0.05:  return '*'
    return ''


# ----------------------------------------------------------------------------
#  FIG A: Catalyst property radar + ranked bar
# ----------------------------------------------------------------------------
if want('A'):
    banner('FIG A — Catalyst property fingerprints + ranking')

    # Radar axes are normalised F4-F7 values. F4-F7 are characterisation data,
    # not model inputs; they appear here only as physical descriptors.
    radar_axes = ['Crystallinity', 'Surface area\n(1/CrysSize)', 'Anatase\nphase',
                  'Nanorod\nmorphology', 'Calc. temp\n(normalised)']
    surf_all = 1.0 / rank['F5_CrysSize']
    smin, smax = surf_all.min(), surf_all.max()
    def radar_vals(row):
        cryst = row['F6_Crystallinity'] / 100.0
        surf  = ((1.0 / row['F5_CrysSize']) - smin) / (smax - smin)
        anat  = float(row['F4_CrystalPhase'] == 2)
        rod   = float(row['F7_Morphology'] == 4)
        temp  = (row['F1_CalcTemp'] - 300) / (750 - 300)
        return np.array([cryst, surf, anat, rod, temp])

    best  = rank.iloc[0]
    anr   = rank[rank['CatalystID'] == 'ANR-650-8-4'].iloc[0]
    worst = rank.iloc[-1]

    fig = plt.figure(figsize=(15, 6.2))
    axr = fig.add_subplot(1, 2, 1, polar=True)
    angles = np.linspace(0, 2*np.pi, len(radar_axes), endpoint=False).tolist() + [0]
    for row, label, color in [
            (best,  f"Top: {best['CatalystID']}",  '#2E7D32'),
            (anr,   'ANR-650-8-4 (deployed)',      '#F57C00'),
            (worst, f"Worst: {worst['CatalystID']}",'#C62828')]:
        vals = radar_vals(row).tolist() + [radar_vals(row)[0]]
        axr.plot(angles, vals, 'o-', linewidth=2, label=label, color=color)
        axr.fill(angles, vals, alpha=0.12, color=color)
    axr.set_xticks(angles[:-1]); axr.set_xticklabels(radar_axes, fontsize=9)
    axr.set_yticklabels([])
    axr.set_ylim(0, 1.05)
    axr.set_title('(a) Catalyst property fingerprints', loc='left', fontsize=11, pad=18)
    axr.legend(loc='upper right', bbox_to_anchor=(1.32, 1.12), fontsize=8.5)

    axb = fig.add_subplot(1, 2, 2)
    ypos = np.arange(len(rank))[::-1]
    colors_b = ['#F57C00' if c == 'ANR-650-8-4' else
                (PHASE_COLORS[2] if p == 2 else PHASE_COLORS[1])
                for c, p in zip(rank['CatalystID'], rank['F4_CrystalPhase'])]
    axb.barh(ypos, rank['MeanDegEff_60min'], color=colors_b,
             edgecolor='black', linewidth=0.6)
    axb.set_yticks(ypos); axb.set_yticklabels(rank['CatalystID'], fontsize=8.5)
    axb.set_xlabel('Experimental mean DegEff at t=60 min (%)')
    axb.set_xlim(0, 108)
    axb.set_title('(b) Catalyst ranking (baseline conditions)', loc='left', fontsize=11)
    for yp, v in zip(ypos, rank['MeanDegEff_60min']):
        axb.text(v + 1, yp, f'{v:.1f}', va='center', fontsize=8)
    legend_b = [Patch(facecolor=PHASE_COLORS[2], edgecolor='black', label='Anatase (other)'),
                Patch(facecolor=PHASE_COLORS[1], edgecolor='black', label='TiO₂-B phase'),
                Patch(facecolor='#F57C00', edgecolor='black', label='ANR-650-8-4')]
    axb.legend(handles=legend_b, loc='lower right', fontsize=8.5)

    plt.tight_layout()
    save_fig('fig_A_catalyst_radar.png')


    # ----------------------------------------------------------------------------
    #  FIG D: Ion / scavenger interference (experimental data)
    # ----------------------------------------------------------------------------
if want('D'):
    banner('FIG D — Ion & scavenger interference quantification')

    base_60 = df1_train[(df1_train['CatalystID']=='ANR-650-8-4') &
                        (df1_train['F11_Ions']==0) &
                        (df1_train['F12_Scavengers']==0) &
                        (df1_train['F8_TimeStep']==60)]['F13_Degradation'].mean()

    label_map = {
        '(6b) Co-existing ions-Anion Mix.':'Anion mixture',
        '(6c) Co-existing ions-Cation Mix.':'Cation mixture',
        '(6d) Co-existing ions-Combined Mix.':'Combined ions',
        '(7b) Radical Scavengers-electron':'e⁻ scavenger',
        '(7c) Radical Scavengers-hole':'h⁺ scavenger',
        '(7d) Radical Scavengers-super oxide radical':'O₂•⁻ scavenger',
        '(7e) Radical Scavengers-hydroxyl radical':'•OH scavenger',
    }
    intf_rows = []
    for lab, nice in label_map.items():
        sub = df1_train[(df1_train['Experiment']==lab) & (df1_train['F8_TimeStep']==60)]
        if not sub.empty:
            deg = sub['F13_Degradation'].mean()
            intf_rows.append({'Species':nice, 'DegEff_60':deg,
                              'Reduction':base_60 - deg,
                              'Type':'Scavenger' if 'scaveng' in nice.lower() else 'Ion'})
    intf = pd.DataFrame(intf_rows).sort_values('Reduction')
    intf.to_csv(f'{TBLDIR}/09_interference_quantification.csv', index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ypos = np.arange(len(intf))
    colors_t = ['#C62828' if t == 'Scavenger' else '#1565C0' for t in intf['Type']]
    ax.barh(ypos, intf['Reduction'], color=colors_t, edgecolor='black', linewidth=0.6)
    ax.set_yticks(ypos); ax.set_yticklabels(intf['Species'], fontsize=10)
    ax.set_xlabel('Suppression of RhB degradation (% points)')
    ax.set_title('Interference quantification — ANR-650-8-4 deployed catalyst',
                 loc='left', fontsize=11)
    xmax = max(intf['Reduction'].max(), 1)
    for yp, v, d in zip(ypos, intf['Reduction'], intf['DegEff_60']):
        txt = (f'−{v:.0f} pp  (→{d:.0f}%)' if v >= 0 else f'+{-v:.0f} pp  (→{d:.0f}%)')
        if abs(v) < 0.08 * xmax:
            ax.text(0.10 * xmax, yp, txt, va='center', ha='left', fontsize=8, color='#444')
        else:
            ax.text(v + 0.8, yp, txt, va='center', ha='left', fontsize=8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlim(-3, xmax * 1.18)
    ax.text(0.98, 0.04,
            'Stage-1 model SHAP (v3):\nF11_Ions = 4.97 pp (12.5%)  |  F12_Scavengers = 3.68 pp (9.2%)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1', edgecolor='#F57C00'))
    legend_d = [Patch(facecolor='#1565C0', edgecolor='black', label='Co-existing ions'),
                Patch(facecolor='#C62828', edgecolor='black', label='Radical scavengers')]
    ax.legend(handles=legend_d, loc='center right', fontsize=9)
    plt.tight_layout()
    save_fig('fig_D_interference.png')


    # ----------------------------------------------------------------------------
    #  FIG B: Calcination temperature response (Stage-1 model)
    # ----------------------------------------------------------------------------
if want('B'):
    banner('FIG B — Calcination temperature response')

    # Stage-1 prediction curve over the measured range (300-750 °C). F2=2 and F3=3
    # are held at the temperature-series recipe so the curve and the experimental
    # points share the same ramp and dwell. Baseline operating conditions, t=60.
    # The curve interpolates between observed (F1, F2, F3) recipes, so treat it
    # as indicative rather than a confident extrapolation.
    temps = np.linspace(300, 750, 60)
    pred_T = [s1_predict(T, 2, 3, 60) for T in temps]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(temps, pred_T, '-', color=STAGE1, linewidth=2.5,
            label='Stage-1 model (F2=2, F3=3, t=60)', zorder=3)

    # Experimental temperature series (F2=2, F3=3) spanning 300-750 °C, coloured by
    # crystal phase. TiO2-B below 600 °C; anatase from 600 °C upward (incl. 700/750).
    ts = rank[(rank['F2_CalcRamp'] == 2) & (rank['F3_CalcTime'] == 3)].sort_values('F1_CalcTemp')
    anat = ts[ts['F4_CrystalPhase'] == 2]
    btio = ts[ts['F4_CrystalPhase'] == 1]
    ax.scatter(anat['F1_CalcTemp'], anat['MeanDegEff_60min'], s=130, color=OI['vermillion'],
               edgecolor='black', linewidth=1, zorder=5, marker='D',
               label='Experimental — anatase')
    ax.scatter(btio['F1_CalcTemp'], btio['MeanDegEff_60min'], s=110, color=OI['skyblue'],
               edgecolor='black', linewidth=1, zorder=5, marker='o',
               label='Experimental — TiO\u2082-B')
    # Deployed catalyst (its own recipe F2=8, F3=4)
    dep = rank[rank['CatalystID'] == 'ANR-650-8-4']
    if len(dep):
        ax.scatter(dep['F1_CalcTemp'], dep['MeanDegEff_60min'], s=150, color=OI['orange'],
                   edgecolor='black', linewidth=1.2, zorder=6, marker='*',
                   label='Deployed ANR-650-8-4')

    ax.axvline(650, ls='--', color=OI['orange'], alpha=0.7, linewidth=1.4)
    ax.set_xlabel('Calcination temperature, F1 (°C)')
    ax.set_ylabel('DegEff at t=60 min (%)', color=STAGE1)
    ax.tick_params(axis='y', labelcolor=STAGE1)
    ax.set_ylim(60, 108); ax.set_xlim(280, 770)
    ax.set_title('Calcination temperature response (F2=2, F3=3 series)\n'
                 '(F4–F7 are NOT model inputs — crystal size plotted as observed)',
                 loc='left', fontsize=11)

    # Secondary axis: observed crystal size for the same series
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(ts['F1_CalcTemp'], ts['F5_CrysSize'], 's--', color=OI['purple'],
             linewidth=1.6, markersize=7, alpha=0.85,
             label='F5 crystal size (nm) — observed')
    ax2.set_ylabel('Average crystal size F5 (nm)', color=OI['purple'])
    ax2.tick_params(axis='y', labelcolor=OI['purple'])
    ax2.spines['right'].set_visible(True)

    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc='lower right', ncol=1, fontsize=8.5,
              frameon=True, framealpha=0.95)

    plt.tight_layout()
    save_fig('fig_B_temperature_response.png')


    # ----------------------------------------------------------------------------
    #  FIG C: Pseudo-first-order kinetic fits per catalyst
    # ----------------------------------------------------------------------------
if want('C'):
    banner('FIG C — Pseudo-first-order kinetics')

    def first_order(t, k): return 100.0 * (1.0 - np.exp(-k * t))

    def cat_curve(cat_id):
        sub = df1_raw[(df1_raw['CatalystID']==cat_id) &
                      (df1_raw['F11_Ions']==0) &
                      (df1_raw['F12_Scavengers']==0)]
        g = sub.groupby('F8_TimeStep')['F13_Degradation'].mean().sort_index()
        return g.index.values.astype(float), g.values.astype(float)

    # Fit k for each catalyst
    k_rows = []
    for _, r in rank.iterrows():
        t, d = cat_curve(r['CatalystID'])
        if len(t) < 3:
            continue
        try:
            popt, _ = curve_fit(first_order, t, d, p0=[0.05],
                                bounds=(1e-4, 1.0), maxfev=10000)
            k = popt[0]
            resid = d - first_order(t, k)
            r2 = 1 - np.sum(resid**2) / max(np.sum((d - d.mean())**2), 1e-9)
        except Exception:
            k, r2 = np.nan, np.nan
        k_rows.append({'CatalystID':r['CatalystID'], 'k_min-1':k, 'fit_R2':r2,
                       'CrystalPhase':r['F4_CrystalPhase'], 'Morphology':r['F7_Morphology'],
                       'F1_CalcTemp':r['F1_CalcTemp']})
    k_df = pd.DataFrame(k_rows)
    k_df.to_csv(f'{TBLDIR}/10_kinetic_rate_constants.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    selected = ['ANR-650-8-4', 'ANR-650-5-3', 'BNT-500-2-3', 'BTNT-300-2-3']
    sel_colors = ['#F57C00', '#2E7D32', '#1976D2', '#C62828']
    ax = axes[0]
    t_smooth = np.linspace(0, 60, 200)
    for cid, color in zip(selected, sel_colors):
        t, d = cat_curve(cid)
        r = rank[rank['CatalystID'] == cid].iloc[0]
        ax.scatter(t, d, s=55, color=color, edgecolor='black', linewidth=0.7, zorder=5)
        krow = k_df[k_df['CatalystID'] == cid]
        if not krow.empty and np.isfinite(krow['k_min-1'].iloc[0]):
            k = krow['k_min-1'].iloc[0]
            ax.plot(t_smooth, first_order(t_smooth, k), '-', color=color, linewidth=2,
                    label=f'{cid}  (k={k:.3f} min⁻¹)')
        # Stage-1 model prediction overlay (dotted)
        mp = s1_predict_vec(r['F1_CalcTemp'], r['F2_CalcRamp'], r['F3_CalcTime'],
                            t_smooth[t_smooth > 0])
        ax.plot(t_smooth[t_smooth > 0], mp, ':', color=color, linewidth=1.4, alpha=0.7)
    ax.set_xlabel(TIME_LABEL); ax.set_ylabel('Degradation efficiency (%)')
    ax.set_xlim(0, 62); ax.set_ylim(0, 105)
    ax.set_title('(a) Kinetic curves: experiment (points), first-order fit (solid),\n'
                 'v3 Stage-1 model (dotted)', loc='left', fontsize=10.5)
    ax.legend(loc='lower right', fontsize=8.5)

    ax = axes[1]
    k_plot = k_df.dropna(subset=['k_min-1']).sort_values('k_min-1', ascending=True)
    ypos = np.arange(len(k_plot))
    bar_colors = [PHASE_COLORS[p] for p in k_plot['CrystalPhase']]
    hatches = {3:'///', 4:'', 5:'xxx'}
    bars = ax.barh(ypos, k_plot['k_min-1'], color=bar_colors,
                   edgecolor='black', linewidth=0.6)
    for b, m in zip(bars, k_plot['Morphology']):
        b.set_hatch(hatches.get(int(m), ''))
    ax.set_yticks(ypos); ax.set_yticklabels(k_plot['CatalystID'], fontsize=8.5)
    ax.set_xlabel('Apparent first-order rate constant k (min⁻¹)')
    ax.set_title('(b) Apparent rate constants across catalysts', loc='left', fontsize=10.5)
    for yp, v in zip(ypos, k_plot['k_min-1']):
        ax.text(v + 0.002, yp, f'{v:.3f}', va='center', fontsize=7.5)
    legend_k = [Patch(facecolor=PHASE_COLORS[2], edgecolor='black', label='Anatase'),
                Patch(facecolor=PHASE_COLORS[1], edgecolor='black', label='TiO₂-B'),
                Patch(facecolor='white', edgecolor='black', hatch='///', label='nanotubes'),
                Patch(facecolor='white', edgecolor='black', label='nanorods'),
                Patch(facecolor='white', edgecolor='black', hatch='xxx', label='nanoparticles')]
    ax.legend(handles=legend_k, loc='lower right', fontsize=8)

    plt.tight_layout()
    save_fig('fig_C_kinetics.png')


    # ----------------------------------------------------------------------------
    #  FIG G: Stability suite with Stage-1 model overlay
    # ----------------------------------------------------------------------------
if want('G'):
    banner('FIG G — Stability suite of optimised catalyst')

    stab_groups_def = {
        'Repeatability'  : ['(8a)','(8b)','(8c)'],
        'Reproducibility': ['(9a)','(9b)','(9c)'],
        'Reusability'    : ['(10a)','(10b)','(10c)','(10d)'],
    }
    df_stab_full = df1_raw[df1_raw['Experiment'].apply(
        lambda x: any(x.startswith(p) for prefixes in stab_groups_def.values() for p in prefixes))]

    def curve_for(label):
        sub = df_stab_full[df_stab_full['Experiment'] == label]
        g = sub.groupby('F8_TimeStep')['F13_Degradation'].mean().sort_index()
        return g.index.values.astype(float), g.values.astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    def plot_suite(ax, labels_pref, title, panel, cmap_name):
        matching_labs = sorted(set(e for e in df_stab_full['Experiment']
                                   if any(e.startswith(p) for p in labels_pref)))
        cols = plt.get_cmap(cmap_name)(np.linspace(0.35, 0.9, len(matching_labs)))
        all_t60 = []; all_t20 = []
        for lab, c in zip(matching_labs, cols):
            t, d = curve_for(lab)
            if len(t) == 0: continue
            nice = lab.split(')')[-1].strip()
            ax.plot(t, d, 'o-', color=c, linewidth=1.8, markersize=5, label=nice)
            v60 = d[t == 60]; v20 = d[t == 20]
            if len(v60): all_t60.append(v60[0])
            if len(v20): all_t20.append(v20[0])
        # 95% band
        times_common = np.array([3, 6, 10, 15, 20, 30, 45, 60])
        stacked = []
        for lab in matching_labs:
            t, d = curve_for(lab)
            row = {ti: di for ti, di in zip(t, d)}
            stacked.append([row.get(tc, np.nan) for tc in times_common])
        stacked = np.array(stacked, dtype=float)
        m_c = np.nanmean(stacked, axis=0); s_c = np.nanstd(stacked, axis=0)
        ax.fill_between(times_common, m_c - 1.96*s_c, m_c + 1.96*s_c,
                        color='gray', alpha=0.15, zorder=1, label='95% band')
        # Stage-1 model prediction at baseline (ANR-650-8-4 conditions)
        mp = s1_predict_vec(650, 8, 4, times_common)
        ax.plot(times_common, mp, 'k--', linewidth=1.8, label='v3 Stage-1 model')
        ax.set_xlabel(TIME_LABEL)
        if panel == 'a': ax.set_ylabel('Degradation efficiency (%)')
        ax.set_xlim(0, 62); ax.set_ylim(0, 105)
        txt_lines = []
        if len(all_t20) > 1:
            cv = np.std(all_t20) / np.mean(all_t20) * 100
            txt_lines.append(f'RSD at t=20 min: {cv:.1f}%')
        if len(all_t60) > 1:
            cv = np.std(all_t60) / np.mean(all_t60) * 100
            txt_lines.append(f'RSD at t=60 min: {cv:.1f}%')
        if txt_lines:
            ax.text(0.04, 0.06, '\n'.join(txt_lines), transform=ax.transAxes,
                    fontsize=8.5, bbox=dict(boxstyle='round,pad=0.4',
                                           facecolor='#FFF8E1', edgecolor='#F57C00'))
        ax.set_title(f'({panel}) {title}', loc='left', fontsize=11)
        ax.legend(loc='lower right', fontsize=8)
        return all_t60, all_t20

    plot_suite(axes[0], stab_groups_def['Repeatability'],
               'Repeatability (3 reactors)', 'a', 'Oranges')
    plot_suite(axes[1], stab_groups_def['Reproducibility'],
               'Reproducibility (3 days)', 'b', 'Blues')
    reuse_t60, reuse_t20 = plot_suite(axes[2], stab_groups_def['Reusability'],
                                      'Reusability (4 cycles)', 'c', 'Greens')

    if len(reuse_t20) >= 2:
        ret = reuse_t20[-1] / reuse_t20[0] * 100
        axes[2].text(0.04, 0.22,
                     f'Activity at t=20 min\nretained after 4 cycles: {ret:.0f}%',
                     transform=axes[2].transAxes, fontsize=8.5,
                     bbox=dict(boxstyle='round,pad=0.4',
                               facecolor='#E8F5E9', edgecolor='#2E7D32'))

    fig.suptitle('Stability suite of ANR-650-8-4 — experiment (points) vs '
                 'v3 Stage-1 model (dashed, never trained on these rows)',
                 fontsize=12.5, fontweight='bold', y=1.03)
    plt.tight_layout()
    save_fig('fig_G_stability_suite.png')


    # ----------------------------------------------------------------------------
    #  FIG H: Stage 1 vs Stage 2 SHAP attribution by feature category
    # ----------------------------------------------------------------------------
if want('H'):
    banner('FIG H — Stage 1 vs Stage 2 SHAP by category (v3)')

    shap1 = pd.read_csv(f'{TBLDIR}/06_shap_stage1.csv')
    shap2 = pd.read_csv(f'{TBLDIR}/06_shap_stage2.csv')

    CAT = {
        # Stage 1
        'F1_CalcTemp':'material','F2_CalcRamp':'material','F3_CalcTime':'material',
        'F8_TimeStep':'kinetic',
        'F9_RhBConc':'operational','F10_pH':'operational',
        'F11_Ions':'operational','F12_Scavengers':'operational',
        # Stage 2
        'SunlightLux':'environmental','Temperature_C':'environmental',
    }
    shap1['Category'] = shap1['Feature'].map(CAT)
    shap2['Category'] = shap2['Feature'].map(CAT)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    def shap_panel(ax, df_r, title, panel):
        df_r = df_r.sort_values('Relative_pct', ascending=True)
        colors = [CAT_COLORS[c] for c in df_r['Category']]
        ypos = np.arange(len(df_r))
        ax.barh(ypos, df_r['Relative_pct'], color=colors,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(ypos); ax.set_yticklabels([flabel(f) for f in df_r['Feature']], fontsize=10.5)
        ax.set_xlabel('Relative SHAP contribution (%)')
        ax.set_title(f'({panel}) {title}', loc='left', fontsize=11)
        for yp, v in zip(ypos, df_r['Relative_pct']):
            ax.text(v + 0.4, yp, f'{v:.1f}%', va='center', fontsize=8)

    shap_panel(axes[0], shap1, 'Stage 1 — catalyst + operational design', 'a')
    shap_panel(axes[1], shap2, 'Stage 2 — deployment (fixed catalyst)',  'b')

    # Category totals
    def cat_totals(sdf):
        return sdf.groupby('Category')['Relative_pct'].sum().to_dict()
    t1, t2 = cat_totals(shap1), cat_totals(shap2)
    axes[0].text(0.97, 0.04,
        'By category:\n' + '\n'.join(
            f'{k}: {t1.get(k,0):.0f}%' for k in ['kinetic','material','operational']),
        transform=axes[0].transAxes, ha='right', va='bottom', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='gray'))
    axes[1].text(0.97, 0.04,
        'By category:\n' + '\n'.join(
            f'{k}: {t2.get(k,0):.0f}%' for k in ['kinetic','environmental']),
        transform=axes[1].transAxes, ha='right', va='bottom', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='gray'))

    legend_cat = [Patch(facecolor=CAT_COLORS[c], edgecolor='black', label=c.capitalize())
                  for c in ['kinetic','material','operational','environmental']]
    fig.legend(handles=legend_cat, loc='upper center', ncol=4, fontsize=9.5,
               bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle('Feature attribution shift (v3.1): material design (Stage 1) → '
                 'deployment (Stage 2)   [SHAP = model attribution only]',
                 fontsize=12, fontweight='bold', y=1.07)
    plt.tight_layout()
    save_fig('fig_H_shap_by_category.png')

    # Fig H (split): Stage 1 standalone
    fig, ax = plt.subplots(figsize=(8, 6))
    shap_panel(ax, shap1, 'Stage 1 — catalyst + operational design', 'a')
    ax.text(0.97, 0.04, 'By category:\n' + '\n'.join(
            f'{k}: {t1.get(k,0):.0f}%' for k in ['kinetic','material','operational']),
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='gray'))
    fig.legend(handles=legend_cat, loc='upper center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle('Stage 1 feature attribution by category   [SHAP = model attribution only]',
                 fontsize=11.5, fontweight='bold', y=1.05)
    plt.tight_layout()
    save_fig('fig_H_shap_by_category_stage1.png')

    # Fig H (split): Stage 2 standalone
    fig, ax = plt.subplots(figsize=(8, 4.5))
    shap_panel(ax, shap2, 'Stage 2 — deployment (fixed catalyst)', 'b')
    ax.text(0.97, 0.04, 'By category:\n' + '\n'.join(
            f'{k}: {t2.get(k,0):.0f}%' for k in ['kinetic','environmental']),
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='gray'))
    fig.legend(handles=legend_cat, loc='upper center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle('Stage 2 feature attribution by category   [SHAP = model attribution only]',
                 fontsize=11.5, fontweight='bold', y=1.05)
    plt.tight_layout()
    save_fig('fig_H_shap_by_category_stage2.png')


    # ----------------------------------------------------------------------------
    #  FIG E: Operational guideline heatmap (lux x time)
    # ----------------------------------------------------------------------------
if want('E'):
    banner('FIG E — Operational guideline: lux × time → DegEff')

    # Stage 2 takes measured lux as a direct input. Heatmap rows are reaction
    # times and columns are lux bins spanning the observed range; each bin is
    # also annotated with the matching hour of day for each weather type.
    #
    # The raw Stage-2 model is non-monotonic in lux because the data are: a few
    # runs at 76k and 98k lux underperformed, probably because the averaged lux
    # reading misses cloud-cover changes during the run. Panel (a) therefore
    # shows the monotonic envelope
    #       envelope(lux, t) = max_{lux' <= lux} predict(lux', t)
    # which reads as "at this lux or higher, expect at least this degradation".
    # Panel (d) shows the raw scatter so the non-monotonicity stays visible.
    #
    lux_bins = [10000, 25000, 50000, 75000, 100000, 130000]
    times = np.array([3, 6, 10, 15, 20, 30, 45, 60])

    def parse_hour(lab):
        m = re.search(r'(\d{1,2})-\d{1,2}\b', lab)
        return int(m.group(1)) if m else None
    def weather_of(lab):
        m = re.search(r'\((\d+)', lab)
        if not m: return 'stability'
        n = int(m.group(1))
        return {11:'sunny_summer', 12:'sunny_winter', 13:'cloudy_winter'}.get(n, 'stability')
    df2['Hour'] = df2['Experiment'].apply(parse_hour)
    df2['Weather'] = df2['Experiment'].apply(weather_of)
    hour_lux = (df2[df2['Weather']!='stability']
                .groupby(['Weather','Hour'])['SunlightLux'].median()
                .reset_index())
    hour_lux_pivot = hour_lux.pivot(index='Hour', columns='Weather',
                                    values='SunlightLux').fillna(0)

    def traffic(v):
        if v >= 90: return '#2E7D32'
        if v >= 70: return '#F9A825'
        return '#C62828'

    # Build raw model grid then monotonise across lux for each time row
    raw_grid = np.zeros((len(times), len(lux_bins)))
    for ih, lux in enumerate(lux_bins):
        for it, t in enumerate(times):
            raw_grid[it, ih] = s2_predict(lux, t)
    # Monotonic envelope (non-decreasing in lux for each fixed time)
    env_grid = raw_grid.copy()
    for it in range(len(times)):
        env_grid[it, :] = np.maximum.accumulate(env_grid[it, :])

    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 0.75, 1.0],
                          width_ratios=[2, 1.1], hspace=0.55, wspace=0.22)

    # (a) Monotonic envelope heatmap
    axA = fig.add_subplot(gs[0, 0])
    for it in range(len(times)):
        for ih in range(len(lux_bins)):
            v = env_grid[it, ih]
            axA.add_patch(plt.Rectangle((ih - 0.5, it - 0.5), 1, 1,
                          facecolor=traffic(v), edgecolor='white', linewidth=1.5))
            axA.text(ih, it, f'{v:.0f}', ha='center', va='center',
                     color='white', fontsize=9, fontweight='bold')
    axA.set_xlim(-0.5, len(lux_bins)-0.5); axA.set_ylim(-0.5, len(times)-0.5)
    axA.set_xticks(range(len(lux_bins)))
    axA.set_xticklabels([f'{l/1000:.0f}k' for l in lux_bins])
    axA.set_yticks(range(len(times))); axA.set_yticklabels(times)
    axA.set_xlabel('Sunlight intensity (lux)'); axA.set_ylabel(TIME_LABEL)
    axA.set_title('(a) Stage-2 monotonic envelope (lower-bound DegEff %)',
                  loc='left', fontsize=11)
    axA.set_aspect('auto'); axA.grid(False)

    # (b) Hour -> lux reference table
    axB = fig.add_subplot(gs[0, 1]); axB.axis('off')
    axB.set_title('(b) Hour → measured lux reference\n(median across runs)',
                  loc='left', fontsize=11)
    header = ['Hour', 'Sunny\nSummer', 'Sunny\nWinter', 'Cloudy\nWinter']
    hours_sorted = sorted(hour_lux_pivot.index.tolist())
    rows = []
    for h in hours_sorted:
        row = [str(h)]
        for w in ['sunny_summer','sunny_winter','cloudy_winter']:
            v = hour_lux_pivot.loc[h, w] if w in hour_lux_pivot.columns else 0
            row.append(f'{v/1000:.0f}k' if v > 0 else '—')
        rows.append(row)
    tbl = axB.table(cellText=rows, colLabels=header, loc='upper center',
                    cellLoc='center', colColours=['#ECEFF1']*4)
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)

    # (c) Min time to ≥90% across lux bins (uses envelope)
    axC = fig.add_subplot(gs[1, :])
    mt = []
    for ih, lux in enumerate(lux_bins):
        times_dense = np.arange(3, 61, 1)
        raw_yps = np.array([s2_predict(lux, t) for t in times_dense])
        # Apply the envelope across lux here as well:
        # for each time, take the best model output at or below this lux
        env_yps = []
        for t in times_dense:
            env_val = max([s2_predict(l, t) for l in lux_bins if l <= lux] + [s2_predict(lux, t)])
            env_yps.append(env_val)
        env_yps = np.array(env_yps)
        reached = np.where(env_yps >= 90)[0]
        mt.append(times_dense[reached[0]] if len(reached) else np.nan)
    xb = np.arange(len(lux_bins))
    bars = axC.bar(xb, [m if np.isfinite(m) else 0 for m in mt],
                   color=['#C62828' if not np.isfinite(m)
                          else ('#2E7D32' if m <= 30 else '#F9A825') for m in mt],
                   edgecolor='black', linewidth=0.6, width=0.6)
    for b, m in zip(bars, mt):
        if np.isfinite(m):
            axC.text(b.get_x()+b.get_width()/2, m+1, f'{int(m)} min',
                     ha='center', fontsize=10, fontweight='bold')
        else:
            axC.text(b.get_x()+b.get_width()/2, 1.5, '✗ never under envelope',
                     ha='center', fontsize=9, color='#C62828', rotation=90)
    axC.set_xticks(xb); axC.set_xticklabels([f'{l/1000:.0f}k' for l in lux_bins])
    axC.set_xlabel('Sunlight intensity (lux)'); axC.set_ylabel('Min time to\n≥90% (min)')
    axC.set_title('(c) Practical guide — minimum reaction time to reach 90% DegEff (envelope)',
                  loc='left', fontsize=11)
    axC.set_ylim(0, 70)

    # (d) Raw experimental scatter at t=60 (shows the non-monotonicity)
    axD = fig.add_subplot(gs[2, :])
    exp_t60 = df2[df2['F8_TimeStep']==60].groupby('SunlightLux')['F13_Degradation'].mean().reset_index()
    axD.scatter(exp_t60['SunlightLux']/1000, exp_t60['F13_Degradation'],
                s=70, color='#1565C0', edgecolor='black', linewidth=0.7, zorder=3,
                label='Experimental mean at t=60 min')
    # Raw model prediction line at t=60
    lux_smooth = np.linspace(10000, 140000, 200)
    axD.plot(lux_smooth/1000, [s2_predict(l, 60) for l in lux_smooth],
             '-', color='#C62828', linewidth=1.6, alpha=0.85,
             label='Stage-2 raw model at t=60')
    axD.axhline(90, ls=':', color='gray', alpha=0.7)
    axD.set_xlabel('Sunlight intensity (lux, thousands)'); axD.set_ylabel('DegEff (%)')
    axD.set_title('(d) Raw data and raw model at t=60 min  '
                  '(four experiments around 76k–98k lux underperform; envelope smooths this)',
                  loc='left', fontsize=10)
    axD.set_ylim(0, 105); axD.legend(loc='lower right', fontsize=8.5)

    legend_tl = [Patch(facecolor='#2E7D32', label='≥90%'),
                 Patch(facecolor='#F9A825', label='70–90%'),
                 Patch(facecolor='#C62828', label='<70%')]
    fig.legend(handles=legend_tl, loc='upper right', bbox_to_anchor=(0.99, 1.00),
               ncol=3, fontsize=9, frameon=True)
    fig.suptitle('Operational guideline — v3 (uses MEASURED sunlight intensity)',
                 fontsize=13, fontweight='bold', y=1.02)
    save_fig('fig_E_operational_guide.png')


    # ----------------------------------------------------------------------------
    #  FIG F: not generated (DegEff vs cumulative light dose)
    #  Each run lasts 60 min with one averaged lux value, so lux x time adds no
    #  new information. fig_05_stage2_lux_temp covers this instead.
    # ----------------------------------------------------------------------------


    # ----------------------------------------------------------------------------
    #  FIG I: Stage-1 synthesis design surface (F1 x F3)
    # ----------------------------------------------------------------------------
if want('I'):
    banner('FIG I — Stage-1 synthesis design surface')

    # The synthesis-side Stage-1 inputs are F1, F2 and F3. The surface uses
    # F1 (CalcTemp) x F3 (CalcTime) with F2 (ramp) fixed at 8 °C/min.
    temp_grid = np.linspace(300, 750, 60)
    time_grid = np.linspace(1, 5, 50)
    TT, NN = np.meshgrid(temp_grid, time_grid)
    ZZ = np.zeros_like(TT)
    for i in range(TT.shape[0]):
        for j in range(TT.shape[1]):
            ZZ[i, j] = s1_predict(TT[i, j], 8, NN[i, j], 60)

    fig, ax = plt.subplots(figsize=(10.5, 7))
    cf = ax.contourf(TT, NN, ZZ, levels=np.linspace(60, 102, 22), cmap='viridis')
    cont = ax.contour(TT, NN, ZZ, levels=[80, 90, 95, 99], colors='white',
                      linewidths=0.9, alpha=0.7)
    ax.clabel(cont, inline=True, fontsize=8, fmt='%d%%')
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('Predicted DegEff at t=60 min (%)')

    # Overlay catalysts: marker size ∝ empirical DegEff, shape = morphology
    # (morphology comes from the characterisation table; it is not a model input)
    for _, r in rank.iterrows():
        m = MORPH_MARKERS.get(int(r['F7_Morphology']), 'o')
        sz = 60 + (r['MeanDegEff_60min'] / 100.0) * 220
        is_anr = (r['CatalystID'] == 'ANR-650-8-4')
        ax.scatter(r['F1_CalcTemp'], r['F3_CalcTime'], s=sz, marker=m,
                   facecolor='none' if not is_anr else '#F57C00',
                   edgecolor='white' if is_anr else 'black',
                   linewidth=1.8 if is_anr else 1.1, zorder=5)
    ax.scatter(650, 4, marker='*', s=420, color='#F57C00', edgecolor='black',
               linewidth=1.2, zorder=7, label='ANR-650-8-4 (deployed)')
    ax.set_xlabel('Calcination temperature, F1 (°C)')
    ax.set_ylabel('Calcination time, F3 (h)')
    ax.set_title('v3 Stage-1 synthesis design landscape (F2 ramp fixed at 8 °C/min)\n'
                 'Marker size ∝ experimental DegEff; shape = observed morphology',
                 loc='left', fontsize=11)
    morph_leg = [Line2D([0],[0], marker=MORPH_MARKERS[k], color='black', linestyle='',
                        markerfacecolor='none', markersize=9, label=MORPH_NAMES[k])
                 for k in [3, 4, 5]]
    key_leg = [Line2D([0],[0], marker='*', color='#F57C00', linestyle='', markersize=15,
                      markeredgecolor='black', label='ANR-650-8-4')]
    ax.legend(handles=morph_leg + key_leg, loc='lower left', fontsize=8.5, ncol=2)
    ax.grid(False)
    plt.tight_layout()
    save_fig('fig_I_synthesis_surface.png')


    # ----------------------------------------------------------------------------
    #  FIG J: Cross-stage bridge, DE optimum vs ANR-650-8-4
    # ----------------------------------------------------------------------------
if want('J'):
    banner('FIG J — Cross-stage bridge (v3)')

    # Differential evolution on the Stage-1 model over the synthesis space
    # (F1, F2, F3) at baseline operating conditions and t=60 min. Bounds are
    # limited to the ranges covered by the dataset.
    def neg_deg(syn):
        F1, F2, F3 = syn
        return -s1_predict(F1, F2, F3, 60)

    DE_BOUNDS = [(550, 750), (2, 10), (1, 5)]
    de_res = differential_evolution(neg_deg, DE_BOUNDS, seed=SEED, maxiter=300,
                                    popsize=25, tol=1e-7, polish=True,
                                    updating='deferred', workers=1)
    opt_F1, opt_F2, opt_F3 = de_res.x
    opt_deg = -de_res.fun

    anr_pred = s1_predict(650, 8, 4, 60)
    anr_emp = float(rank[rank['CatalystID']=='ANR-650-8-4']['MeanDegEff_60min'].iloc[0])
    anr_emp_std = float(rank[rank['CatalystID']=='ANR-650-8-4']['StdDegEff_60min'].iloc[0])

    print(f'  DE optimum: F1={opt_F1:.0f}°C, F2={opt_F2:.2f}°C/min, F3={opt_F3:.2f}h')
    print(f'             predicted DegEff = {opt_deg:.2f}%')
    print(f'  ANR-650-8-4 predicted = {anr_pred:.2f}%   empirical = {anr_emp:.2f}±{anr_emp_std:.2f}%')
    print(f'  Model gap: |opt − ANR_pred| = {abs(opt_deg - anr_pred):.2f}%')

    with open(f'{TBLDIR}/11_linking_validation_v3.json', 'w') as f:
        json.dump({
            'DE_optimum_F1_CalcTemp_C': float(opt_F1),
            'DE_optimum_F2_CalcRamp_C_per_min': float(opt_F2),
            'DE_optimum_F3_CalcTime_h': float(opt_F3),
            'DE_optimum_predicted_DegEff_60min': float(opt_deg),
            'ANR_650_8_4_predicted_DegEff_60min': float(anr_pred),
            'ANR_650_8_4_empirical_mean_60min': float(anr_emp),
            'ANR_650_8_4_empirical_std_60min': float(anr_emp_std),
            'model_gap_pct': float(abs(opt_deg - anr_pred)),
            'note': ('v3 Stage-1 DE search constrained to synthesis-input space '
                     '(F1, F2, F3) only. F4-F7 are not model inputs.'),
        }, f, indent=2)

    # The prediction is capped at 100 % and is flat near the optimum, so the
    # exact DE point is not unique. Record how large the top plateau is.
    _g1, _g2, _g3 = np.linspace(550, 750, 21), np.linspace(2, 10, 9), np.linspace(1, 5, 9)
    _grid = np.array([[a, b, c] for a in _g1 for b in _g2 for c in _g3])
    _pred = np.array([s1_predict(a, b, c, 60) for a, b, c in _grid])
    _top = _pred >= opt_deg - 0.5
    pd.DataFrame([{
        'objective': 'Stage-1 XGBoost predicted DegEff at 60 min, baseline reaction conditions',
        'DE_F1_C': opt_F1, 'DE_F2_C_per_min': opt_F2, 'DE_F3_h': opt_F3,
        'DE_predicted_pct': opt_deg,
        'ANR_650_8_4_predicted_pct': anr_pred,
        'grid_points': len(_grid),
        'grid_points_within_0.5pct_of_optimum': int(_top.sum()),
        'plateau_F1_range_C': f'{_grid[_top,0].min():.0f}-{_grid[_top,0].max():.0f}' if _top.any() else '',
        'plateau_F2_range': f'{_grid[_top,1].min():.0f}-{_grid[_top,1].max():.0f}' if _top.any() else '',
        'plateau_F3_range_h': f'{_grid[_top,2].min():.1f}-{_grid[_top,2].max():.1f}' if _top.any() else '',
        'DE_settings': 'scipy differential_evolution, best1bin, popsize=25, maxiter=300, '
                       'mutation=(0.5,1), recombination=0.7, tol=1e-7, polish=True, seed=42',
    }]).to_csv(f'{TBLDIR}/18_de_optimum_plateau.csv', index=False)

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.25)

    # (a) Conceptual flow
    axL = fig.add_subplot(gs[0, 0]); axL.axis('off')
    axL.set_xlim(0, 10); axL.set_ylim(0, 10)
    boxes = [
        (1.4, 'STAGE 1\n14 catalyst\nrecipes\n(F1, F2, F3)', '#1565C0'),
        (4.0, 'Stage-1 ML\n+ Differential\nEvolution', '#7B1FA2'),
        (6.6, 'Near-optimal\nrecipe\nconfirmed', '#2E7D32'),
        (9.0, 'STAGE 2\ndeploy\nANR-650-8-4', '#F57C00'),
    ]
    for x, label, color in boxes:
        axL.add_patch(plt.Rectangle((x-1.0, 4.0), 2.0, 2.4, facecolor=color,
                      alpha=0.18, edgecolor=color, linewidth=2, zorder=2))
        axL.text(x, 5.2, label, ha='center', va='center', fontsize=9.5,
                 fontweight='bold', color=color, zorder=3)
    for x0, x1 in [(2.4, 3.0), (5.0, 5.6), (7.6, 8.0)]:
        axL.annotate('', xy=(x1, 5.2), xytext=(x0, 5.2),
                     arrowprops=dict(arrowstyle='-|>', lw=2.2, color='#37474F'))

    # (b) Bar comparison
    axR = fig.add_subplot(gs[0, 1])
    cats = ['DE-predicted\noptimum', 'ANR-650-8-4\nmodel pred.', 'ANR-650-8-4\nempirical mean']
    vals = [opt_deg, anr_pred, anr_emp]
    errs = [0, 0, anr_emp_std]
    colors_j = ['#C62828', '#7B1FA2', '#F57C00']
    bars = axR.bar(range(3), vals, yerr=errs, capsize=6, color=colors_j,
                   edgecolor='black', linewidth=0.7, width=0.6, alpha=0.88)
    axR.set_xticks(range(3)); axR.set_xticklabels(cats, fontsize=9)
    axR.set_ylabel('DegEff at t=60 min (%)')
    ymin = min(vals) - max(errs) - 3
    axR.set_ylim(max(ymin, 80), 108)
    axR.set_title('(b) Linking validation (v3): predicted vs deployed',
                  loc='left', fontsize=11)
    for b, v in zip(bars, vals):
        axR.text(b.get_x()+b.get_width()/2, v+0.6, f'{v:.1f}%', ha='center',
                 fontsize=9.5, fontweight='bold')

    fig.suptitle('Cross-stage bridge: Stage-1 DE optimum and the deployed catalyst\n'
                 'using only synthesis-input features (F1, F2, F3)',
                 fontsize=12, fontweight='bold', y=1.04)
    save_fig('fig_J_cross_stage_bridge.png')


    # ----------------------------------------------------------------------------
    #  SUMMARY
    # ----------------------------------------------------------------------------
    banner('Chemistry figures (v3.1) — empirical / hybrid block complete')
    chem = ['fig_A_catalyst_radar','fig_B_temperature_response','fig_C_kinetics',
            'fig_D_interference','fig_E_operational_guide',
            'fig_G_stability_suite','fig_H_shap_by_category','fig_I_synthesis_surface',
            'fig_J_cross_stage_bridge']
    print(f'  {len(chem)} chemistry figures produced (A, B, C, D, E, G, H, I, J):')
    for f in chem:
        print(f'    • {f}.png')
    print('  (Fig F not generated: cumulative light dose is not used.)')
    print(f'\n  Supporting tables:')
    for t in ['07_catalyst_characterisation.csv','08_catalyst_empirical_ranking.csv',
              '09_interference_quantification.csv','10_kinetic_rate_constants.csv',
              '11_linking_validation_v3.json']:
        print(f'    • {t}')
    print(f'\n  All saved under {OUT}/')


    # ----------------------------------------------------------------------------
    #  MODEL-DRIVEN FIGURES (K, L, M, N)
    #  These use the Stage-1 and Stage-2 models for things the raw data cannot
    #  show directly: counterfactual scenarios, optimum regions and operating
    #  nomograms.
    # ----------------------------------------------------------------------------


    # ----------------------------------------------------------------------------
    #  FIG K: One-at-a-time (OAT) response curves (8-panel grid)
    # ----------------------------------------------------------------------------
    #  Each panel sweeps one feature over its observed range with all others held
    #  at the deployed baseline. Evaluated at t=20 min (mid-kinetic regime), where
    #  the features actually separate. Some features have little effect here
    #  (F2 ramp; F12 scavengers saturate to 100 at t=60).
    # ----------------------------------------------------------------------------
if want('K'):
    banner('FIG K — Model-predicted OAT response surfaces')

    # Baseline conditions (ANR-650-8-4, no interferents, mid-kinetic time)
    BASE = dict(F1=650, F2=8, F3=4, t=20, F9=11, F10=11, F11=0, F12=0)
    def p1(**kw):
        F1, F2, F3, t = kw.get('F1',BASE['F1']), kw.get('F2',BASE['F2']), kw.get('F3',BASE['F3']), kw.get('t',BASE['t'])
        F9, F10, F11, F12 = kw.get('F9',BASE['F9']), kw.get('F10',BASE['F10']), kw.get('F11',BASE['F11']), kw.get('F12',BASE['F12'])
        return float(np.clip(stage1_mdl.predict([[F1,F2,F3,t,F9,F10,F11,F12]])[0], 0, 100))

    # Feature ranges (dataset spans plus a small extrapolation buffer)
    # F11 (ion mixtures, ppm per the dataset header) and F12 (scavenger types,
    # categorical) take only a few discrete observed levels, and for F11 the
    # species identity varies with the level. We therefore show them as discrete
    # bar charts rather than continuous sweeps.
    oat_continuous_specs = [
        ('F1_CalcTemp',  np.linspace(300, 750, 60),   '°C',         'Calcination temperature',  650, (200, 800)),
        ('F2_CalcRamp',  np.linspace(2,   10, 30),    '°C/min',     'Calcination ramp',         8,   (1, 11)),
        ('F3_CalcTime',  np.linspace(1,   5,  30),    'h',          'Calcination time',         4,   (0, 6)),
        ('F9_RhBConc',   np.linspace(3,   19, 30),    '×10⁻⁵ M',    'RhB concentration',        11,  (6, 14)),
        ('F10_pH',       np.linspace(3,   13, 30),    '',           'Solution pH',              11,  (2, 14)),
    ]
    # Categorical codes: observed levels shown as bars
    oat_categorical_specs = [
        ('F11_Ions',     [0, 20, 30, 50],
            {0:'none', 20:'cation mix', 30:'anion mix', 50:'combined'},
            'Co-existing ions (F11)', 0),
        ('F12_Scavengers', [0, 6, 7, 8, 9],
            {0:'none', 6:'e⁻ scav.', 7:'h⁺ scav.', 8:'O₂•⁻ scav.', 9:'•OH scav.'},
            'Radical scavengers (F12)', 0),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.4))
    axes_flat = axes.flatten()

    # Continuous panels
    for idx, (feat, sweep, units, title, baseline_val, xlim) in enumerate(oat_continuous_specs):
        ax = axes_flat[idx]
        feat_key = feat.split('_')[0]
        yp = [p1(**{feat_key: v}) for v in sweep]
        ax.plot(sweep, yp, '-', color='#1565C0', linewidth=2.3, zorder=3)
        ax.fill_between(sweep, yp, alpha=0.12, color='#1565C0')
        base_y = p1()
        ax.scatter([baseline_val], [base_y], s=110, color='#F57C00',
                   edgecolor='black', linewidth=1.2, zorder=5,
                   label=f'Deployed = {baseline_val}')
        # Mark observed dataset values
        obs_vals = sorted(df1_train[feat].dropna().unique())
        for v in obs_vals:
            ax.axvline(v, ls=':', color='#777', alpha=0.35, linewidth=0.7, zorder=1)
        ax.set_xlabel(f'{title}  ({units})' if units else title)
        ax.set_ylabel('Predicted degradation efficiency (%)')
        ax.set_ylim(0, 105)
        ax.set_xlim(*xlim)
        ax.set_title(f'({"abcdefg"[idx]}) {title.split(" (")[0]}', loc='left', fontsize=10.5)
        ax.legend(loc='lower right', fontsize=7.5)
        span = max(yp) - min(yp)
        ax.text(0.04, 0.95, f'Δ range: {span:.1f}%', transform=ax.transAxes,
                fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1',
                          edgecolor='#F57C00'))

    # Categorical panels (F11, F12)
    for j, (feat, levels, level_names, title, baseline_val) in enumerate(oat_categorical_specs):
        idx = len(oat_continuous_specs) + j
        ax = axes_flat[idx]
        feat_key = feat.split('_')[0]
        ys = [p1(**{feat_key: v}) for v in levels]
        colors_bar = ['#2E7D32' if v == baseline_val else '#C62828' for v in levels]
        bars = ax.bar(range(len(levels)), ys, color=colors_bar,
                      edgecolor='black', linewidth=0.6, width=0.6)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([level_names[v] for v in levels], rotation=20, ha='right',
                           fontsize=8.5)
        ax.set_ylabel('Predicted degradation efficiency (%)')
        ax.set_ylim(0, 105)
        ax.set_title(f'({"abcdefg"[idx]}) {title.split(" (")[0]}', loc='left',
                     fontsize=10.5)
        for b, v in zip(bars, ys):
            ax.text(b.get_x()+b.get_width()/2, v+1.5, f'{v:.0f}',
                    ha='center', fontsize=8.5, fontweight='bold')
        span = max(ys) - min(ys)
        ax.text(0.04, 0.95, f'Δ range: {span:.1f}%', transform=ax.transAxes,
                fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1',
                          edgecolor='#F57C00'))

    # Panel (h): Feature-8 Time Step OAT response (replaces the caption text box)
    axT = axes_flat[7]
    t_sweep = np.linspace(0, 60, 60)
    yp_t = [p1(t=v) for v in t_sweep]
    axT.plot(t_sweep, yp_t, '-', color='#1565C0', linewidth=2.3, zorder=3)
    axT.fill_between(t_sweep, yp_t, alpha=0.12, color='#1565C0')
    axT.scatter([20], [p1(t=20)], s=110, color='#F57C00', edgecolor='black',
                linewidth=1.2, zorder=5, label='Probe t = 20')
    for v in sorted(df1_train['F8_TimeStep'].dropna().unique()):
        axT.axvline(v, ls=':', color='#777', alpha=0.35, linewidth=0.7, zorder=1)
    axT.set_xlabel('Irradiation Time (min)')
    axT.set_ylabel('Predicted degradation efficiency (%)')
    axT.set_ylim(0, 105)
    axT.set_xlim(0, 60)
    axT.set_title('(h) Time Step', loc='left', fontsize=10.5)
    axT.legend(loc='lower right', fontsize=7.5)
    span_t = max(yp_t) - min(yp_t)
    axT.text(0.04, 0.95, f'Δ range: {span_t:.1f}%', transform=axT.transAxes,
             fontsize=8, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1',
                       edgecolor='#F57C00'))

    fig.suptitle('Stage-1 model: one-at-a-time response surfaces  '
                 '(t = 20 min, ANR-650-8-4 baseline)',
                 fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    save_fig('fig_K_OAT_response_surfaces.png')


    # ----------------------------------------------------------------------------
    #  FIG L: Robust optimum region (model-predicted "good enough" zones)
    # ----------------------------------------------------------------------------
    #  Rather than a single optimum (Fig J), this shows the region of synthesis
    #  parameters (F1, F3) predicted to reach ≥95% DegEff at t=60 min under four
    #  scenarios:
    #     (a) baseline operating conditions
    #     (b) + cation mixture (ionic interference)
    #     (c) + O₂•⁻ scavenger (radical scavenging)
    #     (d) low-light deployment (approximated by t=30 instead of t=60)
    #  Any (F1, F3) inside the shaded zone is predicted to reach ≥95% for that
    #  scenario.
    #
    # ----------------------------------------------------------------------------
if want('L'):
    banner('FIG L — Robust optimum regions across operational scenarios')

    f1_grid = np.linspace(550, 780, 70)
    f3_grid = np.linspace(1, 5, 50)
    F1G, F3G = np.meshgrid(f1_grid, f3_grid)

    def grid_pred(t=60, F9=11, F10=11, F11=0, F12=0):
        Z = np.zeros_like(F1G)
        for i in range(F1G.shape[0]):
            for j in range(F1G.shape[1]):
                Z[i,j] = p1(F1=F1G[i,j], F2=8, F3=F3G[i,j],
                            t=t, F9=F9, F10=F10, F11=F11, F12=F12)
        return Z

    print('  Computing 4 grids (≈14k predictions)...')
    Z_a = grid_pred()                       # baseline
    Z_b = grid_pred(F11=20)                 # cation mix
    Z_c = grid_pred(F12=8)                  # O₂•⁻ scavenger
    Z_d = grid_pred(t=30)                   # low effective dose (short time)

    scenarios = [
        (Z_a, '(a) Baseline\nF11=0, F12=0, t=60 min', 'No interferent — operational best case'),
        (Z_b, '(b) + cation mixture\nF11=20, t=60 min', 'Worst ion interference'),
        (Z_c, '(c) + O₂•⁻ scavenger\nF12=8, t=60 min',  'Worst radical scavenger'),
        (Z_d, '(d) Half-time stress\nt=30 min, baseline', 'Lower effective light dose'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes_flat = axes.flatten()
    for ax, (Z, title, sub) in zip(axes_flat, scenarios):
        cf = ax.contourf(F1G, F3G, Z, levels=np.linspace(0, 102, 25), cmap='RdYlGn',
                         vmin=0, vmax=100)
        # Highlight ≥95% region
        ax.contour(F1G, F3G, Z, levels=[95], colors='black', linewidths=2.0,
                   linestyles='-')
        ax.contour(F1G, F3G, Z, levels=[80, 90], colors='black', linewidths=0.8,
                   linestyles=':', alpha=0.6)
        # Compute area of the ≥95% region
        cell_area = (f1_grid[1] - f1_grid[0]) * (f3_grid[1] - f3_grid[0])
        total_area = (f1_grid[-1] - f1_grid[0]) * (f3_grid[-1] - f3_grid[0])
        pct_robust = (Z >= 95).sum() * cell_area / total_area * 100
        # Mark ANR-650-8-4
        ax.scatter([650], [4], s=200, marker='*', color='#1976D2',
                   edgecolor='white', linewidth=1.5, zorder=5)
        ax.text(652, 4.05, 'ANR-650-8-4', fontsize=9, fontweight='bold', color='#1976D2')
        ax.set_xlabel('F1 — Calcination temperature (°C)')
        ax.set_ylabel('F3 — Calcination time (h)')
        ax.set_title(title, loc='left', fontsize=10.5)
        # Annotation: robust-zone fraction
        ax.text(0.04, 0.96,
                f'≥95% zone covers\n{pct_robust:.1f}% of (F1,F3) space',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='black', linewidth=0.8))
        ax.grid(False)

    # Shared colorbar
    cbar = fig.colorbar(cf, ax=axes_flat.tolist(), pad=0.02, fraction=0.022)
    cbar.set_label('Stage-1 predicted DegEff (%)')
    fig.suptitle('Robust optimum regions (F2 ramp held at 8 °C/min)  —  '
                 'where ANR-650-8-4 sits, and how the safe zone shrinks under stress',
                 fontsize=12, fontweight='bold', y=1.00)
    save_fig('fig_L_robust_optimum_regions.png')


    # ----------------------------------------------------------------------------
    #  FIG M: Time-to-target operating nomogram (Stage 2)
    # ----------------------------------------------------------------------------
    #  For a measured sunlight intensity, how long does it take to reach
    #  50 / 75 / 90 / 95% degradation? Uses the same monotonic envelope as Fig E,
    #  so the reading is "at this lux or higher, at least this much within this time".
    # ----------------------------------------------------------------------------
if want('M'):
    banner('FIG M — Time-to-target operational nomogram')

    lux_axis = np.linspace(15000, 140000, 100)
    times_dense = np.arange(2, 65, 1)
    targets = [50, 75, 90, 95]
    target_colors = {50:'#C62828', 75:'#F57C00', 90:'#1976D2', 95:'#2E7D32'}

    # For each lux, compute the monotonic envelope across time
    # envelope(lux, t) = max_{lux' ≤ lux} predict(lux', t)
    def env_predict(lux, t, lux_axis):
        return max(s2_predict(L, t) for L in lux_axis if L <= lux) if any(L <= lux for L in lux_axis) else s2_predict(lux, t)

    # Faster: precompute the prediction grid
    grid = np.zeros((len(lux_axis), len(times_dense)))
    for i, lux in enumerate(lux_axis):
        for j, t in enumerate(times_dense):
            grid[i, j] = s2_predict(lux, t)
    # Monotonise across lux at each fixed time
    env = np.maximum.accumulate(grid, axis=0)

    # Time-to-target curves
    target_curves = {}
    for tgt in targets:
        tt = []
        for i in range(len(lux_axis)):
            reached = np.where(env[i, :] >= tgt)[0]
            tt.append(times_dense[reached[0]] if len(reached) else np.nan)
        target_curves[tgt] = np.array(tt, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # Time-to-target curves
    for tgt in targets:
        valid = ~np.isnan(target_curves[tgt])
        ax.plot(lux_axis[valid]/1000, target_curves[tgt][valid], '-',
                color=target_colors[tgt], linewidth=2.4, label=f'≥{tgt}% DegEff')
    ax.set_xlabel('Sunlight intensity (\u00d710\u00b3 lux)', fontsize=13)
    ax.set_ylabel('Predicted degradation time (min)', fontsize=13)
    ax.set_xlim(15, 140); ax.set_ylim(0, 65)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='lower left', fontsize=11, title='Target', title_fontsize=11,
              framealpha=0.95)
    # Mark canonical weather lux values from data. Labels anchored just inside the
    # top of the axes (va='top') so the three-line text stays within the frame.
    canonical = {'Sunny\nsummer\nnoon': 134, 'Sunny\nwinter\nnoon': 120, 'Cloudy\nwinter\nnoon': 55}
    for label, klux in canonical.items():
        ax.axvline(klux, ls=':', color='#37474F', alpha=0.6, linewidth=0.9)
        ax.text(klux, 63.5, label, ha='center', va='top', fontsize=10,
                color='#37474F', fontweight='bold')

    plt.tight_layout()
    save_fig('fig_M_time_to_target_nomogram.png')


    # ----------------------------------------------------------------------------
    #  FIG M2: Time-to-target bar chart
    # ----------------------------------------------------------------------------
    #  Same Stage-2 monotonic envelope as Fig M, rendered as grouped bars at the
    #  canonical lux levels. Recomputed here so this figure is identical to the
    #  Fig M curves and builds standalone under ONLY=fig_M2.
    # ----------------------------------------------------------------------------
if want('M2'):
    banner('FIG M2 — Time-to-target bar chart (canonical lux levels)')

    lux_axis_m2 = np.linspace(15000, 140000, 100)
    times_m2    = np.arange(2, 65, 1)
    targets_m2  = [50, 75, 90, 95]
    tcol_m2     = {50:'#C62828', 75:'#F57C00', 90:'#1976D2', 95:'#2E7D32'}

    grid_m2 = np.zeros((len(lux_axis_m2), len(times_m2)))
    for i, lx in enumerate(lux_axis_m2):
        for j, t in enumerate(times_m2):
            grid_m2[i, j] = s2_predict(lx, t)
    env_m2 = np.maximum.accumulate(grid_m2, axis=0)   # monotonise across lux

    lux_table = [25000, 50000, 75000, 100000, 120000, 134000]
    lux_lbls  = [f'{l//1000}' for l in lux_table]
    bar_data = {tgt: [] for tgt in targets_m2}
    for lx in lux_table:
        ridx = np.argmin(np.abs(lux_axis_m2 - lx))
        for tgt in targets_m2:
            reached = np.where(env_m2[ridx, :] >= tgt)[0]
            bar_data[tgt].append(int(times_m2[reached[0]]) if len(reached) else 0)

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    x = np.arange(len(lux_table)); w = 0.2
    for i, tgt in enumerate(targets_m2):
        v = bar_data[tgt]
        bars = ax.bar(x + (i - 1.5)*w, v, w, label=f'\u2265{tgt}',
                      color=tcol_m2[tgt], edgecolor='black', linewidth=0.6)
        for b, val in zip(bars, v):
            if val > 0:
                ax.text(b.get_x()+b.get_width()/2, val+0.8, f'{val}',
                        ha='center', va='bottom', fontsize=11.5, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(lux_lbls, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel('Irradiation time (min)', fontsize=13)
    ax.set_xlabel('Sunlight intensity (\u00d710\u00b3 lux)', fontsize=13)
    ax.set_ylim(0, 66)
    ax.legend(title='RhB degradation efficiency (%)', ncol=4, fontsize=11,
              title_fontsize=11.5, loc='lower center', bbox_to_anchor=(0.5, 1.0),
              frameon=True, columnspacing=1.6, handletextpad=0.5)
    plt.tight_layout()
    save_fig('fig_M2_nomogram_bars.png')


    # ----------------------------------------------------------------------------
    #  FIG N: Counterfactual stress test (Stage-1 model)
    # ----------------------------------------------------------------------------
    #  Uses the Stage-1 model to predict ANR-650-8-4 performance in scenarios
    #  that were not tested directly in Sheet 1 (or only in other combinations):
    #    - deployment baseline (no interferent)
    #    - + cation / + anion / + combined ion mixture (operational stress)
    #    - + O₂•⁻ scavenger (radical-mechanism probe)
    #    - acidic deployment (pH 3)
    #    - alkaline deployment (pH 13)
    #    - combined stress case (cation mix + pH 3)
    #
    # ----------------------------------------------------------------------------
if want('N'):
    banner('FIG N — Counterfactual stress test')

    # Evaluate at t = 30 min (operational time) under each scenario
    T_STRESS = 30
    scenarios_N = [
        ('Baseline\n(F11=0, F12=0, pH 11)', dict(),                                     '#2E7D32'),
        ('+ cation mixture\nFeCl₃ + CaCl₂\n(10 ppm each, 20 ppm)', dict(F11=20),     '#7B1FA2'),
        ('+ anion mixture\nNaCl + NaNO₃ + Na₂CO₃\n(10 ppm each, 30 ppm)', dict(F11=30), '#1976D2'),
        ('+ combined ions\nall five salts\n(10 ppm each, 50 ppm)', dict(F11=50),         '#C62828'),
        ('+ O₂•⁻ scavenger\n(F12=8)',         dict(F12=8),                                '#E65100'),
        ('Acidic pH\n(F10=3)',                dict(F10=3),                                '#5C6BC0'),
        ('Very alkaline\n(F10=13)',           dict(F10=13),                               '#26A69A'),
        ('Combined stress\ncase (lab)\n(cation mix + pH 3)', dict(F11=20, F10=3),         '#37474F'),
    ]
    # F11 is the total mixture concentration in ppm, with each salt at 10 ppm:
    #   Cation mix  (F11=20): FeCl3 + CaCl2                (2 salts x 10 ppm)
    #   Anion mix   (F11=30): NaCl + NaNO3 + Na2CO3        (3 salts x 10 ppm)
    #   Combined    (F11=50): all five salts               (5 salts x 10 ppm)
    # Per-salt dose is constant, so cation vs anion is compared at equal per-salt
    # concentration; the cation mix is more harmful despite the lower total ppm.
    # These are laboratory stress tests, not simulated real wastewater.

    # Evaluate at three reaction times
    times_eval = [15, 30, 60]
    results = np.zeros((len(scenarios_N), len(times_eval)))
    for i, (lab, perturb, color) in enumerate(scenarios_N):
        for j, t in enumerate(times_eval):
            results[i, j] = p1(t=t, **perturb)

    # Save to CSV
    counterfact_df = pd.DataFrame(results, columns=[f't={t}min' for t in times_eval])
    counterfact_df.insert(0, 'Scenario', [s[0].replace('\n',' ') for s in scenarios_N])
    counterfact_df.to_csv(f'{TBLDIR}/12_counterfactual_stress_test.csv', index=False)

    fig, ax = plt.subplots(figsize=(11, 11))
    y = np.arange(len(scenarios_N))
    w = 0.26
    for j, t in enumerate(times_eval):
        ypos = y + (j-1)*w
        bars = ax.barh(ypos, results[:, j], w,
                       label=f't = {t} min',
                       color=SEQ[j],
                       edgecolor='black', linewidth=0.5)
        for b, v in zip(bars, results[:, j]):
            ax.text(v + 1.2, b.get_y()+b.get_height()/2, f'{v:.0f}',
                    va='center', ha='left', fontsize=11, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in scenarios_N], fontsize=11)
    ax.invert_yaxis()                      # keep first scenario at the top
    ax.set_xlabel('Stage-1 predicted DegEff (%)')
    ax.set_xlim(0, 112)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3,
              fontsize=10, framealpha=0.95)
    plt.tight_layout()
    save_fig('fig_N_counterfactual_stress.png')


    # ----------------------------------------------------------------------------
    #  FIG O: not generated (iso-dose contours)
    #  Cumulative light-dose plots were dropped. The DegEff vs (lux, time) plane
    #  is covered by Fig E panel (a) and Fig M.
    # ----------------------------------------------------------------------------


    # Final summary
    banner('Chemistry figures (v3.1) — K, L, M, N block complete')
    extras = ['fig_K_OAT_response_surfaces', 'fig_L_robust_optimum_regions',
              'fig_M_time_to_target_nomogram', 'fig_N_counterfactual_stress']
    print(f'  {len(extras)} model-driven extension figures (K, L, M, N):')
    for f in extras:
        print(f'    • {f}.png')
    print('  (Fig O not generated: cumulative light dose is not used.)')
    print(f'\n  Supporting table:')
    print(f'    • 12_counterfactual_stress_test.csv')


    # ----------------------------------------------------------------------------
    #  FIG P: Multi-panel SHAP summary (Stage 1)
    # ----------------------------------------------------------------------------
    #  Four panels:
    #     (A) violin plot of SHAP value distributions per feature
    #     (B) global ranking by mean |SHAP|
    #     (C) beeswarm of individual impacts, coloured by feature value
    #     (D) category pie chart (catalyst properties / reaction conditions / kinetic)
    #  Stage 1 only; Stage 2 has three features, too few for this layout.
    # ----------------------------------------------------------------------------
if want('P'):
    banner('FIG P — NH3-style multi-panel SHAP (Stage 1)')


    # Sort features by mean |SHAP| (descending) for both the violin and the bar
    mean_abs_shap = np.abs(shap_vals1).mean(axis=0)
    order_desc = np.argsort(mean_abs_shap)[::-1]   # for bar / pie
    order_asc  = order_desc[::-1]                   # for violin / beeswarm (so top feature is at top)

    # Category mapping for Stage 1
    S1_CATEGORY = {
        'F1_CalcTemp':'Catalyst properties','F2_CalcRamp':'Catalyst properties',
        'F3_CalcTime':'Catalyst properties',
        'F8_TimeStep':'Kinetic',
        'F9_RhBConc':'Reaction conditions','F10_pH':'Reaction conditions',
        'F11_Ions':'Reaction conditions','F12_Scavengers':'Reaction conditions',
    }
    CAT_COLOURS_P = {
        'Catalyst properties': '#C62828',
        'Reaction conditions': '#1565C0',
        'Kinetic':             '#F9A825',
    }

    fig = plt.figure(figsize=(15.5, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.95],
                          hspace=0.45, wspace=0.55)

    # Panel (A): Violin SHAP distributions
    axA = fig.add_subplot(gs[0, 0])
    parts = axA.violinplot([shap_vals1[:, i] for i in order_asc],
                           positions=range(len(S1_FEATS)),
                           vert=False, showmeans=False, showmedians=True,
                           widths=0.85)
    for i, body in enumerate(parts['bodies']):
        feat = S1_FEATS[order_asc[i]]
        body.set_facecolor(CAT_COLOURS_P[S1_CATEGORY[feat]])
        body.set_edgecolor('black')
        body.set_alpha(0.7)
    for partname in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
        parts[partname].set_color('black')
        parts[partname].set_linewidth(0.8)
    axA.axvline(0, ls='--', color='gray', linewidth=0.8)
    axA.set_yticks(range(len(S1_FEATS)))
    axA.set_yticklabels([labels[i] for i in order_asc], fontsize=8.5)
    axA.set_xlabel('SHAP value (impact on DegEff, %-points)', fontsize=9)
    axA.set_title('(a) SHAP value distribution per feature', loc='left', fontsize=10.5)
    axA.grid(False, axis='y')

    # Panel (B): Global bar chart of mean |SHAP|
    axB = fig.add_subplot(gs[0, 1])
    yi = np.arange(len(S1_FEATS))
    bar_colors = [CAT_COLOURS_P[S1_CATEGORY[S1_FEATS[i]]] for i in order_asc]
    axB.barh(yi, mean_abs_shap[order_asc], color=bar_colors,
             edgecolor='black', linewidth=0.5)
    for i, v in enumerate(mean_abs_shap[order_asc]):
        axB.text(v + 0.15, i, f'{v:.2f}', va='center', fontsize=8)
    axB.set_yticks(yi)
    axB.set_yticklabels([labels[i] for i in order_asc], fontsize=8.5)
    axB.set_xlabel('Mean |SHAP value|  (%-points)', fontsize=9)
    axB.set_title('(b) Global feature importance ranking', loc='left', fontsize=10.5)
    axB.grid(False, axis='y')

    # Panel (C): Beeswarm of SHAP values, coloured by feature value
    axC = fig.add_subplot(gs[1, 0:2])
    np.random.seed(42)
    for i, fi in enumerate(order_asc):
        sv = shap_vals1[:, fi]
        fv = X1[:, fi]
        # Normalise feature value to [0,1] for colormap
        fv_min, fv_max = fv.min(), fv.max()
        fv_norm = (fv - fv_min) / max(fv_max - fv_min, 1e-9)
        jitter = np.random.uniform(-0.30, 0.30, size=len(sv))
        axC.scatter(sv, np.full_like(sv, i, dtype=float) + jitter,
                    c=fv_norm, cmap='coolwarm', vmin=0, vmax=1,
                    s=8, alpha=0.55, edgecolor='none')
    axC.axvline(0, ls='--', color='gray', linewidth=0.8)
    axC.set_yticks(range(len(S1_FEATS)))
    axC.set_yticklabels([labels[i] for i in order_asc], fontsize=8.5)
    axC.set_xlabel('SHAP value (impact on DegEff, %-points)', fontsize=9)
    axC.set_title('(c) Beeswarm — individual prediction impacts; colour = feature value',
                  loc='left', fontsize=10.5)
    axC.grid(False, axis='y')
    # Colour bar for the beeswarm
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cax = fig.add_axes([0.66, 0.10, 0.012, 0.32])
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 0.5, 1])
    cb.set_ticklabels(['low', 'mid', 'high'])
    cb.set_label('Feature value', fontsize=8.5)

    # Panel (D): Category pie
    axD = fig.add_subplot(gs[1, 2])
    cat_totals_P = {}
    for i, f in enumerate(S1_FEATS):
        cat = S1_CATEGORY[f]
        cat_totals_P[cat] = cat_totals_P.get(cat, 0) + mean_abs_shap[i]
    total_all = sum(cat_totals_P.values())
    cat_pcts = {k: v / total_all * 100 for k, v in cat_totals_P.items()}
    cat_names = list(cat_pcts.keys())
    cat_vals  = [cat_pcts[k] for k in cat_names]
    cat_cols  = [CAT_COLOURS_P[k] for k in cat_names]
    wedges, texts, autotexts = axD.pie(cat_vals, colors=cat_cols,
        labels=None, autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2),
        textprops=dict(fontsize=10, fontweight='bold', color='white'))
    axD.set_title('(d) Category share of total |SHAP|', loc='left', fontsize=10.5)
    # Custom legend on the right
    legend_entries = [Patch(facecolor=CAT_COLOURS_P[k], edgecolor='black',
                            label=f'{k}  ({cat_pcts[k]:.1f}%)') for k in cat_names]
    axD.legend(handles=legend_entries, loc='center left',
               bbox_to_anchor=(1.05, 0.5), fontsize=8.5, frameon=True)

    fig.suptitle('Stage-1 SHAP analysis — feature importance for photocatalytic '
                 'RhB degradation by ANR catalysts  [model attribution only]',
                 fontsize=12, fontweight='bold', y=0.99)
    save_fig('fig_P_shap_multipanel.png')


    # ----------------------------------------------------------------------------
    #  FIG Q: PCA of the Stage-1 dataset
    # ----------------------------------------------------------------------------
    #  Three panels:
    #     (A) Variance explained per PC + cumulative variance
    #     (B) Loadings heatmap (features × PCs)
    #     (C) 3D biplot: sample scores in PC1-PC2-PC3 with loading vectors
    #  The target F13 is included to show how it aligns with the input PCs.
    # ----------------------------------------------------------------------------
if want('Q'):
    banner('FIG Q — PCA group (Stage 1)')


    # PCA input: 8 features + target F13, so the loadings show where the target
    # projects in PC space.
    Q_FEATS = S1_FEATS + ['F13_Degradation']
    Q_LABELS = {
        'F1_CalcTemp':   'Calcination temp (°C)',
        'F2_CalcRamp':   'Calcination ramp (°C/min)',
        'F3_CalcTime':   'Calcination time (h)',
        'F8_TimeStep':   TIME_LABEL,
        'F9_RhBConc':    'RhB concentration (×10⁻⁵ M)',
        'F10_pH':        'Solution pH',
        'F11_Ions':      'Co-existing ions',
        'F12_Scavengers':'Radical scavengers',
        'F13_Degradation':'DegEff (%)',
    }
    X_pca_raw = df1_train[Q_FEATS].values.astype(float)
    scaler = StandardScaler()
    X_pca = scaler.fit_transform(X_pca_raw)
    pca = PCA(n_components=len(Q_FEATS))
    scores = pca.fit_transform(X_pca)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)   # standard loadings
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    print(f'  PCA: {len(Q_FEATS)} variables, '
          f'PC1-3 cumulative variance = {cum_var[2]*100:.1f}%')

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.6], width_ratios=[1, 1.15],
                          hspace=0.42, wspace=0.30)

    # Panel (A): scree + cumulative variance
    axA = fig.add_subplot(gs[0, 0])
    pcs = np.arange(1, len(Q_FEATS) + 1)
    axA.bar(pcs, exp_var * 100, color='#1565C0', edgecolor='black',
            linewidth=0.5, label='Individual variance')
    axA2 = axA.twinx()
    axA2.plot(pcs, cum_var * 100, 'o-', color='#C62828', linewidth=2, markersize=7,
              label='Cumulative variance')
    for p, v in zip(pcs, cum_var * 100):
        axA2.text(p, v + 1.5, f'{v:.0f}%', ha='center', fontsize=7.5, color='#C62828')
    axA.set_xlabel('Principal component')
    axA.set_ylabel('Individual variance explained (%)', color='#1565C0')
    axA.tick_params(axis='y', labelcolor='#1565C0')
    axA2.set_ylabel('Cumulative variance (%)', color='#C62828')
    axA2.tick_params(axis='y', labelcolor='#C62828')
    axA2.set_ylim(0, 110)
    axA.set_xticks(pcs)
    axA.set_title('(a) Variance explained per principal component',
                  loc='left', fontsize=11)
    axA2.spines['right'].set_visible(True)
    axA.grid(False)

    # Panel (B): loadings heatmap
    axB = fig.add_subplot(gs[0, 1])
    n_show = 5     # first 5 PCs
    heat = loadings[:, :n_show]
    im = axB.imshow(heat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axB.set_xticks(range(n_show))
    axB.set_xticklabels([f'PC{i+1}\n({exp_var[i]*100:.1f}%)' for i in range(n_show)],
                        fontsize=9)
    axB.set_yticks(range(len(Q_FEATS)))
    axB.set_yticklabels([Q_LABELS[f] for f in Q_FEATS], fontsize=9)
    for i in range(len(Q_FEATS)):
        for j in range(n_show):
            val = heat[i, j]
            col = 'white' if abs(val) > 0.5 else 'black'
            axB.text(j, i, f'{val:+.2f}', ha='center', va='center',
                     fontsize=8, color=col)
    cb = fig.colorbar(im, ax=axB, pad=0.02, fraction=0.04)
    cb.set_label('Loading')
    axB.set_title('(b) Component loadings (first 5 PCs)', loc='left', fontsize=11)
    axB.grid(False)

    # Panel (C): 3D biplot
    axC = fig.add_subplot(gs[1, :], projection='3d')
    # Score scatter coloured by F13 to show the target gradient in PC space
    target_col = X_pca_raw[:, -1]    # F13 values
    scat = axC.scatter(scores[:, 0], scores[:, 1], scores[:, 2],
                       c=target_col, cmap='viridis', s=18, alpha=0.7,
                       edgecolor='none')
    # Loading vectors, scaled to be visible alongside the scores
    max_score = max(abs(scores[:, :3]).max(), 1.0)
    arrow_scale = max_score * 1.0
    # Compact feature labels for biplot readability
    BIPLOT_LABELS = {
        'F1_CalcTemp':'CalcTemp','F2_CalcRamp':'Ramp','F3_CalcTime':'CalcTime',
        'F8_TimeStep':'Time','F9_RhBConc':'RhB','F10_pH':'pH',
        'F11_Ions':'Ions','F12_Scavengers':'Scav','F13_Degradation':'DegEff',
    }
    for i, feat in enumerate(Q_FEATS):
        x_arr, y_arr, z_arr = (loadings[i, 0] * arrow_scale,
                                loadings[i, 1] * arrow_scale,
                                loadings[i, 2] * arrow_scale)
        color = '#C62828' if feat == 'F13_Degradation' else '#37474F'
        lw    = 2.4       if feat == 'F13_Degradation' else 1.4
        axC.plot([0, x_arr], [0, y_arr], [0, z_arr], color=color, linewidth=lw)
        axC.text(x_arr * 1.12, y_arr * 1.12, z_arr * 1.12,
                 BIPLOT_LABELS[feat],
                 color=color, fontsize=9, fontweight='bold',
                 ha='center')
    axC.set_xlabel(f'PC1 ({exp_var[0]*100:.1f}%)')
    axC.set_ylabel(f'PC2 ({exp_var[1]*100:.1f}%)')
    axC.set_zlabel(f'PC3 ({exp_var[2]*100:.1f}%)')
    cb2 = fig.colorbar(scat, ax=axC, pad=0.10, fraction=0.025, shrink=0.6)
    cb2.set_label('DegEff (%)')
    axC.set_title('(c) 3D biplot — scores coloured by DegEff; arrows = feature loadings; '
                  'red = target F13',
                  loc='left', fontsize=10.5)

    fig.suptitle('Principal Component Analysis of the Stage-1 photocatalytic dataset',
                 fontsize=12.5, fontweight='bold', y=0.99)
    save_fig('fig_Q_PCA_group.png')


    # ----------------------------------------------------------------------------
    #  FIG R: Pearson correlation heatmap with significance (Stage 1)
    # ----------------------------------------------------------------------------
    #  Pearson correlation between each Stage-1 input and the target F13, with
    #  significance stars marking significant pairs
    #  (* p≤0.05, ** p≤0.01, *** p≤0.001).
    # ----------------------------------------------------------------------------
if want('R'):
    banner('FIG R — Pearson correlation heatmap (Stage 1)')


    R_FEATS = S1_FEATS + ['F13_Degradation']
    R_data = df1_train[R_FEATS].values.astype(float)

    n_f = len(R_FEATS)
    corr = np.zeros((n_f, n_f))
    pval = np.zeros((n_f, n_f))
    for i in range(n_f):
        for j in range(n_f):
            if i == j:
                corr[i,j] = 1.0; pval[i,j] = 0.0
            else:
                r, p = pcc(R_data[:, i], R_data[:, j])
                corr[i,j] = r; pval[i,j] = p


    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(n_f))
    ax.set_yticks(range(n_f))
    ax.set_xticklabels([_pcode(f) for f in R_FEATS], rotation=0, ha='center', fontsize=11)
    ax.set_yticklabels([_pcode(f) for f in R_FEATS], fontsize=11)

    for i in range(n_f):
        for j in range(n_f):
            r = corr[i, j]
            st = stars(pval[i, j])
            col = 'white' if abs(r) > 0.55 else 'black'
            ax.text(j, i, f'{r:+.2f}', ha='center', va='center',
                    color=col, fontsize=10, fontweight='bold')

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label('Pearson correlation coefficient (r)')

    ax.set_title('Pearson correlation matrix of the Stage-1 inputs and target  '
                 '(N = 312 training rows)', loc='left', fontsize=11,
                 fontweight='bold', pad=12)
    ax.grid(False)
    plt.tight_layout()
    save_fig('fig_R_pearson_heatmap.png')


    # ----------------------------------------------------------------------------
    #  FIG R2: Pearson correlation heatmap with significance (Stage 2)
    # ----------------------------------------------------------------------------
    #  Stage-2 counterpart of Fig R. Pearson correlation between the Stage-2
    #  deployment inputs (SunlightLux, F8_TimeStep, Temperature_C) and the target
    #  F13, with the same significance stars (* p≤0.05, ** p≤0.01, *** p≤0.001).
    # ----------------------------------------------------------------------------
if want('R2'):
    banner('FIG R2 — Pearson correlation heatmap (Stage 2)')

    R2_LABELS = {
        'SunlightLux':     'Sunlight intensity (lux)',
        'F8_TimeStep':     TIME_LABEL,
        'Temperature_C':   'Temperature (°C)',
        'F13_Degradation': 'DegEff (%)',
    }
    R2_FEATS = list(S2_FEATS) + ['F13_Degradation']
    R2_data  = df2[R2_FEATS].values.astype(float)

    n_f2 = len(R2_FEATS)
    corr2 = np.zeros((n_f2, n_f2)); pval2 = np.zeros((n_f2, n_f2))
    for i in range(n_f2):
        for j in range(n_f2):
            if i == j:
                corr2[i, j] = 1.0; pval2[i, j] = 0.0
            else:
                r, p = pcc(R2_data[:, i], R2_data[:, j])
                corr2[i, j] = r; pval2[i, j] = p

    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    im = ax.imshow(corr2, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(n_f2)); ax.set_yticks(range(n_f2))
    ax.set_xticklabels([{'Lux':'F9','Temp':'F10'}.get(_pcode(f), _pcode(f)) for f in R2_FEATS], rotation=0, ha='center', fontsize=12)
    ax.set_yticklabels([{'Lux':'F9','Temp':'F10'}.get(_pcode(f), _pcode(f)) for f in R2_FEATS], fontsize=12)

    for i in range(n_f2):
        for j in range(n_f2):
            r = corr2[i, j]
            st = stars(pval2[i, j])
            col = 'white' if abs(r) > 0.55 else 'black'
            ax.text(j, i, f'{r:+.2f}', ha='center', va='center',
                    color=col, fontsize=12, fontweight='bold')

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label('Pearson correlation coefficient (r)')

    ax.set_title(f'Pearson correlation matrix of the Stage-2 inputs and target  '
                 f'(N = {len(df2)} rows)', loc='left', fontsize=11,
                 fontweight='bold', pad=12)
    ax.grid(False)
    plt.tight_layout()
    save_fig('fig_R2_pearson_heatmap_stage2.png')


    # ----------------------------------------------------------------------------
    #  FIG X: SHAP dependence plots for the top Stage-1 feature pairs
    # ----------------------------------------------------------------------------
    #  Shows how the attribution for one feature changes with its value, coloured
    #  by the strongest interacting feature. Pairs: F8 (time) coloured by F11
    #  (ions), and F1 (calcination temp) coloured by F8.
    #  SHAP here is model attribution only, not a mechanistic claim.
    # ----------------------------------------------------------------------------
if want('X'):
    banner('FIG X — Stage-1 SHAP dependence plots (v4)')

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    pairs = [('F8_TimeStep', 'F11_Ions'), ('F1_CalcTemp', 'F8_TimeStep')]
    for ax, (feat, color_feat) in zip(axes, pairs):
        i  = S1_FEATS.index(feat)
        ci = S1_FEATS.index(color_feat)
        sc = ax.scatter(X1[:, i], shap_vals1[:, i], c=X1[:, ci], cmap='viridis',
                        s=26, alpha=0.75, edgecolors='white', linewidths=0.3)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
        cb.set_label(S1_LABELS[color_feat], fontsize=9)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xlabel(S1_LABELS[feat])
        ax.set_ylabel(f'SHAP value for {S1_LABELS[feat]} (%-points)')
    panel_letters = ['a', 'b']
    for ax, pl, (feat, color_feat) in zip(axes, panel_letters, pairs):
        ax.set_title(f'({pl}) {S1_LABELS[feat]}, coloured by {S1_LABELS[color_feat]}',
                     fontsize=10.5)
    fig.suptitle('Stage-1 SHAP dependence — attribution vs feature value '
                 '[model attribution only]', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_fig('fig_X_shap_dependence.png')


    # ----------------------------------------------------------------------------
    #  FIG Y: Stage-2 response surfaces at three fixed temperatures
    # ----------------------------------------------------------------------------
    #  Predicted DegEff over (lux x time) at T = 25 / 35 / 45 °C. Complements
    #  Fig E (which uses the monotonic envelope without temperature) by showing
    #  the raw Stage-2 model's temperature sensitivity directly. Temperatures
    #  chosen to span the measured 23-45 °C field range.
    # ----------------------------------------------------------------------------
if want('Y'):
    banner('FIG Y — Stage-2 (lux × time) response surfaces at fixed T (v4)')

    lux_grid  = np.linspace(20000, 135000, 60)
    time_grid = np.linspace(3, 60, 60)
    T_PANELS  = [25, 35, 45]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    last_im = None
    for ax, T_fix in zip(axes, T_PANELS):
        Z = np.zeros((len(time_grid), len(lux_grid)))
        for r, t in enumerate(time_grid):
            for c, lx in enumerate(lux_grid):
                Z[r, c] = s2_predict(lx, t, T_fix)
        last_im = ax.pcolormesh(lux_grid / 1000, time_grid, Z, cmap='viridis',
                                vmin=0, vmax=100, shading='auto')
        cs = ax.contour(lux_grid / 1000, time_grid, Z, levels=[50, 90],
                        colors='white', linewidths=1.4)
        ax.clabel(cs, fmt='%d%%', fontsize=8)
        ax.set_xlabel('Sunlight intensity (klux)')
        ax.set_title(f'T = {T_fix} °C', fontsize=11)
        ax.grid(False)
    axes[0].set_ylabel(TIME_LABEL)
    cb = fig.colorbar(last_im, ax=axes, pad=0.015, fraction=0.025)
    cb.set_label('Predicted DegEff (%)')
    fig.suptitle('Stage-2 model response — predicted DegEff over (lux × time) at '
                 'three temperatures (raw model, not envelope)',
                 fontsize=12, fontweight='bold', y=1.02)
    save_fig('fig_Y_stage2_temp_surfaces.png')


    # Final summary update
    banner('Chemistry figures (v3.1) — NEW figures P, Q, R complete')
    new_figs = ['fig_P_shap_multipanel', 'fig_Q_PCA_group', 'fig_R_pearson_heatmap']
    print(f'  {len(new_figs)} figures added:')
    for f in new_figs:
        print(f'    • {f}.png')


    # ----------------------------------------------------------------------------
    #  FIG S: Season attribution / Stage-2 deployment analysis
    # ----------------------------------------------------------------------------
    #  Four panels addressing six questions about Stage-2 behaviour:
    #  to give explicit answers to six questions about Stage-2 behaviour:
    #     Q1  Among the three seasons, which gives BEST results?
    #     Q2  Which gives POOR results?
    #     Q3  Which (season, light intensity, clock time) combination gives the
    #          HIGHEST and LOWEST RhB-degradation efficiency?
    #     Q4  Along with light intensity, does outdoor temperature play a role?
    #     Q5  How does 12-13 h efficiency compare across the three seasons?
    #     Q6  What is the desirable (lux, temperature) region from the ML model?
    #
    #  Panels:
    #     (a) Box plot of DegEff by season at t=60 min  (answers Q1, Q2)
    #     (b) Hour × Season heatmap of DegEff at t=60 min  (answers Q3, Q5)
    #     (c) DegEff vs SunlightLux at t=30 min, coloured by Temperature_C,
    #          with a temperature-conditioned model line  (answers Q4)
    #     (d) Model-derived predicted DegEff over (lux, temperature) at t=30 min,
    #          monotonic envelope, ≥90 % zone outlined  (answers Q6)
    #  plus a text panel summarising the six answers.
    # ----------------------------------------------------------------------------
if want('S'):
    banner('FIG S — Season attribution & deployment recommendations (Stage 2)')

    SEASON_DISPLAY = {
        'sunny_summer':  'Sunny Summer',
        'sunny_winter':  'Sunny Winter',
        'cloudy_winter': 'Cloudy Winter',
    }
    SEASON_COLOUR = {
        'sunny_summer':  '#F57C00',
        'sunny_winter':  '#1976D2',
        'cloudy_winter': '#7986CB',
    }
    SEASON_ORDER = ['sunny_summer', 'sunny_winter', 'cloudy_winter']

    # Restrict to the three field-campaign weather conditions
    df2_s = df2[df2['Weather'].isin(SEASON_ORDER)].copy()

    # Quantitative answers (computed first so the figure can be annotated)
    ans_t60 = (df2_s[df2_s['F8_TimeStep']==60]
               .groupby('Weather')['F13_Degradation'].mean())
    best_season  = ans_t60.idxmax()
    worst_season = ans_t60.idxmin()

    per_cell_t60 = (df2_s[df2_s['F8_TimeStep']==60]
                    .groupby(['Weather','Hour'])['F13_Degradation'].mean())
    hi_idx = per_cell_t60.idxmax()
    lo_idx = per_cell_t60.idxmin()
    hi_val = per_cell_t60.loc[hi_idx]
    lo_val = per_cell_t60.loc[lo_idx]
    n_top = int((per_cell_t60 == per_cell_t60.max()).sum())
    # Median lux at the highest- and lowest-performing cells (for the answer text)
    lux_at = (df2_s.groupby(['Weather','Hour'])['SunlightLux'].median())
    hi_lux = lux_at.loc[hi_idx]
    lo_lux = lux_at.loc[lo_idx]

    noon_t60 = (df2_s[(df2_s['F8_TimeStep']==60) & (df2_s['Hour'].isin([12,13]))]
                .groupby('Weather')['F13_Degradation'].mean())

    # Stage-2 model: scan (lux, T) at t=30 to find the desirable zone
    lux_grid = np.linspace(20000, 140000, 70)
    T_grid   = np.linspace(22, 46, 50)
    Lg, Tg   = np.meshgrid(lux_grid, T_grid)
    Zg = np.zeros_like(Lg)
    T_FIXED_T = 30
    for i in range(Lg.shape[0]):
        for j in range(Lg.shape[1]):
            Zg[i, j] = s2_predict(Lg[i, j], T_FIXED_T, Tg[i, j])
    # Monotonise across lux for each fixed temperature (rows)
    Zg_env = np.maximum.accumulate(Zg, axis=1)
    # Desirable zone = (lux, T) where envelope at t=30 reaches ≥85 %
    DESIRABLE_TH = 85
    desirable_mask = Zg_env >= DESIRABLE_TH
    if desirable_mask.any():
        # "Robust" = lux at which the envelope crosses the threshold at warm-noon
        # temperature (≈42 °C, the summer-noon condition)
        T_warm_idx = int(np.argmin(np.abs(T_grid - 42)))
        warm_row = desirable_mask[T_warm_idx, :]
        lux_des_robust = lux_grid[warm_row.argmax()] / 1000 if warm_row.any() else np.nan
        # Best (T, lux) cell from the envelope
        flat_idx = np.unravel_index(np.argmax(Zg_env), Zg_env.shape)
        best_T   = Tg[flat_idx]; best_L = Lg[flat_idx] / 1000
        best_val = Zg_env[flat_idx]
    else:
        lux_des_robust = np.nan
        best_T = np.nan; best_L = np.nan; best_val = np.nan

    # Lay out the figure
    fig = plt.figure(figsize=(16, 11))
    gs  = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.35],
                           width_ratios=[1, 1, 1], hspace=0.55, wspace=0.32)

    # Panel (a): Box plot of DegEff at t=30 by season
    # Use t=30 (mid-kinetic) because at t=60 sunny conditions saturate at 100 %
    # and the box plot has no visible spread.
    T_BOX = 30
    axA = fig.add_subplot(gs[0, 0])
    box_data = [df2_s[(df2_s['Weather']==w) & (df2_s['F8_TimeStep']==T_BOX)]['F13_Degradation'].values
                for w in SEASON_ORDER]
    bp = axA.boxplot(box_data, patch_artist=True, widths=0.55,
                     medianprops=dict(color='black', linewidth=1.5),
                     showfliers=True, flierprops=dict(marker='o', markersize=4,
                                                       markerfacecolor='black'))
    for patch, w in zip(bp['boxes'], SEASON_ORDER):
        patch.set_facecolor(SEASON_COLOUR[w])
        patch.set_alpha(0.7)
    # Strip-plot overlay so individual measurements are visible
    for i, w in enumerate(SEASON_ORDER, start=1):
        vals = df2_s[(df2_s['Weather']==w) & (df2_s['F8_TimeStep']==T_BOX)]['F13_Degradation']
        jitter = np.random.uniform(-0.15, 0.15, size=len(vals))
        axA.scatter(np.full_like(vals, i, dtype=float) + jitter, vals,
                    color=SEASON_COLOUR[w], edgecolor='black',
                    linewidth=0.4, s=22, alpha=0.5, zorder=3)
    axA.set_xticks([1, 2, 3])
    axA.set_xticklabels([SEASON_DISPLAY[w] for w in SEASON_ORDER], fontsize=9.5)
    axA.set_ylabel(f'DegEff at t = {T_BOX} min (%)')
    axA.set_ylim(0, 105)
    axA.set_title(f'(a) Per-season DegEff distribution at t = {T_BOX} min',
                  loc='left', fontsize=10.5)
    # Annotate best vs worst (the headline uses the t=60 mean, since that is
    # the practical "is the dye gone?" metric)
    axA.text(0.04, 0.96,
             f'Best at t=60:  {SEASON_DISPLAY[best_season]}  '
             f'({ans_t60[best_season]:.1f} %)\n'
             f'Worst at t=60: {SEASON_DISPLAY[worst_season]}  '
             f'({ans_t60[worst_season]:.1f} %)',
             transform=axA.transAxes, va='top', fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1',
                       edgecolor='#F57C00'))

    # Panel (b): Hour × Season heatmap at t=60
    axB = fig.add_subplot(gs[0, 1])
    heat_data = (df2_s[df2_s['F8_TimeStep']==60]
                 .groupby(['Weather', 'Hour'])['F13_Degradation'].mean()
                 .unstack())
    heat_data = heat_data.reindex(SEASON_ORDER)
    im = axB.imshow(heat_data.values, aspect='auto', cmap='RdYlGn',
                    vmin=0, vmax=100)
    axB.set_xticks(range(heat_data.shape[1]))
    axB.set_xticklabels(heat_data.columns, fontsize=9)
    axB.set_yticks(range(heat_data.shape[0]))
    axB.set_yticklabels([SEASON_DISPLAY[w] for w in heat_data.index], fontsize=9.5)
    for i in range(heat_data.shape[0]):
        for j in range(heat_data.shape[1]):
            v = heat_data.values[i, j]
            if np.isnan(v): continue
            col = 'white' if v < 50 else 'black'
            axB.text(j, i, f'{v:.0f}', ha='center', va='center',
                     color=col, fontsize=9, fontweight='bold')
    axB.set_xlabel('Clock hour (start of slot)')
    axB.set_title('(b) DegEff (%) at t = 60 min  —  hour × season',
                  loc='left', fontsize=10.5)
    cbB = fig.colorbar(im, ax=axB, pad=0.02, fraction=0.04)
    cbB.set_label('DegEff (%)', fontsize=8.5)
    axB.grid(False)

    # Mark the highest and lowest cells with a star and an "x"
    y_hi = SEASON_ORDER.index(hi_idx[0]); x_hi = list(heat_data.columns).index(hi_idx[1])
    y_lo = SEASON_ORDER.index(lo_idx[0]); x_lo = list(heat_data.columns).index(lo_idx[1])
    axB.scatter([x_hi], [y_hi], marker='*', s=180, color='#2E7D32',
                edgecolor='white', linewidth=1.5, zorder=5)
    axB.scatter([x_lo], [y_lo], marker='X', s=140, color='#37474F',
                edgecolor='white', linewidth=1.5, zorder=5)

    # Panel (c): DegEff vs SunlightLux at t=30, coloured by temperature
    axC = fig.add_subplot(gs[0, 2])
    sub_t30 = df2_s[df2_s['F8_TimeStep'] == 30]
    SHAPE_S = {'sunny_summer':'o', 'sunny_winter':'s', 'cloudy_winter':'^'}
    for w in SEASON_ORDER:
        ss = sub_t30[sub_t30['Weather']==w]
        sc = axC.scatter(ss['SunlightLux']/1000, ss['F13_Degradation'],
                         c=ss['Temperature_C'], cmap='plasma', vmin=22, vmax=46,
                         marker=SHAPE_S[w], s=78, edgecolor='black', linewidth=0.5,
                         alpha=0.92, label=SEASON_DISPLAY[w])
    # Two model lines at low- vs high-temperature reference values
    lux_smooth = np.linspace(20000, 140000, 80)
    for T_ref, lstyle, lab in [(26, ':', 'Model @ T=26°C (cold)'),
                                (42, '--', 'Model @ T=42°C (hot)')]:
        yp = [s2_predict(L, 30, T_ref) for L in lux_smooth]
        axC.plot(lux_smooth/1000, yp, lstyle, color='black', linewidth=1.5,
                 alpha=0.85, label=lab)
    axC.set_xlabel('Sunlight intensity (klux)')
    axC.set_ylabel('DegEff at t = 30 min (%)')
    axC.set_xlim(20, 140); axC.set_ylim(0, 105)
    axC.set_title('(c) Lux + temperature jointly drive DegEff  (t = 30 min)',
                  loc='left', fontsize=10.5)
    axC.legend(loc='lower right', fontsize=7.5, framealpha=0.9)
    cbC = fig.colorbar(sc, ax=axC, pad=0.02, fraction=0.04)
    cbC.set_label('Temperature (°C)', fontsize=8.5)

    # Panel (d): Model contour over (lux, temperature) at t=30
    axD = fig.add_subplot(gs[1, 0:2])
    cf = axD.contourf(Lg/1000, Tg, Zg_env,
                      levels=np.linspace(0, 100, 21), cmap='RdYlGn', vmin=0, vmax=100)
    cl = axD.contour(Lg/1000, Tg, Zg_env, levels=[70, 85, 95],
                     colors='black', linewidths=[0.8, 1.4, 1.8],
                     linestyles=['--', '-', '-'])
    axD.clabel(cl, inline=True, fontsize=8, fmt='%d%%')
    # Overlay the actual (lux, temperature) points from the dataset at t=30
    sub_t30 = df2_s[df2_s['F8_TimeStep']==30]
    for w in SEASON_ORDER:
        ss = sub_t30[sub_t30['Weather']==w]
        axD.scatter(ss['SunlightLux']/1000, ss['Temperature_C'],
                    marker=SHAPE_S[w], s=60, facecolor='none',
                    edgecolor='black', linewidth=1.2, alpha=0.9,
                    label=SEASON_DISPLAY[w])
    axD.set_xlabel('Sunlight intensity (klux)')
    axD.set_ylabel('Ambient temperature (°C)')
    axD.set_title('(d) Stage-2 model — predicted DegEff at t = 30 min  '
                  '(envelope; black contour = 85 %)',
                  loc='left', fontsize=10.5)
    axD.legend(loc='lower right', fontsize=8, framealpha=0.92)
    cbD = fig.colorbar(cf, ax=axD, pad=0.02, fraction=0.025)
    cbD.set_label('Predicted DegEff (%)', fontsize=8.5)
    axD.grid(False)

    # Answers panel
    axAns = fig.add_subplot(gs[1, 2])
    axAns.axis('off')
    answers = (
        f"Q1  Best season:\n"
        f"     {SEASON_DISPLAY[best_season]}  "
        f"({ans_t60[best_season]:.1f}% at t=60 min)\n\n"
        f"Q2  Worst season:\n"
        f"     {SEASON_DISPLAY[worst_season]}  "
        f"({ans_t60[worst_season]:.1f}% at t=60 min)\n\n"
        f"Q3  Highest condition (one of {n_top}\n"
        f"     cells tied at 100%):\n"
        f"     {SEASON_DISPLAY[hi_idx[0]]}, {hi_idx[1]}–{hi_idx[1]+1} h\n"
        f"     {hi_lux/1000:.0f} klux → {hi_val:.1f}%\n"
        f"     Lowest single condition:\n"
        f"     {SEASON_DISPLAY[lo_idx[0]]}, {lo_idx[1]}–{lo_idx[1]+1} h\n"
        f"     {lo_lux/1000:.0f} klux → {lo_val:.1f}%\n\n"
        f"Q4  Temperature contribution:\n"
        f"     SHAP {13.5:.1f}%  (lux 27.6%, F8 58.9%).\n"
        f"     Lux dominates; temp is a meaningful\n"
        f"     secondary driver.\n\n"
        f"Q5  Noon (12–13 h) DegEff (t=60 min):\n"
        f"     Sunny Summer:  {noon_t60.get('sunny_summer', np.nan):.1f}%\n"
        f"     Sunny Winter:  {noon_t60.get('sunny_winter', np.nan):.1f}%\n"
        f"     Cloudy Winter: {noon_t60.get('cloudy_winter', np.nan):.1f}%\n\n"
        f"Q6  Desirable region (model, t=30):\n"
        f"     At warm-noon T ≈ 42 °C, envelope\n"
        f"     reaches ≥ {DESIRABLE_TH}% at lux ≥\n"
        f"     ~{lux_des_robust:.0f} klux.\n"
        f"     Best model cell: {best_L:.0f} klux,\n"
        f"     T = {best_T:.0f} °C → {best_val:.0f}%."
    )
    axAns.text(0.0, 1.0, "Explicit answers", transform=axAns.transAxes,
               va='top', fontsize=11, fontweight='bold', color='#1565C0')
    axAns.text(0.0, 0.93, answers, transform=axAns.transAxes,
               va='top', fontsize=8.3, family='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#F2F6FB',
                         edgecolor='#1565C0'))

    # Caption banner across the bottom
    axCap = fig.add_subplot(gs[2, :])
    axCap.axis('off')
    cap = (
        "Stage-2 model: SunlightLux + F8_TimeStep + Temperature_C → DegEff.   "
        "Panels (a)–(b) use experimental data; (c) overlays the model at two "
        "reference temperatures; (d) is the model-derived deployment landscape "
        "(monotonic envelope across lux).  Note: cloudy-winter cells around "
        "15–16 h show the weakest performance — short lux + low temperature."
    )
    axCap.text(0.5, 0.6, cap, ha='center', va='top', fontsize=8.5,
               style='italic', color='#37474F',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#FAFAFA',
                         edgecolor='#CFD8DC'))

    fig.suptitle('Stage-2 deployment analysis — season, light, temperature',
                 fontsize=13, fontweight='bold', y=0.995)
    save_fig('fig_S_season_attribution.png')

    # Save the per-(season, hour) summary table
    season_table = (df2_s.groupby(['Weather', 'Hour'])
                    .agg(MeanDegEff_60=('F13_Degradation',
                                         lambda s: s[df2_s.loc[s.index,'F8_TimeStep']==60].mean()),
                         MedianLux=('SunlightLux','median'),
                         N=('F13_Degradation','size'))
                    .reset_index())
    season_table['Weather'] = season_table['Weather'].map(SEASON_DISPLAY)
    season_table = season_table.round({'MeanDegEff_60':1, 'MedianLux':0})
    season_table.to_csv(f'{TBLDIR}/13_season_attribution.csv', index=False)
    print(f'  Season-attribution table → {TBLDIR}/13_season_attribution.csv')


# Overall summary
banner('Chemistry figures (v3.4) — run complete')
print(f'  Figures requested this run (BUILD): {sorted(BUILD)}')
print(f'  Default manuscript set = {sorted(_USED_FIGS)}')
print(f'  (skipped this run: {sorted(_ALL_FIGS - BUILD)})')
