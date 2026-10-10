"""
Stage-2 t50 analysis.

A complementary view of the Stage-2 (outdoor sunlight) data that uses a
different target. It does not replace the Stage-1 / Stage-2 models.

Target
    t50: time (min) needed to reach 50 % RhB degradation for each
    experimental condition (weather, hour). Below ~50 % the kinetics stay
    close to first order; beyond that, intermediates start competing for
    radicals. t50 therefore focuses on the well-behaved part of the curve.

Inputs
    Physical (continuous): SunlightLux, Temperature_C, Hour
    Categorical (for interpretation): Season, Weather, Diurnal slot

The categoricals are added in a second model and compared against the
physical-only model to check whether season simply acts as a proxy for
light intensity and temperature.

Outputs
    Tables : 14_t50_dataset.csv, 15_t50_model_comparison.csv
    Figures: fig_T_t50_strata.png, fig_U_t50_model.png,
             fig_V_caseAB_comparison.png, fig_W_recommendations.png
"""

import os, re, json, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import partial_dependence
import shap as shap_lib

# Shared plotting style
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project_style import (apply_style, OI, WEATHER_COLOUR, DIURNAL_COLOUR,
                           SEASON_COLOUR, WEATHERTYPE_COLOUR, SEQ,
                           TARGET_LINE, GOOD, BAD, NEUTRAL, TIME_LABEL)
apply_style()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'ANR_TiO2_RhB_dataset.xlsx')
OUT       = os.path.join(ROOT, 'outputs')
FIGDIR    = f'{OUT}/figures'
TBLDIR    = f'{OUT}/tables'
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TBLDIR, exist_ok=True)

WEATHER_DISPLAY = {
    'sunny_summer':  'Sunny Summer',
    'sunny_winter':  'Sunny Winter',
    'cloudy_winter': 'Cloudy Winter',
}

def banner(msg):
    print('\n' + '═'*78); print(f'  {msg}'); print('═'*78)

def save(name):
    plt.savefig(f'{FIGDIR}/{name}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'    saved → {FIGDIR}/{name}')

# ----------------------------------------------------------------------------
#  SECTION 1: Load Stage-2 data and attach temperature
# ----------------------------------------------------------------------------
banner('SECTION 1 — Load Sheet 2a + attach Temperature')

df = pd.read_excel(DATA_PATH, sheet_name='2(a) ANR-23-05-2026 (SunInt)', header=0).iloc[:, :5]
df.columns = ['Experiment','CatalystID','SunlightLux','F8_TimeStep','F13_Degradation']
df = df[df['CatalystID'].notna()].reset_index(drop=True)
def parse_lux(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float)): return float(x)
    return float(str(x).replace(',','').strip())
df['SunlightLux']     = df['SunlightLux'].apply(parse_lux)
df['F8_TimeStep']     = pd.to_numeric(df['F8_TimeStep'], errors='coerce')
df['F13_Degradation'] = pd.to_numeric(df['F13_Degradation'], errors='coerce').clip(lower=0)
df['Experiment']      = df['Experiment'].astype(str).str.strip().str.replace(r'\s+',' ',regex=True)
df.loc[df['Experiment'].str.startswith('(13h)',na=False),'Experiment'] = '(13h) Cloudy Winter 16-17'

def parse_hour(lab):
    m = re.search(r'(\d{1,2})-\d{1,2}\b', lab); return int(m.group(1)) if m else None
def parse_weather(lab):
    m = re.search(r'\((\d+)', lab); n = int(m.group(1)) if m else None
    return {11:'sunny_summer',12:'sunny_winter',13:'cloudy_winter'}.get(n)
df['Hour']    = df['Experiment'].apply(parse_hour)
df['Weather'] = df['Experiment'].apply(parse_weather)
df = df[df['Weather'].notna() & (df['F8_TimeStep'] >= 0)].copy()

temp_df = pd.read_excel(DATA_PATH, sheet_name='(3) Temperature', header=3).iloc[:24]
temp_df = temp_df[['Weather','Hour','Temperature_C']].dropna()
TEMP = {}
for _,r in temp_df.iterrows():
    TEMP.setdefault(r['Weather'],{})[int(r['Hour'])] = float(r['Temperature_C'])
df['Temperature_C'] = df.apply(
    lambda r: TEMP.get(r['Weather'],{}).get(r['Hour'],np.nan), axis=1)

print(f'  Sheet 2a rows after cleaning + Temperature attach: {len(df)}')
print(f'  Unique (Weather, Hour) conditions: {df.groupby(["Weather","Hour"]).ngroups}')


# ----------------------------------------------------------------------------
#  SECTION 2: Extract t25, t50, t75, t90, t100 per condition
# ----------------------------------------------------------------------------
banner('SECTION 2 — Extract t25/t50/t100 per (Weather, Hour) condition')

def extract_t(times, degs, threshold):
    """Return the time (min) at which DegEff crosses `threshold`, by linear interpolation.

    Returns (value, status) where status is one of
    'ok', 'censored', 'immediate' or 'flat'.
    """
    pairs = sorted(zip(times, degs))
    ts = np.array([p[0] for p in pairs])
    ds = np.array([p[1] for p in pairs])
    if not (ds >= threshold).any():
        return np.nan, 'censored'
    if ds[0] >= threshold:
        return 0.0, 'immediate'
    idx = np.argmax(ds >= threshold)
    t1, t0 = ts[idx], ts[idx-1]
    d1, d0 = ds[idx], ds[idx-1]
    if d1 == d0:
        return float(t1), 'flat'
    return float(t0 + (threshold - d0) * (t1 - t0) / (d1 - d0)), 'ok'

# Diurnal slots
def diurnal(h):
    if 9 <= h <= 10: return 'Morning'   # 09:00-11:00 (start hours 9, 10)
    if 11 <= h <= 13: return 'Midday'   # 11:00-14:00 (start hours 11, 12, 13)
    if 14 <= h <= 16: return 'Evening'  # 14:00-17:00 (start hours 14, 15, 16)
    return 'Other'

records = []
for (w, h), g in df.groupby(['Weather','Hour']):
    times = g['F8_TimeStep'].values
    degs  = g['F13_Degradation'].values
    t25,  s25  = extract_t(times, degs, 25)
    t50,  s50  = extract_t(times, degs, 50)
    t75,  s75  = extract_t(times, degs, 75)
    t90,  s90  = extract_t(times, degs, 90)
    t100, s100 = extract_t(times, degs, 99.5)
    records.append({
        'Weather': w, 'Season': 'Summer' if 'summer' in w else 'Winter',
        'WeatherType': 'Sunny' if 'sunny' in w else 'Cloudy',
        'Hour': h, 'Diurnal': diurnal(h),
        'SunlightLux': float(g['SunlightLux'].median()),
        'Temperature_C': float(g['Temperature_C'].median()),
        't25': t25, 's25': s25,
        't50': t50, 's50': s50,
        't75': t75, 's75': s75,
        't90': t90, 's90': s90,
        't100': t100, 's100': s100,
    })
t_df = pd.DataFrame(records).sort_values(['Weather','Hour']).reset_index(drop=True)
t_df.to_csv(f'{TBLDIR}/14_t50_dataset.csv', index=False)
print(f'  Extracted: {len(t_df)} conditions  →  {TBLDIR}/14_t50_dataset.csv')
print(f'  Reached 25%:  {t_df["t25"].notna().sum()} / {len(t_df)}')
print(f'  Reached 50%:  {t_df["t50"].notna().sum()} / {len(t_df)}  (modelling target)')
print(f'  Reached 100%: {t_df["t100"].notna().sum()} / {len(t_df)}  (heavy censoring)')


# ----------------------------------------------------------------------------
#  SECTION 3: Train t50 models, physical-only vs physical + categorical
# ----------------------------------------------------------------------------
banner('SECTION 3 — t50 model: physical-only vs +categorical')

# Drop the 2 censored conditions where t50 was never reached
t50_train = t_df.dropna(subset=['t50']).copy()
print(f'  N for modelling: {len(t50_train)} (dropped {len(t_df)-len(t50_train)} censored rows)')

# Physical features only
X_phys  = t50_train[['SunlightLux','Temperature_C','Hour']].values
# Physical + categorical (one-hot)
cat = pd.get_dummies(t50_train[['Season','WeatherType','Diurnal']], drop_first=False)
X_full = np.column_stack([X_phys, cat.values.astype(float)])
y      = t50_train['t50'].values

# Three small models (N=22, so keep them simple)
def evaluate(X, y, label):
    """Leave-one-out CV for the three small models."""
    out = []
    for name, model in [
        ('Ridge',            Pipeline([('s', StandardScaler()),
                                       ('m', RidgeCV(alphas=[0.01,0.1,1,10,100]))])),
        ('GradientBoosting', GradientBoostingRegressor(n_estimators=120, max_depth=3,
                                                       learning_rate=0.06, random_state=42)),
        ('RandomForest',     RandomForestRegressor(n_estimators=200, max_depth=6,
                                                   random_state=42, n_jobs=-1)),
    ]:
        loo = LeaveOneOut()
        y_pred = cross_val_predict(model, X, y, cv=loo, n_jobs=-1)
        r2  = r2_score(y, y_pred); mae = mean_absolute_error(y, y_pred)
        # Refit on full set
        model.fit(X, y)
        train_r2 = r2_score(y, model.predict(X))
        out.append((name, r2, mae, model, train_r2))
        print(f'    {label:18s} {name:18s}  train R²={train_r2:+.3f}  '
              f'LOOCV R²={r2:+.3f}  MAE={mae:.2f} min')
    return out

print('  --- Model A: physical features (Lux, Temp, Hour) only ---')
res_phys = evaluate(X_phys, y, 'A: physical only')
print('  --- Model B: physical + categorical (one-hot) ---')
res_full = evaluate(X_full, y, 'B: + categorical')

# Pick the best physical model for downstream use
best_phys = max(res_phys, key=lambda x: x[1])
best_name = best_phys[0]
best_model = best_phys[3]
print(f'\n  Selected primary model: {best_name}  (LOOCV R² = {best_phys[1]:.3f})')

# Save comparison table
comp = pd.DataFrame({
    'Model':       [r[0] for r in res_phys] + [r[0] for r in res_full],
    'FeatureSet':  ['physical']*3 + ['physical+categorical']*3,
    'Train_R2':    [r[4] for r in res_phys] + [r[4] for r in res_full],
    'LOOCV_R2':    [r[1] for r in res_phys] + [r[1] for r in res_full],
    'LOOCV_MAE':   [r[2] for r in res_phys] + [r[2] for r in res_full],
})
comp.to_csv(f'{TBLDIR}/15_t50_model_comparison.csv', index=False)
print(f'  Comparison → {TBLDIR}/15_t50_model_comparison.csv')

# Headline result: is season acting as a proxy for (I, T)?
delta = (max(r[1] for r in res_full) - max(r[1] for r in res_phys))
print(f'\n  Δ best-R² (categorical − physical) = {delta:+.3f}')
if delta < 0.02:
    proxy_verdict = ('Categorical variables (Season/Weather/Diurnal) add no measurable '
                     'predictive value over physical features. Season IS a proxy for (Lux, T, Hour).')
elif delta < 0.05:
    proxy_verdict = ('Categorical variables add a small predictive improvement, suggesting '
                     'they encode residual information not captured by (Lux, T, Hour) alone.')
else:
    proxy_verdict = ('Categorical variables add substantial predictive value, indicating '
                     'they carry information beyond physical features.')
print('  Verdict:', proxy_verdict)


# ----------------------------------------------------------------------------
#  SECTION 4: Predict t25 and t100 the same way
# ----------------------------------------------------------------------------
banner('SECTION 4 — Auxiliary models for t25 and t100')

# t25: all 24 conditions available
t25_train = t_df.dropna(subset=['t25']).copy()
X25 = t25_train[['SunlightLux','Temperature_C','Hour']].values
y25 = t25_train['t25'].values
m25 = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.06,
                                random_state=42)
y25_pred = cross_val_predict(m25, X25, y25, cv=LeaveOneOut(), n_jobs=-1)
r2_25 = r2_score(y25, y25_pred); mae_25 = mean_absolute_error(y25, y25_pred)
m25.fit(X25, y25)
print(f'  t25 model (GB):  N={len(t25_train)}  LOOCV R² = {r2_25:+.3f}  MAE = {mae_25:.2f} min')

# t100: only 7 uncensored conditions, too few to train on
t100_train = t_df.dropna(subset=['t100']).copy()
print(f'  t100 model: only {len(t100_train)}/{len(t_df)} conditions reach 100% within 60 min;')
print('               too censored to train a defensible model. Report empirically.')


# ----------------------------------------------------------------------------
#  SECTION 5: FIG T, t50 distributions by diurnal slot / season / weather
# ----------------------------------------------------------------------------
banner('FIG T — t50 distributions across diurnal / season / weather strata')

fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True)

# Panel (a): diurnal stratification (horizontal boxes, stacked top-down)
ax = axes[0]
order_d = ['Morning','Midday','Evening']
data_d = [t50_train[t50_train['Diurnal']==d]['t50'].values for d in order_d]
bp = ax.boxplot(data_d, patch_artist=True, widths=0.55, orientation='horizontal',
                medianprops=dict(color='black', linewidth=1.5))
for patch, d in zip(bp['boxes'], order_d):
    patch.set_facecolor(DIURNAL_COLOUR[d]); patch.set_alpha(0.6)
for i, d in enumerate(order_d, start=1):
    sub = t50_train[t50_train['Diurnal']==d]
    j = np.random.uniform(-0.18, 0.18, size=len(sub))
    for k, (_, row) in enumerate(sub.iterrows()):
        ax.scatter(row['t50'], i + j[k],
                   color=WEATHER_COLOUR[row['Weather']], edgecolor='black',
                   linewidth=0.4, s=42, alpha=0.85, zorder=3)
ax.set_yticks([1,2,3]); ax.set_yticklabels(order_d); ax.invert_yaxis()
ax.set_xlabel('Time to 50 % degradation, t\u2085\u2080 (min)')
ax.set_title('(a) Diurnal slot', loc='left', fontsize=11)

# Panel (b): season
ax = axes[1]
data_s = [t50_train[t50_train['Season']=='Summer']['t50'].values,
          t50_train[t50_train['Season']=='Winter']['t50'].values]
bp = ax.boxplot(data_s, patch_artist=True, widths=0.45, orientation='horizontal',
                medianprops=dict(color='black', linewidth=1.5))
for patch, s, col in zip(bp['boxes'], ['Summer','Winter'], [SEASON_COLOUR['Summer'], SEASON_COLOUR['Winter']]):
    patch.set_facecolor(col); patch.set_alpha(0.6)
for i, s in enumerate(['Summer','Winter'], start=1):
    sub = t50_train[t50_train['Season']==s]
    j = np.random.uniform(-0.18, 0.18, size=len(sub))
    for k, (_, row) in enumerate(sub.iterrows()):
        ax.scatter(row['t50'], i + j[k],
                   color=WEATHER_COLOUR[row['Weather']], edgecolor='black',
                   linewidth=0.4, s=42, alpha=0.85, zorder=3)
ax.set_yticks([1,2]); ax.set_yticklabels(['Summer','Winter']); ax.invert_yaxis()
ax.set_xlabel('Time to 50 % degradation, t\u2085\u2080 (min)')
ax.set_title('(b) Season', loc='left', fontsize=11)

# Panel (c): weather (sunny vs cloudy)
ax = axes[2]
data_w = [t50_train[t50_train['WeatherType']=='Sunny']['t50'].values,
          t50_train[t50_train['WeatherType']=='Cloudy']['t50'].values]
bp = ax.boxplot(data_w, patch_artist=True, widths=0.45, orientation='horizontal',
                medianprops=dict(color='black', linewidth=1.5))
for patch, s, col in zip(bp['boxes'], ['Sunny','Cloudy'], [WEATHERTYPE_COLOUR['Sunny'], WEATHERTYPE_COLOUR['Cloudy']]):
    patch.set_facecolor(col); patch.set_alpha(0.6)
for i, s in enumerate(['Sunny','Cloudy'], start=1):
    sub = t50_train[t50_train['WeatherType']==s]
    j = np.random.uniform(-0.18, 0.18, size=len(sub))
    for k, (_, row) in enumerate(sub.iterrows()):
        ax.scatter(row['t50'], i + j[k],
                   color=WEATHER_COLOUR[row['Weather']], edgecolor='black',
                   linewidth=0.4, s=42, alpha=0.85, zorder=3)
ax.set_yticks([1,2]); ax.set_yticklabels(['Sunny','Cloudy']); ax.invert_yaxis()
ax.set_xlabel('Time to 50 % degradation, t\u2085\u2080 (min)')
ax.set_title('(c) Weather', loc='left', fontsize=11)

# Shared legend (weather colour code)
legend_entries = [Patch(facecolor=WEATHER_COLOUR[w], edgecolor='black',
                        label=WEATHER_DISPLAY[w]) for w in WEATHER_DISPLAY]
fig.legend(handles=legend_entries, loc='upper center', ncol=3,
           bbox_to_anchor=(0.5, 1.05), fontsize=9, frameon=False)

fig.suptitle('Time-to-50 % degradation across diurnal, seasonal, and weather strata',
             fontsize=12.5, fontweight='bold', y=1.10)
save('fig_T_t50_strata.png')


# ----------------------------------------------------------------------------
#  SECTION 6: FIG U, t50 model parity, residuals, partial dependence
# ----------------------------------------------------------------------------
banner('FIG U — t50 model parity + partial dependence (physical features)')

# Use the best physical-feature model
y_loo = cross_val_predict(best_model, X_phys, y, cv=LeaveOneOut(), n_jobs=-1)

fig = plt.figure(figsize=(15, 9))
gs  = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.4, wspace=0.35)

# Panel (a): parity plot
axA = fig.add_subplot(gs[0, 0])
mn, mx = 0, max(y.max(), y_loo.max()) * 1.1
axA.plot([mn, mx], [mn, mx], 'k--', linewidth=1, alpha=0.5, label='1:1')
for w in WEATHER_DISPLAY:
    mask = (t50_train['Weather'].values == w)
    axA.scatter(y[mask], y_loo[mask],
                color=WEATHER_COLOUR[w], edgecolor='black', linewidth=0.5,
                s=70, alpha=0.85, label=WEATHER_DISPLAY[w])
axA.set_xlabel('Experimental t\u2085\u2080 (min)')
axA.set_ylabel('Model-predicted t\u2085\u2080 (min, LOOCV)')
axA.set_xlim(mn, mx); axA.set_ylim(mn, mx)
axA.set_title(f'(a) LOOCV parity — {best_name}\n'
              f'R\u00B2 = {best_phys[1]:.3f}, MAE = {best_phys[2]:.2f} min',
              loc='left', fontsize=10.5)
axA.legend(loc='lower right', fontsize=8)

# Panel (b): residuals vs lux
axB = fig.add_subplot(gs[0, 1])
resid = y - y_loo
for w in WEATHER_DISPLAY:
    mask = (t50_train['Weather'].values == w)
    axB.scatter(t50_train['SunlightLux'][mask]/1000, resid[mask],
                color=WEATHER_COLOUR[w], edgecolor='black', linewidth=0.5,
                s=70, alpha=0.85, label=WEATHER_DISPLAY[w])
axB.axhline(0, ls='--', color='gray', linewidth=0.8)
axB.set_xlabel('Sunlight intensity (klux)')
axB.set_ylabel('Residual (experimental − predicted, min)')
axB.set_title('(b) Residuals vs lux', loc='left', fontsize=10.5)

# Panel (c): residuals vs temperature
axC = fig.add_subplot(gs[0, 2])
for w in WEATHER_DISPLAY:
    mask = (t50_train['Weather'].values == w)
    axC.scatter(t50_train['Temperature_C'][mask], resid[mask],
                color=WEATHER_COLOUR[w], edgecolor='black', linewidth=0.5,
                s=70, alpha=0.85, label=WEATHER_DISPLAY[w])
axC.axhline(0, ls='--', color='gray', linewidth=0.8)
axC.set_xlabel('Temperature (°C)')
axC.set_ylabel('Residual (min)')
axC.set_title('(c) Residuals vs temperature', loc='left', fontsize=10.5)

# Partial dependence on the three physical features (model B uses the full set)
feature_grids = {
    0: ('Sunlight intensity (klux)', np.linspace(20000, 140000, 60)/1000, 0, 1000),
    1: ('Temperature (°C)',          np.linspace(22, 46, 60),               1, 1),
    2: ('Hour (clock)',              np.arange(9, 17),                       2, 1),
}
for idx, (lab, _grid_hint, fidx, _mult) in feature_grids.items():
    ax = fig.add_subplot(gs[1, idx])
    pdp = partial_dependence(best_model, X_phys, [fidx],
                              grid_resolution=40, kind='average')
    xs = pdp['grid_values'][0]
    ys = pdp['average'][0]
    # Convert lux to klux for the display axis if this is the lux feature
    if fidx == 0:
        xs_disp = xs / 1000.0
        xlab = lab
    else:
        xs_disp = xs
        xlab = lab
    ax.plot(xs_disp, ys, '-', color='#1565C0', linewidth=2.2)
    ax.fill_between(xs_disp, ys, alpha=0.18, color='#1565C0')
    ax.set_xlabel(xlab); ax.set_ylabel('Partial dep. on t\u2085\u2080 (min)')
    ax.set_title(f'(d-{idx+1}) PDP — {lab.split(" (")[0]}', loc='left', fontsize=10.5)

fig.suptitle(f't\u2085\u2080 model (physical features only): performance and partial dependence  '
             f'\u2014 selected: {best_name}',
             fontsize=12.5, fontweight='bold', y=1.00)
save('fig_U_t50_model.png')


# ----------------------------------------------------------------------------
#  SECTION 7: FIG V, Case A vs Case B (does T add anything beyond I?)
# ----------------------------------------------------------------------------
banner("FIG V — Case A (high-lux warm summer) vs Case B (moderate-lux cool winter) decomposition")

# Find the observed conditions closest to the two reference cases
def nearest_row(target_lux, target_T):
    d = ((t50_train['SunlightLux'] - target_lux)**2 / 1e10 +
         (t50_train['Temperature_C'] - target_T)**2)
    return t50_train.loc[d.idxmin()]

caseA = nearest_row(138000, 35)
caseB = nearest_row(98000, 25)
print(f'  Case A nearest: {WEATHER_DISPLAY[caseA["Weather"]]} {caseA["Hour"]}h, '
      f'{caseA["SunlightLux"]/1000:.0f} klux, {caseA["Temperature_C"]:.0f}°C → t50 = {caseA["t50"]:.2f} min')
print(f'  Case B nearest: {WEATHER_DISPLAY[caseB["Weather"]]} {caseB["Hour"]}h, '
      f'{caseB["SunlightLux"]/1000:.0f} klux, {caseB["Temperature_C"]:.0f}°C → t50 = {caseB["t50"]:.2f} min')

# Counterfactual decomposition with the trained model
def predict_t50(lux, T, hour):
    return float(best_model.predict([[lux, T, hour]])[0])

# Baseline = Case A actual
predA  = predict_t50(caseA['SunlightLux'], caseA['Temperature_C'], caseA['Hour'])
predB  = predict_t50(caseB['SunlightLux'], caseB['Temperature_C'], caseB['Hour'])
# Counterfactual: Case A with Case B's lux only (A's temperature and hour kept)
predA_BLux  = predict_t50(caseB['SunlightLux'], caseA['Temperature_C'], caseA['Hour'])
# Counterfactual: Case A with Case B's temperature only
predA_BTemp = predict_t50(caseA['SunlightLux'], caseB['Temperature_C'], caseA['Hour'])
# Combined (lux and temp both flipped)
predA_BBoth = predict_t50(caseB['SunlightLux'], caseB['Temperature_C'], caseA['Hour'])

print(f'\n  Model decomposition of t50 change Case A → Case B:')
print(f'    Case A baseline:                       {predA:.2f} min')
print(f'    + swap to Case B lux only:             {predA_BLux:.2f} min  (Δ = {predA_BLux-predA:+.2f})')
print(f'    + swap to Case B temperature only:     {predA_BTemp:.2f} min  (Δ = {predA_BTemp-predA:+.2f})')
print(f'    + swap to Case B lux AND temperature:  {predA_BBoth:.2f} min  (Δ = {predA_BBoth-predA:+.2f})')
print(f'    Case B actual model prediction:        {predB:.2f} min')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): kinetic curves for the two cases
axL = axes[0]
for case, lbl, col in [(caseA, 'Case A (Sunny Summer, 9h)', WEATHER_COLOUR['sunny_summer']),
                       (caseB, 'Case B (Sunny Winter, 9h)', WEATHER_COLOUR['sunny_winter'])]:
    sub = df[(df['Weather']==case['Weather']) & (df['Hour']==case['Hour'])].sort_values('F8_TimeStep')
    axL.plot(sub['F8_TimeStep'], sub['F13_Degradation'], 'o-', color=col,
             markersize=8, linewidth=2,
             label=f'{lbl}\n{case["SunlightLux"]/1000:.0f} klux, {case["Temperature_C"]:.0f}°C')
    # Mark t50 with a star
    axL.scatter([case['t50']], [50], marker='*', s=300, color=col,
                edgecolor='black', linewidth=1.2, zorder=5)
axL.axhline(50, ls=':', color='gray', linewidth=1)
axL.set_xlabel(TIME_LABEL)
axL.set_ylabel('Degradation efficiency (%)')
axL.set_xlim(0, 62); axL.set_ylim(-3, 105)
axL.legend(loc='lower right', fontsize=9)
axL.set_title('(a) Experimental kinetic curves; star = t\u2085\u2080',
              loc='left', fontsize=11)

# Panel (b): counterfactual decomposition (waterfall)
axR = axes[1]
labels = ['Case A\nbaseline',
          'After\nlux swap\nonly',
          'After\ntemperature\nswap only',
          'After\nboth swapped\n(= Case B)']
values = [predA, predA_BLux, predA_BTemp, predA_BBoth]
colors_v = [GOOD, OI['skyblue'], OI['orange'], BAD]
bars = axR.bar(range(4), values, color=colors_v, edgecolor='black',
               linewidth=0.5, width=0.6)
for b, v in zip(bars, values):
    axR.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.1f}\nmin',
             ha='center', fontsize=9, fontweight='bold')
# Connector lines showing the change
for i in range(len(values)-1):
    axR.annotate('', xy=(i+1, values[i+1]), xytext=(i, values[i]),
                 arrowprops=dict(arrowstyle='->', linestyle='--', color='#555', lw=1.2))
axR.set_xticks(range(4))
axR.set_xticklabels(labels, fontsize=9)
axR.set_ylabel('Model-predicted t\u2085\u2080 (min)')
axR.set_ylim(0, max(values)*1.25)
axR.set_title('(b) Counterfactual decomposition: which variable\n'
              'drives the difference between Case A and Case B?',
              loc='left', fontsize=11)
# Annotation: which effect dominates
lux_share  = abs(predA_BLux  - predA) / max(abs(predA_BBoth - predA), 1e-9) * 100
temp_share = abs(predA_BTemp - predA) / max(abs(predA_BBoth - predA), 1e-9) * 100
ann = (f'Single-variable shares of the\n'
       f'A→B change in model t\u2085\u2080:\n'
       f'  Lux contribution:  {lux_share:.0f}%\n'
       f'  Temp contribution: {temp_share:.0f}%')
axR.text(0.04, 0.96, ann, transform=axR.transAxes, va='top', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1', edgecolor=OI['orange']))

fig.suptitle('Decomposing the Case A → Case B difference: model attributes the change to lux + temperature',
             fontsize=12.5, fontweight='bold', y=1.02)
save('fig_V_caseAB_comparison.png')


# ----------------------------------------------------------------------------
#  SECTION 8: FIG W, recommended conditions for 25/50/100%
# ----------------------------------------------------------------------------
banner('FIG W — Lowest-time recommendations for 25 / 50 / 100 % degradation')

# For each threshold, rank conditions and keep the 5 fastest
def topk(col, k=5):
    return (t_df.dropna(subset=[col])
                .sort_values(col)
                .head(k)
                .reset_index(drop=True))

top25  = topk('t25', 5)
top50  = topk('t50', 5)
top100 = topk('t100', 5)

print('\n  Top 5 conditions for ≥25 %:')
print(top25[['Weather','Hour','SunlightLux','Temperature_C','t25']].to_string(index=False))
print('\n  Top 5 conditions for ≥50 %:')
print(top50[['Weather','Hour','SunlightLux','Temperature_C','t50']].to_string(index=False))
print('\n  Top 5 conditions for ≥100 %:')
print(top100[['Weather','Hour','SunlightLux','Temperature_C','t100']].to_string(index=False))

# Build a 4-panel figure: (a) t25, (b) t50, (c) t100 ranked bars
#                        (d) summary recommendation block
fig = plt.figure(figsize=(15, 9))
gs  = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.5, wspace=0.4)

def draw_rank(ax, top_df, col, title):
    labels = [f'{WEATHER_DISPLAY[r["Weather"]].replace(" ", chr(10))}\n{int(r["Hour"])}–{int(r["Hour"])+1} h\n'
              f'{r["SunlightLux"]/1000:.0f} klx\n{r["Temperature_C"]:.0f}°C'
              for _, r in top_df.iterrows()]
    vals = top_df[col].values
    cols = [WEATHER_COLOUR[r['Weather']] for _, r in top_df.iterrows()]
    bars = ax.bar(range(len(top_df)), vals, color=cols, edgecolor='black',
                  linewidth=0.5, width=0.65)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.1f} min',
                ha='center', fontsize=8.5, fontweight='bold')
    ax.set_xticks(range(len(top_df)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel('Time (min)')
    ax.set_title(title, loc='left', fontsize=10.5)
    ax.set_ylim(0, vals.max() * 1.25)

draw_rank(fig.add_subplot(gs[0,0]), top25,  't25',
          '(a) Lowest 5 times to reach 25 % degradation')
draw_rank(fig.add_subplot(gs[0,1]), top50,  't50',
          '(b) Lowest 5 times to reach 50 % degradation')
draw_rank(fig.add_subplot(gs[0,2]), top100, 't100',
          '(c) Lowest 5 times to reach 100 %  (only 7 of 24 conditions reach it within 60 min)')

# Recommendation panel
axR = fig.add_subplot(gs[1, :])
axR.axis('off')

best25  = top25.iloc[0]; best50 = top50.iloc[0]; best100 = top100.iloc[0]

rec_text = (
    'Best operating windows (from experimental data, ANR-650-8-4):\n'
    '\n'
    f'  • Reach 25 % fastest:   {WEATHER_DISPLAY[best25["Weather"]]},  '
    f'{int(best25["Hour"])}–{int(best25["Hour"])+1} h slot  '
    f'({best25["SunlightLux"]/1000:.0f} klux, {best25["Temperature_C"]:.0f} °C)  →  '
    f't\u2082\u2085 = {best25["t25"]:.1f} min\n'
    f'  • Reach 50 % fastest:   {WEATHER_DISPLAY[best50["Weather"]]},  '
    f'{int(best50["Hour"])}–{int(best50["Hour"])+1} h slot  '
    f'({best50["SunlightLux"]/1000:.0f} klux, {best50["Temperature_C"]:.0f} °C)  →  '
    f't\u2085\u2080 = {best50["t50"]:.1f} min\n'
    f'  • Reach 100 % fastest:  {WEATHER_DISPLAY[best100["Weather"]]},  '
    f'{int(best100["Hour"])}–{int(best100["Hour"])+1} h slot  '
    f'({best100["SunlightLux"]/1000:.0f} klux, {best100["Temperature_C"]:.0f} °C)  →  '
    f't\u2081\u2080\u2080 = {best100["t100"]:.1f} min\n'
    '\n'
    'How close are these recommendations to what the model predicts?\n'
)
# Compare empirical t50 with model-predicted t50 for the top-5 conditions
emp = top50['t50'].values
mod = np.array([predict_t50(r['SunlightLux'], r['Temperature_C'], r['Hour'])
                for _, r in top50.iterrows()])
gaps = mod - emp
rec_text += '\n  Top-5 t\u2085\u2080  (experiment | model | gap):\n'
for i in range(len(top50)):
    rec_text += (f'    {i+1}. {WEATHER_DISPLAY[top50.iloc[i]["Weather"]]:>14s}  '
                 f'{int(top50.iloc[i]["Hour"])}h: '
                 f'{emp[i]:5.2f} | {mod[i]:5.2f} | {gaps[i]:+5.2f} min\n')
rec_text += (f'\n  Mean |gap| over top 5: {np.abs(gaps).mean():.2f} min   '
             f'(Model LOOCV MAE = {best_phys[2]:.2f} min over all 22 conditions)')

axR.text(0.0, 1.0, rec_text, transform=axR.transAxes, va='top',
         fontsize=10, family='monospace',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#F2F6FB', edgecolor='#1565C0'))

fig.suptitle('Question (1): recommended (season, weather, slot, lux, temperature) '
             'for fastest 25 / 50 / 100 % RhB degradation',
             fontsize=12.5, fontweight='bold', y=0.99)
save('fig_W_recommendations.png')


# ----------------------------------------------------------------------------
#  Final summary
# ----------------------------------------------------------------------------
banner('Stage-2 t50 work-plan analysis — complete')
print('  Tables:')
print(f'    • {TBLDIR}/14_t50_dataset.csv      (24 conditions, multi-threshold)')
print(f'    • {TBLDIR}/15_t50_model_comparison.csv (physical vs +categorical)')
print('  Figures:')
for f in ['fig_T_t50_strata', 'fig_U_t50_model',
          'fig_V_caseAB_comparison', 'fig_W_recommendations']:
    print(f'    • {FIGDIR}/{f}.png')
print(f'\n  Headline metrics:')
print(f'    Best physical-only model: {best_name},  LOOCV R² = {best_phys[1]:.3f},  '
      f'MAE = {best_phys[2]:.2f} min on N=22 conditions')
print(f'    Δ R² (categorical − physical) = {delta:+.3f}  →  {proxy_verdict}')
