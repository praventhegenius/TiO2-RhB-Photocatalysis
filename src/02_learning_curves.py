"""
Learning curves for the stacked ensemble (Fig. 2g).

Loads the Stage 1 and Stage 2 feature matrices saved by
01_linked_ml_pipeline.py (outputs/models/lc_data.pkl) and computes the
cross-validated R2 of the stacked ensemble at increasing training-set sizes.
The ensemble uses the same eight base models and a Ridge meta-learner as in
01_linked_ml_pipeline.py.

Run 01_linked_ml_pipeline.py first.
"""
import os, pickle, warnings, numpy as np
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import xgboost as xgb, lightgbm as lgb
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
    GradientBoostingRegressor, StackingRegressor)
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.compose import TransformedTargetRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from project_style import apply_style, STAGE1, STAGE2
apply_style()
SEED = 42

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
with open(f'{OUT}/models/lc_data.pkl', 'rb') as f:
    D = pickle.load(f)
X1, y1, X2, y2 = D['X1'], D['y1'], D['X2'], D['y2']


def stack(stage):
    # same base families + Ridge meta as the deployed stacked ensemble;
    # GP restarts and internal cv reduced for the learning-curve sweep only.
    d = [
        ('XGBoost', xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
            min_child_weight=2, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1,
            reg_lambda=1.0, random_state=SEED, n_jobs=1, verbosity=0)),
        ('LightGBM', lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
            min_child_samples=5, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1,
            reg_lambda=1.0, random_state=SEED, n_jobs=1, verbosity=-1)),
        ('ExtraTrees', ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1,
            max_features='sqrt', random_state=SEED, n_jobs=1)),
        ('RandomForest', RandomForestRegressor(n_estimators=500, min_samples_leaf=1,
            max_features='sqrt', random_state=SEED, n_jobs=1)),
        ('GradientBoost', GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
            max_depth=5, min_samples_leaf=2, subsample=0.9, random_state=SEED)),
        ('SVR_RBF', Pipeline([('scl', StandardScaler()),
            ('svr', SVR(C=100, epsilon=1.5, gamma='scale'))])),
        ('GaussianProcess', TransformedTargetRegressor(
            regressor=GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * Matern(nu=2.5, length_scale=1.0,
                    length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(0.1, (1e-6, 1e1)),
                n_restarts_optimizer=3, alpha=1e-6, random_state=SEED),
            transformer=StandardScaler())),
        ('KNN', KNeighborsRegressor(n_neighbors=5, weights='distance')),
    ]
    return StackingRegressor(estimators=d, final_estimator=Ridge(), cv=3, n_jobs=-1)


fractions = [0.30, 0.45, 0.60, 0.80, 1.00]
rng = np.random.RandomState(SEED)


def lc_points(X, y, stage):
    out = []
    for frac in fractions:
        n_sub = max(30, int(len(X) * frac))
        scores = []
        for rep in range(1):
            idx = rng.choice(len(X), size=n_sub, replace=False)
            Xs, ys = X[idx], y[idx]
            for tr, te in KFold(4, shuffle=True, random_state=SEED + rep).split(Xs):
                m = stack(stage); m.fit(Xs[tr], ys[tr])
                scores.append(r2_score(ys[te], np.clip(m.predict(Xs[te]), 0, 100)))
        out.append((n_sub, float(np.mean(scores)), float(np.std(scores))))
        print(f'  stage{stage} N={n_sub:>3}  CV R2 = {np.mean(scores):.3f} ± {np.std(scores):.3f}')
    return out


print('Stage 1 learning curve …')
lc1 = lc_points(X1, y1, 1)
print('Stage 2 learning curve …')
lc2 = lc_points(X2, y2, 2)

fig, ax = plt.subplots(figsize=(9, 6))
for lc, col, label in [(lc1, STAGE1, 'Stage 1 (N=312 available)'),
                       (lc2, STAGE2, 'Stage 2 (N=192 available)')]:
    ns = [p[0] for p in lc]; mus = [p[1] for p in lc]; sds = [p[2] for p in lc]
    ax.plot(ns, mus, 'o-', color=col, linewidth=2, markersize=7,
            markeredgecolor='black', markeredgewidth=0.6, label=label)
    ax.fill_between(ns, [m - s for m, s in zip(mus, sds)],
                    [m + s for m, s in zip(mus, sds)], color=col, alpha=0.15)
ax.set_xlabel('Training rows used')
ax.set_ylabel('CV R² (Stacked ensemble, mean ± sd over folds×repeats)', fontsize=10.5)
ax.set_title('Learning curves — CV R² vs available training data')
ax.legend(loc='lower right')
ax.set_ylim(0.5, 1.02)
plt.tight_layout()
out = f'{OUT}/figures/fig_09_learning_curves.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print('saved →', out)
