"""
Run the full analysis in order and write everything to outputs/.

    python src/run_all.py            # all four steps
    python src/run_all.py --skip-lc  # skip the slow learning-curve step

STRIP_TITLES=1 removes descriptive titles from the figures, as in the paper.
LC_SKIP=1 stops step 01 from drawing its own quick learning curve; the
learning curve in the paper is built by step 02.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ['01_linked_ml_pipeline.py', '02_learning_curves.py',
         '03_chemistry_figures.py', '04_stage2_t50_analysis.py']

env = dict(os.environ, STRIP_TITLES='1', LC_SKIP='1', MPLBACKEND='Agg')
for step in STEPS:
    if step.startswith('02') and '--skip-lc' in sys.argv:
        print(f'skipping {step}')
        continue
    t0 = time.time()
    print(f'\n=== {step} ===', flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, step)], env=env, check=True)
    print(f'=== {step} done in {time.time() - t0:.0f} s ===', flush=True)
