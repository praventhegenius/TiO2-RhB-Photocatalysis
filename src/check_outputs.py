"""
Compare the tables in outputs/tables with the reference tables in
reference/tables (the values reported in the paper).

    python src/check_outputs.py

Numeric columns are compared to 1e-6. Timing columns are ignored.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW = os.path.join(ROOT, 'outputs', 'tables')
REF = os.path.join(ROOT, 'reference', 'tables')
IGNORE = {'time_s'}

def same_json(a, b):
    for k, v in a.items():
        w = b.get(k)
        if isinstance(v, (int, float)) and isinstance(w, (int, float)):
            if abs(v - w) > 1e-6:
                return False
        elif v != w:
            return False
    return True

bad = 0
for name in sorted(os.listdir(REF)):
    ref_path, new_path = os.path.join(REF, name), os.path.join(NEW, name)
    if not os.path.exists(new_path):
        print(f'MISSING  {name}'); bad += 1; continue
    if name.endswith('.json'):
        ok = same_json(json.load(open(ref_path)), json.load(open(new_path)))
    else:
        a, b = pd.read_csv(ref_path), pd.read_csv(new_path)
        ok = list(a.columns) == list(b.columns) and a.shape == b.shape
        if ok:
            for c in a.columns:
                if c in IGNORE:
                    continue
                if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                    if not np.allclose(a[c].values, b[c].values, atol=1e-6, equal_nan=True):
                        ok = False; break
                elif not (a[c].astype(str) == b[c].astype(str)).all():
                    ok = False; break
    print(f'{"OK" if ok else "DIFFERS":8} {name}')
    bad += (not ok)

print('\nAll tables match the reference.' if not bad else f'\n{bad} table(s) differ.')
sys.exit(1 if bad else 0)
