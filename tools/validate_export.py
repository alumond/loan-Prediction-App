"""Compare the exported evaluator against the untouched scikit-learn pipeline."""
import json
from pathlib import Path
import random
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loan_model import EXAMPLE, load_model

original = joblib.load(ROOT / 'loan_pipeline.pkl')
exported = load_model()
rng = random.Random(42)
rows = []
for _ in range(5000):
    row = {field['name']: rng.choice(field['values']) for field in exported.data['categories']}
    row.update(ApplicantIncome=rng.randint(0, 100000), CoapplicantIncome=rng.randint(0, 50000),
               LoanAmount=rng.uniform(1, 1500), Loan_Amount_Term=rng.choice([12, 60, 120, 180, 240, 360, 480]),
               Credit_History=rng.choice([0.0, 1.0]))
    rows.append(row)

# Exercise numeric boundaries on either side of the original tree thresholds.
cat_count = sum(len(field['values']) for field in exported.data['categories'])
boundaries = {(tree['feature'][i] - cat_count, tree['threshold'][i])
              for tree in exported.data['trees'] for i in range(len(tree['feature']))
              if tree['feature'][i] >= cat_count}
for feature, threshold in sorted(boundaries):
    field = exported.data['numeric_fields'][feature]
    for value in (np.nextafter(np.float32(threshold), np.float32(-np.inf)),
                  np.float32(threshold), np.nextafter(np.float32(threshold), np.float32(np.inf))):
        rows.append({**EXAMPLE, field: float(value)})

expected = original.predict_proba(pd.DataFrame(rows))
actual = np.asarray(exported.predict_proba(rows))
assert np.array_equal(expected, actual), float(np.abs(expected - actual).max())
assert np.array_equal(original.predict(pd.DataFrame(rows)), exported.predict(rows))
print(json.dumps({'cases': len(rows), 'identical_probabilities': True,
                  'identical_predictions': True, 'max_probability_difference': float(np.abs(expected - actual).max())}))
