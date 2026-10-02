"""Export the fitted forest without retraining it. Run with requirements-export.txt."""
import gzip
import hashlib
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'loan_pipeline.pkl'
model = joblib.load(source)
preprocessor = model.named_steps['preprocessor']
forest = model.named_steps['classifier']
encoder = preprocessor.named_transformers_['cat']
assert isinstance(forest, RandomForestClassifier)
assert isinstance(encoder, OneHotEncoder)
assert encoder.drop_idx_ is None and encoder.handle_unknown == 'ignore'
assert preprocessor.named_transformers_['num'].func is None
assert forest.n_outputs_ == 1

payload = {
    'format': 'loan-studio-forest-v1',
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'classes': forest.classes_.tolist(),
    'categories': [
        {'name': name, 'values': values.tolist()}
        for name, values in zip(preprocessor.transformers_[0][2], encoder.categories_)
    ],
    'numeric_fields': preprocessor.transformers_[1][2],
    'trees': [{
        'left': estimator.tree_.children_left.tolist(),
        'right': estimator.tree_.children_right.tolist(),
        'feature': estimator.tree_.feature.tolist(),
        'threshold': estimator.tree_.threshold.tolist(),
        'probabilities': estimator.tree_.value[:, 0, :].tolist(),
    } for estimator in forest.estimators_],
}
encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False).encode()
(ROOT / 'loan_forest.json.gz').write_bytes(gzip.compress(encoded, mtime=0))
print(f'Exported {len(payload["trees"])} original trees; no fitting or training performed.')
