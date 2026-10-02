"""Evaluate the original fitted trees using a version-independent data format."""
import gzip
import hashlib
import json
from pathlib import Path
import struct


class PortableForest:
    def __init__(self, root: Path):
        self.data = json.loads(gzip.decompress((root / 'loan_forest.json.gz').read_bytes()))
        if self.data['format'] != 'loan-studio-forest-v1':
            raise ValueError('Unsupported model export')
        if self.data['source_sha256'] != hashlib.sha256((root / 'loan_pipeline.pkl').read_bytes()).hexdigest():
            raise ValueError('Model export does not match the original model')
        self.classes_ = self.data['classes']

    def predict_proba(self, rows):
        results = []
        for row in rows:
            features = [float(row[field['name']] == value)
                        for field in self.data['categories'] for value in field['values']]
            # scikit-learn converts forest inputs to float32 before tree traversal.
            features.extend(struct.unpack('f', struct.pack('f', float(row[name])))[0]
                            for name in self.data['numeric_fields'])
            totals = [0.0] * len(self.classes_)
            for tree in self.data['trees']:
                node = 0
                while tree['left'][node] != -1:
                    node = (tree['left'][node] if features[tree['feature'][node]] <= tree['threshold'][node]
                            else tree['right'][node])
                for index, probability in enumerate(tree['probabilities'][node]):
                    totals[index] += probability
            results.append([value / len(self.data['trees']) for value in totals])
        return results

    def predict(self, rows):
        return [self.classes_[max(range(len(values)), key=values.__getitem__)]
                for values in self.predict_proba(rows)]
