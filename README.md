# Loan Studio

A dark scenario workspace for the existing [loan prediction project](https://github.com/alumond/loan-Prediction-App). Enter invented details, run the saved model and compare recent results.

The interface includes a sample scenario, grouped inputs, a results panel and up to six recent scenarios per session. Selecting a recent scenario restores its inputs and result. Changing the form does not change the previous result until you run it again.

## Run locally

Use Python 3.11 or later with the pinned Streamlit version. The model evaluator has been checked on Python 3.11 and 3.14.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Checks

```sh
python -m unittest discover -s tests -v
```

The checks cover the unchanged model, approval and decline examples, probability labels, input validation, resetting the form and restoring recent scenarios. The responsive interface was also checked in a separate browser at desktop and phone widths.

## Model limits

The model file is unchanged from the original repository. Its fitted trees and category mappings are also exported to `loan_forest.json.gz`, which the app evaluates without loading a version-specific scikit-learn pickle. No training or fitting is performed during export. The displayed percentage is its approval estimate, not a measure of accuracy or a lender's decision. The repository does not document currency, amount units or the credit-history code's criteria, so the interface avoids inventing those definitions.

Use made-up details. The app does not submit loan applications or save scenarios to a database. This interface does not establish the model's accuracy, fairness or suitability for lending.

## Streamlit deployment

Deploy `app.py` from the `main` branch using `requirements.txt`. The exported evaluator supports the existing Python 3.14 deployment, so the app does not need to be deleted or recreated to change Python.

The app's public address is [loan-prediction-app-55ycwzbcqsxtfryemepphw.streamlit.app](https://loan-prediction-app-55ycwzbcqsxtfryemepphw.streamlit.app/).

## Reproduce the model export

In a separate Python 3.11 environment, install `requirements-export.txt`, then run `python tools/export_model.py` and `python tools/validate_export.py`. Validation compares both classes and exact probabilities against the original pipeline for 5,000 varied examples and values at and around the forest's numeric split boundaries.
