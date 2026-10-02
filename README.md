# Loan Studio

A dark scenario workspace for the existing [loan prediction project](https://github.com/alumond/loan-Prediction-App). Enter invented details, run the saved model and compare recent results.

The interface includes a sample scenario, grouped inputs, a results panel and up to six recent scenarios per session. Selecting a recent scenario restores its inputs and result. Changing the form does not change the previous result until you run it again.

## Run locally

Use Python 3.11 with the pinned dependencies. These versions load the existing model without retraining it.

```sh
python3.11 -m venv .venv
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

The model file is unchanged from the original repository. The displayed percentage is its approval estimate, not a measure of accuracy or a lender's decision. The repository does not document currency, amount units or the credit-history code's criteria, so the interface avoids inventing those definitions.

Use made-up details. The app does not submit loan applications or save scenarios to a database. This interface does not establish the model's accuracy, fairness or suitability for lending.

## Streamlit deployment

Deploy `app.py` from the `main` branch using **Python 3.11** in Streamlit Community Cloud's Advanced settings. Install the pinned `requirements.txt`; the model was saved with scikit-learn 1.6.1. Changing an existing deployment's Python version requires recreating that deployment through Streamlit's dashboard.

The app's public address is [loan-prediction-app-55ycwzbcqsxtfryemepphw.streamlit.app](https://loan-prediction-app-55ycwzbcqsxtfryemepphw.streamlit.app/).
