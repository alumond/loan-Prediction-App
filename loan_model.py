"""The original model, with an explicit interface for the scenario workspace."""
from pathlib import Path
from portable_forest import PortableForest

ROOT = Path(__file__).resolve().parent
FEATURES = ('Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
            'Credit_History', 'Property_Area')
EXAMPLE = dict(Gender='Male', Married='Yes', Dependents='0', Education='Graduate',
               Self_Employed='No', ApplicantIncome=5000, CoapplicantIncome=1500,
               LoanAmount=120.0, Loan_Amount_Term=360.0, Credit_History=1.0,
               Property_Area='Urban')


def load_model():
    return PortableForest(ROOT)


def validate_scenario(values):
    if any(values.get(field) is None for field in FEATURES):
        return 'Complete each field, or load the sample scenario to get started.'
    if values['ApplicantIncome'] + values['CoapplicantIncome'] <= 0:
        return 'Add an applicant or co-applicant income greater than zero.'
    if values['LoanAmount'] <= 0:
        return 'Enter a loan amount greater than zero.'
    if values['Loan_Amount_Term'] <= 0:
        return 'Enter a loan term greater than zero.'
    return None


def predict_scenario(model, values):
    frame = [{field: values[field] for field in FEATURES}]
    prediction = model.predict(frame)[0]
    approval_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(frame)[0][approval_index])
    return {'approved': bool(prediction == 1), 'approval_probability': probability}
