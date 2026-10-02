"""Behavior checks for the original model and the new scenario workspace."""
from pathlib import Path
import hashlib
import unittest

from streamlit.testing.v1 import AppTest
from loan_model import EXAMPLE, load_model, predict_scenario, validate_scenario

ROOT = Path(__file__).resolve().parents[1]
DECLINED_EXAMPLE = dict(
    Gender='Female', Married='No', Dependents='1', Education='Graduate',
    Self_Employed='No', ApplicantIncome=2500, CoapplicantIncome=0,
    LoanAmount=180.0, Loan_Amount_Term=360.0, Credit_History=0.0,
    Property_Area='Rural',
)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def test_original_model_and_reference_predictions_are_preserved(self):
        self.assertEqual(
            hashlib.sha256((ROOT / 'loan_pipeline.pkl').read_bytes()).hexdigest(),
            'e8798fa871d41a99004373ad9aa402c62a8be69be2b895b955fe613248d2e07c',
        )
        for values, approved, probability in (
            (EXAMPLE, True, .94), (DECLINED_EXAMPLE, False, .17),
        ):
            result = predict_scenario(self.model, values)
            self.assertEqual(result['approved'], approved)
            self.assertAlmostEqual(result['approval_probability'], probability)

    def test_probability_follows_approval_class_not_column_order(self):
        class ReversedClasses:
            classes_ = [1, 0]

            def predict(self, frame):
                return [0]

            def predict_proba(self, frame):
                return [[.2, .8]]

        result = predict_scenario(ReversedClasses(), EXAMPLE)
        self.assertFalse(result['approved'])
        self.assertAlmostEqual(result['approval_probability'], .2)

    def test_invalid_scenarios_are_rejected_before_prediction(self):
        for values in (
            {}, {**EXAMPLE, 'Credit_History': None},
            {**EXAMPLE, 'ApplicantIncome': 0, 'CoapplicantIncome': 0},
            {**EXAMPLE, 'LoanAmount': 0}, {**EXAMPLE, 'Loan_Amount_Term': 0},
        ):
            self.assertIsNotNone(validate_scenario(values))
        self.assertIsNone(validate_scenario(EXAMPLE))


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=20).run()
        self.assertEqual(len(self.app.exception), 0)

    def submit(self):
        next(button for button in self.app.button if button.label == 'Run scenario').click().run()
        self.assertEqual(len(self.app.exception), 0)

    def test_reset_sample_and_history_restore(self):
        self.submit()
        self.assertAlmostEqual(self.app.session_state['result']['approval_probability'], .94)
        self.app.button(key='new_scenario').click().run()
        self.assertIsNone(self.app.session_state['result'])
        self.assertIsNone(self.app.number_input(key='input_LoanAmount').value)
        self.submit()
        self.assertIn('Complete each field', self.app.error[0].value)
        self.assertEqual(len(self.app.session_state['history']), 1)
        self.app.button(key='load_example').click().run()
        self.assertEqual(self.app.number_input(key='input_LoanAmount').value, 120)
        self.assertEqual(len(self.app.error), 0)
        self.app.number_input(key='input_LoanAmount').set_value(180).run()
        self.submit()
        self.assertEqual(len(self.app.session_state['history']), 2)
        self.app.button(key='history_1').click().run()
        self.assertEqual(self.app.number_input(key='input_LoanAmount').value, 120)
        self.assertEqual(self.app.session_state['result']['number'], 1)
        self.assertAlmostEqual(self.app.session_state['result']['approval_probability'], .94)

    def test_decline_and_prior_result_until_resubmission(self):
        for widget in self.app.number_input:
            field = widget.key.removeprefix('input_')
            widget.set_value(DECLINED_EXAMPLE[field])
        for widget in self.app.selectbox:
            field = widget.key.removeprefix('input_')
            widget.set_value(DECLINED_EXAMPLE[field])
        self.submit()
        self.assertFalse(self.app.session_state['result']['approved'])
        self.assertAlmostEqual(self.app.session_state['result']['approval_probability'], .17)
        self.assertTrue(any('Decline predicted' in item.value for item in self.app.markdown))
        self.app.number_input(key='input_LoanAmount').set_value(100).run()
        self.assertEqual(self.app.session_state['result']['inputs']['LoanAmount'], 180)
        self.submit()
        self.assertEqual(self.app.session_state['result']['inputs']['LoanAmount'], 100)


if __name__ == '__main__':
    unittest.main()
