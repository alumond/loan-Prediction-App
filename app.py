from datetime import datetime
from html import escape
from pathlib import Path
import streamlit as st
from loan_model import EXAMPLE, FEATURES, load_model, predict_scenario, validate_scenario

st.set_page_config(page_title='Loan Studio · Scenario workspace', page_icon='◈', layout='wide')
ICONS = {
    'mark': '<path d="M5 4h7v7h7v9H5V4Z"/><path d="M12 4v7h7M9 16h6"/>',
    'grid': '<rect x="4" y="4" width="6" height="6" rx="1.5"/><rect x="14" y="4" width="6" height="6" rx="1.5"/><rect x="4" y="14" width="6" height="6" rx="1.5"/><rect x="14" y="14" width="6" height="6" rx="1.5"/>',
    'wallet': '<rect x="3" y="6" width="18" height="14" rx="3"/><path d="M17 11h4v5h-4a2.5 2.5 0 0 1 0-5ZM6 6V4h12"/>',
    'person': '<circle cx="12" cy="7" r="3"/><path d="M5 21v-2a7 7 0 0 1 14 0v2"/>',
    'spark': '<path d="m12 3 2.6 6.4L21 12l-6.4 2.6L12 21l-2.6-6.4L3 12l6.4-2.6L12 3Z"/>',
    'info': '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.5"/>',
}


def icon(name, size=20):
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


html('<style>' + Path(__file__).with_name('ui.css').read_text() + '</style>')


def set_inputs(values):
    for field in FEATURES:
        st.session_state['input_' + field] = values.get(field)
    st.session_state.result = None
    st.session_state.form_error = None


def load_example():
    set_inputs(EXAMPLE)
    st.session_state.input_mode = 'Sample scenario'


def new_scenario():
    set_inputs({})
    st.session_state.input_mode = 'New scenario'


def load_history(index):
    item = st.session_state.history[index]
    set_inputs(item['inputs'])
    st.session_state.result = item
    st.session_state.input_mode = f"Scenario {item['number']:02d}"


if 'history' not in st.session_state:
    st.session_state.history = []
    st.session_state.run_number = 0
    load_example()


@st.cache_resource
def cached_model():
    return load_model()


with st.sidebar:
    html(f'<div class="brand"><div class="brand-icon">{icon("mark",22)}</div><div><div class="brand-name">Loan Studio</div><div class="brand-sub">A scenario workspace</div></div></div>')
    st.button('New scenario', icon=':material/add:', key='new_scenario', width='stretch', on_click=new_scenario)
    html(f'<div class="side-label">Workspace</div><div class="side-active">{icon("grid",17)}<span>Loan explorer</span></div><div class="side-description">A place to explore how a model responds to different inputs.</div><div class="side-label">Recent scenarios</div>')
    if not st.session_state.history:
        html('<div class="session-empty">Your scenario results will<br>appear here as you explore.</div>')
    for index, item in enumerate(st.session_state.history):
        st.button(f"Scenario {item['number']:02d} · {item['approval_probability']:.0%}", icon=':material/history:', key=f"history_{item['number']}", type='tertiary', width='stretch', on_click=load_history, args=(index,))
    html('<div class="side-bottom"><div class="creator"><div class="avatar">AO</div><div>Almond Owolabi<br><small>Data analyst & builder</small></div></div><a class="project-link" href="https://github.com/alumond/loan-Prediction-App" target="_blank" rel="noopener noreferrer">View the project ↗</a></div>')

html(f'<div class="topline"><div class="breadcrumb">{icon("grid",15)}<span>Workspace</span><span style="color:#4a535d">/</span><strong>Loan explorer</strong></div><div class="demo-pill"><span class="dot"></span>Portfolio demo</div></div><div class="hero"><div class="eyebrow">Your scenario workspace</div><h1 class="page-title">Explore a loan scenario.</h1><p class="page-subtitle">Adjust the details and see what the model predicts.<br>Start with the sample below, or build a scenario of your own.</p></div>')

left, right = st.container(key='workspace').columns([1.85, 1], gap='large')
with left:
    title_col, example_col = st.columns([2.1, 1], vertical_alignment='center')
    with title_col:
        html(f'<div class="workspace-top"><h2>Scenario details</h2><span class="sample-tag">{escape(st.session_state.input_mode)}</span></div>')
    with example_col:
        st.button('Load sample', icon=':material/refresh:', key='load_example', on_click=load_example, width='stretch')

    def section(name, glyph, number, first=False):
        html(f'<div class="section-heading {"first" if first else ""}"><span class="section-icon">{icon(glyph,15)}</span><span class="section-title">{name}</span><span class="section-number">{number}</span></div>')

    with st.form('scenario_form', enter_to_submit=False):
        section('The loan', 'mark', '01', first=True)
        a, b, c = st.columns(3)
        with a:
            amount = st.number_input('Loan amount', min_value=0.0, step=10.0, format='%.0f', key='input_LoanAmount', help="Keep the original dataset's amount scale. Its currency and units are not documented in this project.", placeholder='e.g. 120')
        with b:
            term = st.number_input('Loan term', min_value=0.0, step=12.0, format='%.0f', key='input_Loan_Amount_Term', help="Uses the original model's term values. The example uses 360.", placeholder='e.g. 360')
        with c:
            area = st.selectbox('Property area', ['Urban', 'Semiurban', 'Rural'], index=None, key='input_Property_Area', placeholder='Select an area')
        html('<div class="divider"></div>')
        section('Income & household', 'wallet', '02')
        a, b, c = st.columns(3)
        with a:
            income = st.number_input('Applicant income', min_value=0, step=500, key='input_ApplicantIncome', placeholder='e.g. 5000')
        with b:
            co_income = st.number_input('Co-applicant income', min_value=0, step=500, key='input_CoapplicantIncome', placeholder='e.g. 1500')
        with c:
            dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'], index=None, key='input_Dependents', placeholder='Select dependents')
        html('<div class="divider"></div>')
        section('Applicant profile', 'person', '03')
        a, b, c = st.columns(3)
        with a:
            gender = st.selectbox('Gender', ['Male', 'Female'], index=None, key='input_Gender', placeholder='Select gender')
        with b:
            married = st.selectbox('Married', ['Yes', 'No'], index=None, key='input_Married', placeholder='Select an option')
        with c:
            education = st.selectbox('Education', ['Graduate', 'Not Graduate'], index=None, key='input_Education', placeholder='Select education')
        a, b = st.columns(2)
        with a:
            employed = st.selectbox('Self-employed', ['Yes', 'No'], index=None, key='input_Self_Employed', placeholder='Select an option')
        with b:
            credit = st.selectbox('Credit-history code', [1.0, 0.0], index=None, format_func=lambda value: str(int(value)), key='input_Credit_History', help='The model expects a 0 or 1 credit-history code. The project does not document its criteria; use sample values for this demo.', placeholder='Select a code')
        submitted = st.form_submit_button('Run scenario', type='primary', icon=':material/arrow_forward:', width='stretch')
        html('<div class="form-footnote">Use made-up details to explore the model.</div>')

    if submitted:
        values = dict(Gender=gender, Married=married, Dependents=dependents, Education=education, Self_Employed=employed, ApplicantIncome=income, CoapplicantIncome=co_income, LoanAmount=amount, Loan_Amount_Term=term, Credit_History=credit, Property_Area=area)
        error = validate_scenario(values)
        st.session_state.form_error = error
        if error:
            st.session_state.result = None
        else:
            try:
                prediction = predict_scenario(cached_model(), values)
                st.session_state.run_number += 1
                item = {**prediction, 'inputs': values, 'number': st.session_state.run_number, 'time': datetime.now().strftime('%H:%M')}
                st.session_state.result = item
                st.session_state.input_mode = f"Scenario {item['number']:02d}"
                st.session_state.history = [item, *st.session_state.history][:6]
            except Exception:
                st.session_state.result = None
                st.session_state.form_error = 'The model could not load or run. Please try again after the app environment has been checked.'
            else:
                st.rerun()
    if st.session_state.form_error:
        st.error(st.session_state.form_error, icon=':material/info:')

with right:
    with st.container(key='result_column'):
        html('<div class="result-spacer"></div>')
        result = st.session_state.result
        if result:
            percent = result['approval_probability'] * 100
            outcome = 'Approval predicted' if result['approved'] else 'Decline predicted'
            inputs = result['inputs']
            html(f'''<div class="result-card"><div class="result-header"><span>Your result</span><span class="result-badge">Scenario {result['number']:02d}</span></div><div class="result-outcome">{outcome}</div><div class="probability">{percent:.0f}<small>%</small></div><div class="probability-label">Estimated approval probability</div><div class="probability-track" role="meter" aria-label="Estimated approval probability" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent:.1f}"><div class="probability-fill" style="width:{percent:.4f}%"></div></div><div class="track-labels"><span>0%</span><span>100%</span></div><div class="snapshot"><div class="snapshot-row"><span>Applicant income</span><strong>{inputs['ApplicantIncome']:,.0f}</strong></div><div class="snapshot-row"><span>Loan amount</span><strong>{inputs['LoanAmount']:,.0f}</strong></div><div class="snapshot-row"><span>Property area</span><strong>{escape(inputs['Property_Area'])}</strong></div></div><p class="result-note">Last submitted scenario. Run again after changing any details.<br>A lender has not reviewed or approved an application.</p></div>''')
        else:
            illustration = '<svg width="162" height="110" viewBox="0 0 162 110" fill="none" aria-hidden="true"><rect x="22" y="22" width="86" height="68" rx="10" fill="#293632" stroke="#496055"/><rect x="45" y="10" width="91" height="74" rx="10" fill="#28322e" stroke="#779183"/><path d="M59 26h25M59 35h43" stroke="#8fa898" stroke-width="2" stroke-linecap="round"/><rect x="59" y="57" width="9" height="13" rx="2" fill="#536f5f"/><rect x="75" y="48" width="9" height="22" rx="2" fill="#799984"/><rect x="91" y="39" width="9" height="31" rx="2" fill="#b5d6c1"/><circle cx="125" cy="79" r="19" fill="#c3e5d2"/><path d="M116 79h18m-6-6 6 6-6 6" stroke="#243c2d" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            html(f'<div class="result-card"><div class="result-header"><span>Your result</span>{icon("spark",18)}</div><div class="empty-visual">{illustration}</div><div class="empty-title">Ready when you are.</div><p class="empty-text">Run a scenario to see the predicted outcome and estimated approval probability.</p><div class="mini-steps"><span>Add details</span>→<span>Run scenario</span>→<span>Review</span></div></div>')
        html(f'<div class="guide-card"><div class="guide-heading">{icon("info",15)}<span>Reading the result</span></div><p class="guide-copy">The percentage is the model’s approval estimate. It does not measure how accurate the model is.</p><div class="guide-rule"></div><p class="guide-copy">Change one input at a time to compare scenarios. Recent results stay in this browser session.</p></div>')

html(f'<div class="bottom-note">{icon("info",14)}<span>A portfolio demonstration using an existing prediction model. No loan application is submitted.<br>Currency and amount units are not documented, so values are shown without currency symbols.</span></div>')
with st.expander('About this demo'):
    st.write('The original model uses the eleven fields shown above, including demographic information. This interface does not establish its accuracy, fairness or suitability for real lending decisions. Use invented examples.')
    st.write('Changing the form does not replace an earlier result until you select Run scenario. Up to six results are held in the current session; they are not saved to a database.')
