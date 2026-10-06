import json
import streamlit as st
import matplotlib.pyplot as plt
from src.config import TITLE, GROUPS, EXAMPLE_FILE
from src.predict import load_models, predict_all, validate_inputs
from src.explain import explain_local, contribution_figure

st.set_page_config(page_title='CVD architecture research prototype', layout='wide')
st.markdown('''<style>
html, body, .stApp, .stApp p, .stApp label, .stApp input, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, [data-testid="stMetricValue"] {
 font-family: "Times New Roman", Times, "Liberation Serif", serif; color: #151515; }
.block-container {max-width: 1100px; padding-top: 2.4rem;}
h1 {font-size: 2.35rem !important; line-height: 1.16 !important;}
.research {text-transform: uppercase; letter-spacing: .16em; font-size: .8rem;}
.result-card {border: 1px solid #d5d5d5; border-radius: 9px; padding: 1.15rem; margin: .25rem 0 1rem; min-height: 150px;}
.result-card h3 {font-size: 1.12rem; margin: 0 0 .8rem;}
.probability {font-size: 2.35rem; font-weight: bold; line-height: 1.2;}
.detail {font-size: .92rem; line-height: 1.4; margin-top: .45rem;}
.route {border-left: 5px solid; padding: .7rem 1rem; background: #fafafa; margin: .7rem 0;}
@media (max-width:640px) {h1 {font-size: 1.75rem !important;} .block-container {padding: 1.25rem 1rem;} .result-card {min-height: auto;}}
</style>''', unsafe_allow_html=True)
st.markdown('<p class="research">Research prototype</p>', unsafe_allow_html=True)
st.title(TITLE)
st.write('Research prototype for phenotype-aware comparison of cardiovascular classification architectures')
st.write('This application estimates the probability of prevalent/concurrent cardiovascular disease. It does not predict future cardiovascular events.')
st.caption('Research use only. Not intended for diagnosis or treatment decisions.')

bundle = load_models()
units = bundle['manifest']['input_units']
examples = json.loads(EXAMPLE_FILE.read_text(encoding='utf8'))


def load_example(index):
    original = examples[index]['inputs']
    # Display rounding must not change the frozen demonstration inputs.
    displayed = {f: round(float(v), 4) for f, v in original.items()}
    for f, v in displayed.items():
        st.session_state['input_' + f] = v
    st.session_state['loaded_example'] = index
    st.session_state['example_display'] = displayed
    st.session_state.pop('result', None)


def clear_inputs():
    for f in units:
        st.session_state['input_' + f] = None
    for k in ('loaded_example', 'example_display', 'result'):
        st.session_state.pop(k, None)


st.header('1. Patient inputs')
b1, b2, b3 = st.columns(3)
b1.button('Load Example 1', on_click=load_example, args=(0,), width='stretch')
b2.button('Load Example 2', on_click=load_example, args=(1,), width='stretch')
b3.button('Clear inputs', on_click=clear_inputs, width='stretch')
st.caption('Enter canonical SI units. Age ≥20 is required. Blank laboratory fields use each model’s frozen training median. No unit conversion is applied.')
if 'loaded_example' in st.session_state:
    st.caption(examples[st.session_state['loaded_example']]['name'] + ' · Figure 6 inputs. Full precision is retained for unchanged example fields; the form displays rounded values.')

with st.form('clinical_inputs'):
    columns = st.columns(3)
    for column, (group, fields) in zip(columns, GROUPS.items()):
        with column:
            st.subheader(group)
            for f in fields:
                st.number_input(f'{f} ({units[f]})', min_value=20.0 if f == 'Age' else 0.0,
                                value=None, step=1.0 if f == 'Age' else .01,
                                format='%.0f' if f == 'Age' else '%.4f', key='input_' + f)
    submitted = st.form_submit_button('Compare frozen architectures', width='stretch')

if submitted:
    inputs = {f: st.session_state['input_' + f] for f in units}
    if 'loaded_example' in st.session_state:
        original = examples[st.session_state['loaded_example']]['inputs']
        for f in units:
            if inputs[f] == st.session_state['example_display'][f]:
                inputs[f] = original[f]
    try:
        validate_inputs(inputs)
        st.session_state['result'] = {'inputs': inputs, 'outputs': predict_all(inputs)}
    except ValueError as exc:
        st.session_state.pop('result', None)
        st.error(str(exc))


def card(title, probability, detail, color='#151515'):
    st.markdown(f'<div class="result-card"><h3>{title}</h3><div class="probability" style="color:{color}">{probability:.2%}</div><div class="detail">{detail}</div></div>', unsafe_allow_html=True)


if 'result' in st.session_state:
    result = st.session_state['result'];out = result['outputs']
    st.header('2. Main results')
    st.caption('Results reflect the last submitted values. Submit again after editing inputs.')
    route = out['route'];route_color = '#a8323c' if route == 'A' else '#245e9b'
    st.subheader('Gatekeeper')
    st.write(f"Phenotype probability: {out['Gatekeeper']:.2%}")
    st.write(f"Frozen routing threshold: {bundle['manifest']['threshold']:.2%} (T = 0.4552)")
    st.markdown(f'<div class="route" style="border-color:{route_color}">Predicted route: <strong style="color:{route_color}">Route {route}</strong></div>', unsafe_allow_html=True)
    st.caption('The Gatekeeper is a phenotype-informed routing model and is not a substitute for UACR-based ODKD diagnosis.')
    left, right = st.columns(2)
    with left: card('M0 Global', out['M0'], 'Estimated prevalent/concurrent CVD probability')
    with right: card('M1 Hard phenotype routing', out['M1'], f'Routed Track {route} · Estimated prevalent/concurrent CVD probability')
    st.caption('Illustrative individual probability changes after routing; not evidence of overall superiority or clinical utility.')
    with st.expander('Show architecture comparison'):
        for col, key, title in zip(st.columns(4), ['M0','M1','M2','M3'], ['M0 Global','M1 Hard observed-trained','M2 Hard OOF-route-trained','M3 Soft integration']):
            with col: card(title, out[key], 'Estimated concurrent CVD probability', '#28734d' if key=='M3' else '#151515')
        st.write('Architecture outputs are shown for research comparison and should not be interpreted as evidence that one strategy is clinically superior.')
    st.header('3. Local model explanations')
    st.caption('Positive and negative SHAP values indicate contributions to the model output; they are not causal effects.')
    st.caption('Frozen canonical models · exact Tree SHAP · tree-path-dependent · log-odds scale · additivity checked.')
    try:
        for col, key, title in zip(st.columns(2), ['m0', 'm1_' + route.lower()], ['M0 Global', 'M1 routed Track ' + route]):
            with col:
                explanation = explain_local(key, result['inputs'])
                fig = contribution_figure(explanation, title)
                st.pyplot(fig, width='stretch');plt.close(fig)
                st.caption(f"Base output: {explanation['base_value']:.3f} log-odds; contributions sum with the base to the model output.")
    except ValueError:
        st.error('The local explanation consistency check did not pass; no explanation is displayed.')
else:
    st.caption('Enter values or load one of the two Figure 6 examples, then submit to view the frozen model outputs.')

st.divider()
st.caption('No sign-in, file upload, data download, or application-level input logging. Inputs are processed in the current server session. Do not enter names or identifying information.')
