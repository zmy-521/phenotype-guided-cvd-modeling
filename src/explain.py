"""Exact local Tree SHAP on raw model output, matching the canonical Figure 6 method."""
from functools import lru_cache
import numpy as np
import xgboost as xgb
import shap
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .predict import load_models, prepare_features


@lru_cache(maxsize=3)
def _explainer(key):
    b = load_models()
    return shap.TreeExplainer(b['models'][key], data=None,
        feature_perturbation='tree_path_dependent', model_output='raw',
        feature_names=b['preprocessing'][key]['feature_order'])


def explain_local(model_key, inputs):
    if model_key not in ('m0', 'm1_a', 'm1_b'):
        raise ValueError('Local explanations are restricted to M0 and routed M1.')
    data = prepare_features(model_key, inputs)
    explainer = _explainer(model_key)
    values = np.asarray(explainer.shap_values(data, approximate=False, check_additivity=True))[0].astype(float)
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    booster = load_models()['models'][model_key]
    margin = float(booster.predict(xgb.DMatrix(data), output_margin=True)[0])
    probability = float(booster.predict(xgb.DMatrix(data))[0])
    error = abs(base + values.sum() - margin)
    probability_error = abs(float(expit(base + values.sum())) - probability)
    if error > 2e-6 or probability_error > 1e-6:
        raise ValueError('Local explanation additivity check failed.')
    return {'features': load_models()['preprocessing'][model_key]['feature_order'],
            'values': values.tolist(), 'base_value': base, 'margin': margin,
            'probability': probability, 'margin_error': float(error),
            'probability_error': probability_error}


def contribution_figure(explanation, title):
    values = np.asarray(explanation['values'])
    order = np.argsort(np.abs(values))
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
                         'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=(6.4, 5.1), layout='constrained')
        colors = ['#30343b' if values[i] >= 0 else '#a9adb4' for i in order]
        ax.barh(np.arange(len(order)), values[order], color=colors, height=.62)
        ax.set_yticks(np.arange(len(order)), [explanation['features'][i] for i in order])
        ax.axvline(0, color='#555555', linewidth=.8)
        ax.set_xlabel('Contribution to model output (log-odds)')
        ax.set_ylabel('Feature')
        ax.set_title(f"{title}\nPredicted probability = {explanation['probability']:.2%}", loc='left', fontsize=13)
        ax.grid(axis='x', color='#eeeeee');ax.set_axisbelow(True)
        return fig
