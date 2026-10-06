"""Single prediction implementation: saved medians, exact feature order, no refitting."""
from functools import lru_cache
import hashlib
import json
import math
import numbers
import numpy as np
import xgboost as xgb
from .config import MODEL_DIR


@lru_cache(maxsize=1)
def load_models():
    manifest = json.loads((MODEL_DIR / 'model_manifest.json').read_text(encoding='utf8'))
    raw = (MODEL_DIR / 'preprocessing.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['preprocessing_sha256']:
        raise ValueError('Preprocessing integrity check failed.')
    preprocessing = json.loads(raw)
    models = {}
    for key, entry in manifest['models'].items():
        path = MODEL_DIR / entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Model integrity check failed.')
        booster = xgb.Booster(params={'nthread': 1})
        booster.load_model(path)
        booster.set_param({'nthread': 1})
        spec = preprocessing[key]
        if entry['feature_list'] != spec['feature_order'] or booster.num_features() != len(spec['feature_order']):
            raise ValueError('Model feature schema mismatch.')
        if len(spec['median_imputer_values']) != booster.num_features():
            raise ValueError('Imputer schema mismatch.')
        models[key] = booster
    return {'models': models, 'preprocessing': preprocessing, 'manifest': manifest}


def validate_inputs(inputs):
    """Accept canonical SI values; missing lab values use model-specific frozen medians."""
    fields = load_models()['manifest']['input_units']
    unknown = set(inputs) - set(fields)
    if unknown:
        raise ValueError('Unrecognized input fields.')
    values = {}
    for name in fields:
        value = inputs.get(name)
        if value is None:
            values[name] = np.nan
            continue
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f'{name} must be numeric or blank.')
        value = float(value)
        if math.isnan(value):
            values[name] = value
            continue
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and non-negative.')
        values[name] = value
    if not math.isfinite(values['Age']) or values['Age'] < 20:
        raise ValueError('Age must be provided and must be at least 20 years.')
    return values


def prepare_features(model_key, inputs, gatekeeper_probability=None):
    values = validate_inputs(inputs)
    if model_key == 'm3':
        if gatekeeper_probability is None:
            gatekeeper_probability = predict_gatekeeper(inputs)
        values['ODKD_probability_OOF'] = float(gatekeeper_probability)
    spec = load_models()['preprocessing'][model_key]
    vector = np.array([[values[f] for f in spec['feature_order']]], dtype=np.float64)
    return np.where(np.isnan(vector), np.asarray(spec['median_imputer_values']), vector)


def _predict(model_key, inputs, gatekeeper_probability=None):
    vector = prepare_features(model_key, inputs, gatekeeper_probability)
    model = load_models()['models'][model_key]
    return float(model.predict(xgb.DMatrix(vector), validate_features=True)[0])


def predict_gatekeeper(inputs):
    return _predict('gatekeeper', inputs)


def assign_route(probability):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Probability must lie between zero and one.')
    return 'A' if probability >= load_models()['manifest']['threshold'] else 'B'


def predict_m0(inputs):
    return _predict('m0', inputs)


def predict_m1(inputs, gatekeeper_probability=None):
    g = predict_gatekeeper(inputs) if gatekeeper_probability is None else gatekeeper_probability
    return _predict('m1_' + assign_route(g).lower(), inputs)


def predict_m2(inputs, gatekeeper_probability=None):
    g = predict_gatekeeper(inputs) if gatekeeper_probability is None else gatekeeper_probability
    return _predict('m2_' + assign_route(g).lower(), inputs)


def predict_m3(inputs, gatekeeper_probability=None):
    g = predict_gatekeeper(inputs) if gatekeeper_probability is None else gatekeeper_probability
    return _predict('m3', inputs, g)


def predict_all(inputs):
    g = predict_gatekeeper(inputs)
    return {'Gatekeeper': g, 'route': assign_route(g), 'M0': predict_m0(inputs),
            'M1': predict_m1(inputs, g), 'M2': predict_m2(inputs, g), 'M3': predict_m3(inputs, g)}
