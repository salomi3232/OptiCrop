from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ── Load artefacts ─────────────────────────────────────────────────────────────
MODEL_DIR = 'model'
model   = joblib.load(os.path.join(MODEL_DIR, 'crop_model.pkl'))
scaler  = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
le      = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
crop_stats = pd.read_csv(os.path.join(MODEL_DIR, 'crop_stats.csv'), index_col=0, header=[0, 1])

FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

CROP_INFO = {
    'rice': 'Staple cereal crop grown in flooded paddies.',
    'maize': 'Versatile cereal used for food, feed, and fuel.',
    'chickpea': 'Protein-rich legume suited to dry climates.',
    'kidneybeans': 'High-protein legume grown in temperate regions.',
    'pigeonpeas': 'Drought-tolerant legume popular in tropical areas.',
    'mothbeans': 'Drought-resistant legume for arid regions.',
    'mungbean': 'Fast-growing legume rich in protein.',
    'blackgram': 'Legume widely used in South Asian cuisine.',
    'lentil': 'Cool-season legume with high nutritional value.',
    'pomegranate': 'Fruit crop thriving in semi-arid climates.',
    'banana': 'Tropical fruit requiring high humidity and warmth.',
    'mango': 'Tropical fruit needing warm dry spells to flower.',
    'grapes': 'Fruit vine adaptable to a wide temperature range.',
    'watermelon': 'Warm-season fruit needing sandy well-drained soil.',
    'muskmelon': 'Sweet melon requiring warm temperatures.',
    'apple': 'Temperate fruit needing cold winters for dormancy.',
    'orange': 'Citrus fruit thriving in subtropical climates.',
    'papaya': 'Fast-growing tropical fruit rich in vitamins.',
    'coconut': 'Tropical palm crop requiring coastal humid conditions.',
    'cotton': 'Fibre crop needing high nitrogen and warm weather.',
    'jute': 'Natural fibre crop grown in humid tropical regions.',
    'coffee': 'Beverage crop thriving in cool tropical highlands.',
}


def parse_inputs(form):
    return np.array([[
        float(form['N']),
        float(form['P']),
        float(form['K']),
        float(form['temperature']),
        float(form['humidity']),
        float(form['ph']),
        float(form['rainfall']),
    ]])


def get_probabilities(X_scaled):
    probs = model.predict_proba(X_scaled)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    return [(le.classes_[i], round(probs[i] * 100, 2)) for i in top5_idx]


def suitability_score(crop_name, X_raw):
    """Return a 0-100 suitability score for a given crop."""
    if crop_name not in crop_stats.index:
        return None, []
    vals = X_raw[0]
    issues = []
    scores = []
    for i, feat in enumerate(FEATURES):
        mean = crop_stats.loc[crop_name, (feat, 'mean')]
        std  = crop_stats.loc[crop_name, (feat, 'std')]
        if std == 0:
            std = 1
        z = abs(vals[i] - mean) / std
        feat_score = max(0, 100 - z * 20)
        scores.append(feat_score)
        if z > 2:
            issues.append(f"{feat} is far from ideal (ideal ≈ {mean:.1f})")
    return round(np.mean(scores), 1), issues


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    result = None
    if request.method == 'POST':
        X_raw    = parse_inputs(request.form)
        X_scaled = scaler.transform(X_raw)
        pred_idx = model.predict(X_scaled)[0]
        crop     = le.classes_[pred_idx]
        top5     = get_probabilities(X_scaled)
        result   = {
            'crop': crop.title(),
            'info': CROP_INFO.get(crop, ''),
            'top5': top5,
            'inputs': dict(zip(FEATURES, X_raw[0]))
        }
    return render_template('recommend.html', result=result)


@app.route('/suitability', methods=['GET', 'POST'])
def suitability():
    result = None
    crops  = sorted(le.classes_)
    if request.method == 'POST':
        X_raw  = parse_inputs(request.form)
        crop   = request.form['crop']
        score, issues = suitability_score(crop, X_raw)
        level = 'High' if score >= 70 else ('Moderate' if score >= 40 else 'Low')
        result = {
            'crop': crop.title(),
            'score': score,
            'level': level,
            'issues': issues,
            'info': CROP_INFO.get(crop, ''),
        }
    return render_template('suitability.html', result=result, crops=crops)


@app.route('/research')
def research():
    df = pd.read_csv('Crop_recommendation.csv')
    stats = df.groupby('label')[FEATURES].mean().round(2).reset_index()
    stats_json = stats.to_dict(orient='records')
    plots = [f for f in os.listdir('static/plots') if f.endswith('.png')]
    return render_template('research.html', stats=stats_json, plots=plots)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    try:
        X_raw    = np.array([[data[f] for f in FEATURES]])
        X_scaled = scaler.transform(X_raw)
        pred_idx = model.predict(X_scaled)[0]
        crop     = le.classes_[pred_idx]
        top5     = get_probabilities(X_scaled)
        return jsonify({'recommended_crop': crop, 'top5': top5})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
