"""
REST API cho Churn Prediction System v3.0
─────────────────────────────────────────
Built-in models (telco_ibm, call_details):
  GET  /health
  GET  /api/models
  POST /api/predict
  POST /api/predict/batch
  POST /api/predict/csv

Custom models (bất kỳ dataset nào):
  GET    /api/custom/models
  POST   /api/custom/analyze
  POST   /api/custom/train
  POST   /api/custom/predict/<model_id>
  POST   /api/custom/predict/<model_id>/batch
  POST   /api/custom/predict/<model_id>/csv
  DELETE /api/custom/models/<model_id>
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import warnings
import traceback
import json
from io import StringIO, BytesIO

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets_config import DATASET_CONFIGS, detect_dataset_type, get_retention_recommendations
from universal_trainer import (
    analyze_columns, suggest_mapping, train_universal,
    list_custom_models, load_custom_model, predict_universal,
    get_generic_recommendations, CUSTOM_MODELS_DIR
)

app = Flask(__name__)
CORS(app)

# ─── Load built-in models ─────────────────────────────────────────────────────
_builtin_cache = {}

def load_all_builtin():
    global _builtin_cache
    for key, config in DATASET_CONFIGS.items():
        path = config['model_path']
        if os.path.exists(path):
            try:
                _builtin_cache[key] = {'model': joblib.load(path), 'config': config}
            except Exception as e:
                print(f"[WARN] Không load được model {key}: {e}")

load_all_builtin()


def _extract_fi(model):
    fi = {}
    try:
        clf = model.named_steps['classifier']
        pre = model.named_steps['preprocessor']
        fn = pre.get_feature_names_out()
        imp = clf.feature_importances_
        for fname, fval in zip(fn, imp):
            root = fname.split('__', 1)[-1].rsplit('_', 1)[0]
            fi[root] = fi.get(root, 0) + round(float(fval), 4)
    except Exception:
        pass
    return dict(sorted(fi.items(), key=lambda x: -x[1]))


def _run_batch(model, feature_cols, df, numeric_features):
    df = df.copy()
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}")
    proba = model.predict_proba(df[feature_cols])
    df['Churn_Probability'] = proba[:, 1]
    df['Churn_Prediction'] = (proba[:, 1] >= 0.5).astype(int)
    df['Risk_Level'] = df['Churn_Probability'].apply(
        lambda p: 'HIGH' if p > 0.7 else ('MEDIUM' if p > 0.4 else 'LOW')
    )
    return df


def _summary(df):
    total = len(df)
    churn_count = int(df['Churn_Prediction'].sum())
    return {
        'total_records': total,
        'churn_count': churn_count,
        'churn_rate_pct': round(churn_count / total * 100, 2) if total else 0,
        'high_risk_count': int((df['Churn_Probability'] > 0.75).sum()),
        'avg_churn_probability': round(float(df['Churn_Probability'].mean()), 4),
    }


def _read_csv_bytes(raw_bytes):
    for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
        try:
            return pd.read_csv(BytesIO(raw_bytes), encoding=enc)
        except Exception:
            continue
    raise ValueError("Không đọc được file CSV — kiểm tra encoding")


# ═════════════════════════════════════════════════════════════════════════════
# BUILT-IN ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    custom = list_custom_models()
    return jsonify({
        'status': 'ok',
        'version': '3.0.0',
        'builtin_models': list(_builtin_cache.keys()),
        'custom_models_count': len(custom),
    })


@app.route('/api/models', methods=['GET'])
def list_models():
    result = []
    for key, info in _builtin_cache.items():
        cfg = info['config']
        result.append({
            'id': key, 'type': 'builtin',
            'name': cfg['name'],
            'numeric_features': cfg['numeric_features'],
            'categorical_features': cfg['categorical_features'],
            'categorical_options': cfg.get('categorical_options', {}),
        })
    return jsonify({'models': result})


@app.route('/api/predict', methods=['POST'])
def predict_single():
    """Body: { "dataset_type": "telco_ibm", "features": {...} }"""
    try:
        data = request.get_json(force=True)
        if not data or 'features' not in data:
            return jsonify({'error': 'Thiếu "features"'}), 400

        features = data['features']
        dataset_type = data.get('dataset_type') or detect_dataset_type(pd.DataFrame([features]))

        if not dataset_type or dataset_type not in _builtin_cache:
            return jsonify({'error': f'Không nhận diện được dataset. dataset_type={dataset_type}'}), 400

        info = _builtin_cache[dataset_type]
        config = info['config']
        model = info['model']
        rename_map = config.get('column_rename', {})
        renamed = {rename_map.get(k, k): v for k, v in features.items()}
        input_df = pd.DataFrame([renamed])
        for col in config['numeric_features']:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        feature_cols = config['numeric_features'] + config['categorical_features']
        proba = model.predict_proba(input_df[feature_cols])[0]
        churn_prob = float(proba[1])
        recs = get_retention_recommendations(dataset_type, renamed, churn_prob)

        return jsonify({
            'dataset_type': dataset_type,
            'churn_probability': round(churn_prob, 4),
            'churn_probability_pct': f"{churn_prob*100:.1f}%",
            'is_churn': churn_prob >= 0.5,
            'risk_level': 'HIGH' if churn_prob > 0.7 else ('MEDIUM' if churn_prob > 0.4 else 'LOW'),
            'feature_importance': _extract_fi(model),
            'retention': recs,
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch_json():
    """Body: { "dataset_type": "telco_ibm", "records": [{...},...] }"""
    try:
        data = request.get_json(force=True)
        records = data.get('records', [])
        if not records:
            return jsonify({'error': 'Thiếu "records"'}), 400

        df = pd.DataFrame(records)
        dataset_type = data.get('dataset_type') or detect_dataset_type(df)
        if not dataset_type or dataset_type not in _builtin_cache:
            return jsonify({'error': 'Không nhận diện được dataset'}), 400

        info = _builtin_cache[dataset_type]
        config = info['config']
        df = df.rename(columns=config.get('column_rename', {}))
        feature_cols = config['numeric_features'] + config['categorical_features']
        df_out = _run_batch(info['model'], feature_cols, df, config['numeric_features'])

        return jsonify({'dataset_type': dataset_type, **_summary(df_out),
            'results': df_out[['Churn_Probability','Churn_Prediction','Risk_Level']].round(4).to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/predict/csv', methods=['POST'])
def predict_batch_csv():
    """multipart/form-data: file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Thiếu field "file"'}), 400
        df = _read_csv_bytes(request.files['file'].read())
        dataset_type = detect_dataset_type(df)
        if not dataset_type or dataset_type not in _builtin_cache:
            return jsonify({'error': 'Không nhận diện được dataset. Dùng /api/custom/predict/<id>/csv cho dataset tùy chỉnh.'}), 400

        info = _builtin_cache[dataset_type]
        config = info['config']
        df = df.rename(columns=config.get('column_rename', {}))
        feature_cols = config['numeric_features'] + config['categorical_features']
        df_out = _run_batch(info['model'], feature_cols, df, config['numeric_features'])

        return jsonify({'dataset_type': dataset_type, **_summary(df_out),
            'results': df_out[['Churn_Probability','Churn_Prediction','Risk_Level']].round(4).to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM MODEL ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/custom/models', methods=['GET'])
def custom_list():
    models = list_custom_models()
    result = []
    for m in models:
        met = m.get('metrics', {})
        result.append({
            'model_id': m['model_id'],
            'dataset_name': m.get('dataset_name'),
            'trained_at': m.get('trained_at'),
            'target_col': m.get('target_col'),
            'numeric_features': m.get('numeric_features', []),
            'categorical_features': m.get('categorical_features', []),
            'total_samples': met.get('total_samples'),
            'churn_rate': met.get('churn_rate'),
            'test_auc': met.get('test_auc'),
            'test_f1': met.get('test_f1'),
        })
    return jsonify({'custom_models': result, 'count': len(result)})


@app.route('/api/custom/analyze', methods=['POST'])
def custom_analyze():
    """
    Phân tích cột CSV, trả về mapping gợi ý — không train.
    multipart/form-data: file
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Thiếu "file"'}), 400
        df = _read_csv_bytes(request.files['file'].read())
        col_analysis = analyze_columns(df)
        mapping = suggest_mapping(col_analysis)

        return jsonify({
            'rows': len(df), 'columns': len(df.columns),
            'suggested_mapping': mapping,
            'column_analysis': {
                col: {
                    'dtype': info['dtype'], 'role': info['role'],
                    'n_unique': info['n_unique'], 'missing_pct': info['missing_pct'],
                    'suggestion': info['suggestion'],
                    'sample_values': [str(v) for v in info['sample_values'][:3]],
                }
                for col, info in col_analysis.items()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/custom/train', methods=['POST'])
def custom_train():
    """
    Train model từ CSV upload.
    multipart/form-data:
      file           — CSV file (bắt buộc)
      model_name     — tên nhận biết (optional)
      target_col     — cột target (optional, auto-detect nếu bỏ)
      numeric_cols   — JSON array (optional)
      categorical_cols — JSON array (optional)
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Thiếu "file"'}), 400
        df = _read_csv_bytes(request.files['file'].read())

        model_name = request.form.get('model_name', 'custom_api')
        target_col = request.form.get('target_col')
        nc_raw = request.form.get('numeric_cols')
        cc_raw = request.form.get('categorical_cols')

        if target_col and (nc_raw or cc_raw):
            mapping = {
                'target': target_col,
                'numeric': json.loads(nc_raw) if nc_raw else [],
                'categorical': json.loads(cc_raw) if cc_raw else [],
                'ignore': [],
            }
        else:
            col_analysis = analyze_columns(df)
            mapping = suggest_mapping(col_analysis)
            if target_col:
                mapping['target'] = target_col

        result = train_universal(df, mapping, model_name)
        m = result['metrics']

        return jsonify({
            'status': 'success',
            'model_id': result['model_id'],
            'dataset_name': model_name,
            'target_col': mapping['target'],
            'numeric_features': mapping['numeric'],
            'categorical_features': mapping['categorical'],
            'total_samples': m['total_samples'],
            'churn_rate': m['churn_rate'],
            'metrics': {
                'test_auc': m['test_auc'], 'test_f1': m['test_f1'],
                'test_accuracy': m['test_accuracy'],
                'test_precision': m['test_precision'], 'test_recall': m['test_recall'],
                'cv_auc_mean': m['cv_auc_mean'], 'cv_f1_mean': m['cv_f1_mean'],
            },
            'feature_importance': result['feature_importance'],
        }), 201

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/custom/predict/<model_id>', methods=['POST'])
def custom_predict_single(model_id):
    """Body: { "features": { "col1": val1, ... } }"""
    try:
        model, meta = load_custom_model(model_id)
        if model is None:
            return jsonify({'error': f'Không tìm thấy model "{model_id}"'}), 404

        data = request.get_json(force=True)
        if not data or 'features' not in data:
            return jsonify({'error': 'Thiếu "features"'}), 400

        features = data['features']
        num_feats = meta.get('numeric_features', [])
        input_df = pd.DataFrame([features])
        for col in num_feats:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        churn_prob = predict_universal(model, meta, input_df)
        fi = meta.get('feature_importance', {})
        recs = get_generic_recommendations(fi, features, churn_prob, meta.get('numeric_stats', {}))

        return jsonify({
            'model_id': model_id,
            'dataset_name': meta.get('dataset_name'),
            'churn_probability': round(churn_prob, 4),
            'churn_probability_pct': f"{churn_prob*100:.1f}%",
            'is_churn': churn_prob >= 0.5,
            'risk_level': 'HIGH' if churn_prob > 0.7 else ('MEDIUM' if churn_prob > 0.4 else 'LOW'),
            'top_features': dict(list(fi.items())[:5]),
            'retention': recs,
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/custom/predict/<model_id>/batch', methods=['POST'])
def custom_predict_batch(model_id):
    """Body: { "records": [{...}, ...] }"""
    try:
        model, meta = load_custom_model(model_id)
        if model is None:
            return jsonify({'error': f'Model "{model_id}" không tồn tại'}), 404

        records = request.get_json(force=True).get('records', [])
        if not records:
            return jsonify({'error': 'Thiếu "records"'}), 400

        df = pd.DataFrame(records)
        num_feats = meta.get('numeric_features', [])
        feature_cols = num_feats + meta.get('categorical_features', [])
        df_out = _run_batch(model, feature_cols, df, num_feats)

        return jsonify({'model_id': model_id, 'dataset_name': meta.get('dataset_name'),
            **_summary(df_out),
            'results': df_out[['Churn_Probability','Churn_Prediction','Risk_Level']].round(4).to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/custom/predict/<model_id>/csv', methods=['POST'])
def custom_predict_csv(model_id):
    """multipart/form-data: file"""
    try:
        model, meta = load_custom_model(model_id)
        if model is None:
            return jsonify({'error': f'Model "{model_id}" không tồn tại'}), 404
        if 'file' not in request.files:
            return jsonify({'error': 'Thiếu "file"'}), 400

        df = _read_csv_bytes(request.files['file'].read())
        num_feats = meta.get('numeric_features', [])
        feature_cols = num_feats + meta.get('categorical_features', [])
        df_out = _run_batch(model, feature_cols, df, num_feats)

        return jsonify({'model_id': model_id, 'dataset_name': meta.get('dataset_name'),
            **_summary(df_out),
            'results': df_out[['Churn_Probability','Churn_Prediction','Risk_Level']].round(4).to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/custom/models/<model_id>', methods=['DELETE'])
def custom_delete(model_id):
    try:
        import glob
        deleted = []
        for f in glob.glob(os.path.join(CUSTOM_MODELS_DIR, f"{model_id}*")):
            os.remove(f)
            deleted.append(os.path.basename(f))
        if not deleted:
            return jsonify({'error': f'Model "{model_id}" không tồn tại'}), 404
        return jsonify({'status': 'deleted', 'model_id': model_id, 'files_removed': deleted})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('API_PORT', 8001))
    print(f"\n🚀 ChurnIQ API v3.0 — http://0.0.0.0:{port}")
    print("─" * 50)
    print("  GET  /health")
    print("  GET  /api/models")
    print("  POST /api/predict              (built-in)")
    print("  POST /api/predict/batch        (built-in)")
    print("  POST /api/predict/csv          (built-in)")
    print("  GET  /api/custom/models")
    print("  POST /api/custom/analyze")
    print("  POST /api/custom/train")
    print("  POST /api/custom/predict/<id>")
    print("  POST /api/custom/predict/<id>/batch")
    print("  POST /api/custom/predict/<id>/csv")
    print("  DEL  /api/custom/models/<id>")
    print("─" * 50)
    app.run(host='0.0.0.0', port=port, debug=False)
