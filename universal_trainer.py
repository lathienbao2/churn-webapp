"""
universal_trainer.py — Train model từ BẤT KỲ CSV nào
Tự động:
  - Phân tích cột (numeric / categorical / datetime / text / id)
  - Gợi ý cột target (Churn)
  - Train Pipeline hoàn chỉnh
  - Đánh giá metrics
  - Tạo đề xuất giữ chân dựa trên SHAP values / feature importance
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import hashlib
import warnings
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_MODELS_DIR = os.path.join(BASE_DIR, "models", "custom")
os.makedirs(CUSTOM_MODELS_DIR, exist_ok=True)


# ─── Phân tích cột ───────────────────────────────────────────────────────────

CHURN_KEYWORDS = [
    'churn', 'leave', 'exit', 'cancel', 'attrition', 'left', 'departed',
    'rời', 'bỏ', 'nghỉ', 'hủy', 'thoát', 'quit', 'dropout', 'unsubscribe'
]

# Các cột chứa thông tin SAU KHI churn (data leakage) — nên ignore
LEAKAGE_KEYWORDS = [
    'churn_reason', 'churnreason', 'churn_category', 'churncategory',
    'churn_score', 'churnscore', 'churn_rate', 'churnrate',
    'customer_status', 'customerstatus', 'satisfaction_score',
    'exit_reason', 'cancellation_reason',
]

ID_KEYWORDS = [
    'id', 'uuid', 'key', 'index', 'serial', 'ref',
    'customerid', 'userid', 'accountid', 'phonenumber',
]

# Columns that are geographic noise
GEO_KEYWORDS = ['zipcode', 'zip_code', 'zip', 'latitude', 'longitude', 'latlong', 'lat_long', 'country', 'city', 'state', 'areaco']

DATE_KEYWORDS = ['date', 'time', 'ngày', 'tháng', 'năm', 'created', 'updated', 'timestamp']

CAT_MAX_CARDINALITY = 30   # Số lượng unique tối đa để coi là categorical
NUM_MIN_UNIQUE = 10        # Số unique tối thiểu để coi là numeric (nếu dtype object)


def analyze_columns(df: pd.DataFrame) -> dict:
    """
    Phân tích từng cột và phân loại thành:
      numeric, categorical, datetime, id, text, target_candidate, leakage
    """
    analysis = {}

    # Tìm target chính xác nhất: cột tên đúng 'Churn' / 'churn' ưu tiên
    # Sau đó mới xét các cột chứa từ khóa churn
    exact_target = None
    for col in df.columns:
        if col.strip().lower() == 'churn':
            exact_target = col
            break

    for col in df.columns:
        series = df[col].dropna()
        n_total = len(df)
        n_unique = series.nunique()
        n_missing = df[col].isna().sum()
        unique_ratio = n_unique / max(n_total, 1)
        col_norm = col.lower().replace(' ', '').replace('_', '').replace('-', '')

        info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'n_unique': n_unique,
            'n_missing': n_missing,
            'missing_pct': round(n_missing / n_total * 100, 1),
            'unique_ratio': round(unique_ratio, 3),
            'sample_values': series.head(5).tolist(),
            'role': None,
            'sub_type': None,
            'suggestion': None,
        }

        # 0. Data leakage columns (post-event info)
        if any(kw in col_norm for kw in LEAKAGE_KEYWORDS):
            info['role'] = 'leakage'
            info['suggestion'] = '⚠️ Có thể là data leakage (thông tin sau sự kiện churn) — nên bỏ qua'
            analysis[col] = info
            continue

        # 1. Kiểm tra target
        is_exact = (col == exact_target)
        has_churn_kw = any(kw in col_norm for kw in CHURN_KEYWORDS)

        if is_exact or has_churn_kw:
            unique_vals = set(str(v).lower().strip() for v in series.unique())
            binary_vals = {'0', '1', 'yes', 'no', 'true', 'false', 'true.', 'false.', '1.0', '0.0', 'churn', 'not churn'}
            if n_unique <= 5 and (unique_vals.issubset(binary_vals) or is_exact):
                info['role'] = 'target_candidate'
                info['suggestion'] = '✅ Cột mục tiêu (Churn) — được gợi ý tự động'
                analysis[col] = info
                continue

        # 2. Geographic / noise
        if any(kw in col_norm for kw in GEO_KEYWORDS):
            info['role'] = 'id'
            info['suggestion'] = '🗺️ Dữ liệu địa lý — thường không có ích, nên bỏ qua'
            analysis[col] = info
            continue

        # 3. ID / irrelevant (high cardinality string or exact keyword match)
        if any(col_norm == kw or col_norm.endswith(kw) for kw in ID_KEYWORDS):
            info['role'] = 'id'
            info['suggestion'] = 'ID / mã định danh — nên bỏ qua'
            analysis[col] = info
            continue
        if unique_ratio > 0.9 and n_unique > 100 and not pd.api.types.is_numeric_dtype(df[col]):
            info['role'] = 'id'
            info['suggestion'] = 'Cardinality quá cao — có thể là ID, nên bỏ qua'
            analysis[col] = info
            continue

        # 4. Datetime
        if any(kw in col_norm for kw in DATE_KEYWORDS):
            info['role'] = 'datetime'
            info['suggestion'] = 'Cột thời gian — có thể trích xuất năm/tháng/ngày'
            analysis[col] = info
            continue

        # 5. Numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            if n_unique <= 2:
                info['role'] = 'categorical'
                info['sub_type'] = 'binary_numeric'
                info['suggestion'] = 'Nhị phân (0/1) — nên dùng là categorical'
            elif n_unique <= CAT_MAX_CARDINALITY and unique_ratio < 0.05:
                info['role'] = 'categorical'
                info['sub_type'] = 'ordinal_numeric'
                info['suggestion'] = f'Có {n_unique} giá trị số — có thể là categorical'
            else:
                info['role'] = 'numeric'
                info['sub_type'] = 'continuous'
                mn, mx, me = series.min(), series.max(), series.mean()
                info['suggestion'] = f'Số liên tục — min={mn:.2f}, max={mx:.2f}, mean={me:.2f}'
            analysis[col] = info
            continue

        # 6. Object / string
        if n_unique <= 2:
            info['role'] = 'categorical'
            info['sub_type'] = 'binary'
            info['suggestion'] = f'Nhị phân: {series.unique().tolist()}'
        elif n_unique <= CAT_MAX_CARDINALITY:
            info['role'] = 'categorical'
            info['sub_type'] = 'nominal'
            info['suggestion'] = f'{n_unique} danh mục: {series.unique()[:5].tolist()}'
        elif unique_ratio > 0.5:
            info['role'] = 'text'
            info['suggestion'] = 'Free text / cardinality cao — nên bỏ qua'
        else:
            info['role'] = 'categorical'
            info['sub_type'] = 'high_cardinality'
            info['suggestion'] = f'Nhiều danh mục ({n_unique}) — cân nhắc bỏ qua'

        analysis[col] = info

    return analysis
    """
    Phân tích từng cột và phân loại thành:
      numeric, categorical, datetime, id, text, target_candidate
    """
    analysis = {}

    for col in df.columns:
        series = df[col].dropna()
        n_total = len(df)
        n_unique = series.nunique()
        n_missing = df[col].isna().sum()
        unique_ratio = n_unique / max(n_total, 1)
        col_lower = col.lower().replace(' ', '').replace('_', '')

        info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'n_unique': n_unique,
            'n_missing': n_missing,
            'missing_pct': round(n_missing / n_total * 100, 1),
            'unique_ratio': round(unique_ratio, 3),
            'sample_values': series.head(5).tolist(),
            'role': None,
            'sub_type': None,
            'suggestion': None,
        }

        # 1. Kiểm tra target (churn)
        if any(kw in col_lower for kw in CHURN_KEYWORDS):
            unique_vals = set(str(v).lower() for v in series.unique())
            binary_vals = {'0', '1', 'yes', 'no', 'true', 'false', 'true.', 'false.', '1.0', '0.0'}
            if unique_vals.issubset(binary_vals) or n_unique <= 5:
                info['role'] = 'target_candidate'
                info['suggestion'] = 'Cột mục tiêu (Churn) — được gợi ý tự động'
                analysis[col] = info
                continue

        # 2. ID / irrelevant
        if any(kw == col_lower for kw in ID_KEYWORDS) or (unique_ratio > 0.9 and n_unique > 100):
            info['role'] = 'id'
            info['suggestion'] = 'ID / mã định danh — nên bỏ qua'
            analysis[col] = info
            continue

        # 3. Datetime
        if any(kw in col_lower for kw in DATE_KEYWORDS):
            info['role'] = 'datetime'
            info['suggestion'] = 'Cột thời gian — có thể trích xuất năm/tháng/ngày'
            analysis[col] = info
            continue

        # 4. Numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            if n_unique <= 2:
                info['role'] = 'categorical'
                info['sub_type'] = 'binary_numeric'
                info['suggestion'] = 'Nhị phân (0/1) — nên dùng là categorical'
            elif n_unique <= CAT_MAX_CARDINALITY and unique_ratio < 0.05:
                info['role'] = 'categorical'
                info['sub_type'] = 'ordinal_numeric'
                info['suggestion'] = f'Có {n_unique} giá trị — có thể là categorical'
            else:
                info['role'] = 'numeric'
                info['sub_type'] = 'continuous'
                info['suggestion'] = f'Số liên tục — min={series.min():.2f}, max={series.max():.2f}, mean={series.mean():.2f}'
            analysis[col] = info
            continue

        # 5. Object / string
        if n_unique <= 2:
            info['role'] = 'categorical'
            info['sub_type'] = 'binary'
            info['suggestion'] = f'Nhị phân: {series.unique().tolist()}'
        elif n_unique <= CAT_MAX_CARDINALITY:
            info['role'] = 'categorical'
            info['sub_type'] = 'nominal'
            info['suggestion'] = f'{n_unique} danh mục: {series.unique()[:5].tolist()}'
        elif unique_ratio > 0.5:
            info['role'] = 'text'
            info['suggestion'] = 'Free text / high cardinality — nên bỏ qua'
        else:
            info['role'] = 'categorical'
            info['sub_type'] = 'high_cardinality'
            info['suggestion'] = f'Nhiều danh mục ({n_unique}) — cân nhắc bỏ qua'

        analysis[col] = info

    return analysis


def suggest_mapping(col_analysis: dict) -> dict:
    """
    Từ kết quả phân tích, tự động đề xuất mapping:
    {
      'target': 'Churn',
      'numeric': [...],
      'categorical': [...],
      'ignore': [...],
    }
    """
    target = None
    numeric = []
    categorical = []
    ignore = []

    for col, info in col_analysis.items():
        role = info['role']
        if role == 'target_candidate':
            target = col
        elif role == 'numeric':
            numeric.append(col)
        elif role == 'categorical':
            categorical.append(col)
        elif role in ('id', 'datetime', 'text'):
            ignore.append(col)
        else:
            ignore.append(col)

    return {
        'target': target,
        'numeric': numeric,
        'categorical': categorical,
        'ignore': ignore,
    }


# ─── Encode target ────────────────────────────────────────────────────────────

def encode_target(series: pd.Series) -> pd.Series:
    """Chuyển target về 0/1"""
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        'yes': 1, 'no': 0, '1': 1, '0': 0,
        'true': 1, 'false': 0, 'true.': 1, 'false.': 0,
        '1.0': 1, '0.0': 0, 'churn': 1, 'not churn': 0,
        'rời': 1, 'ở lại': 0,
    }
    encoded = s.map(mapping)
    # Nếu không map được, dùng LabelEncoder
    if encoded.isna().any():
        le = LabelEncoder()
        encoded = pd.Series(le.fit_transform(series.fillna('missing')), index=series.index)
    return encoded.astype(int)


# ─── Build Pipeline ───────────────────────────────────────────────────────────

def build_universal_pipeline(numeric_features, categorical_features):
    transformers = []

    if numeric_features:
        num_t = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        transformers.append(('num', num_t, numeric_features))

    if categorical_features:
        cat_t = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20)),
        ])
        transformers.append(('cat', cat_t, categorical_features))

    preprocessor = ColumnTransformer(transformers, remainder='drop')

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([('preprocessor', preprocessor), ('classifier', clf)])


# ─── Train & Evaluate ─────────────────────────────────────────────────────────

def train_universal(df: pd.DataFrame, mapping: dict, dataset_name: str = "custom") -> dict:
    """
    Train model từ DataFrame với mapping đã cho.
    Trả về dict: {model, metrics, model_id, feature_importance}
    """
    target_col = mapping['target']
    numeric_features = [c for c in mapping['numeric'] if c in df.columns]
    categorical_features = [c for c in mapping['categorical'] if c in df.columns]

    if not target_col or target_col not in df.columns:
        raise ValueError(f"Cột target '{target_col}' không tồn tại")
    if not numeric_features and not categorical_features:
        raise ValueError("Phải có ít nhất 1 feature numeric hoặc categorical")

    # Chuẩn bị data
    feature_cols = numeric_features + categorical_features
    df_work = df[feature_cols + [target_col]].copy()

    # Coerce numeric
    for col in numeric_features:
        df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

    # Encode target
    df_work['__target__'] = encode_target(df_work[target_col])
    df_work = df_work.dropna(subset=['__target__'])

    y = df_work['__target__']
    X = df_work[feature_cols]

    n_total = len(X)
    churn_rate = float(y.mean())

    if n_total < 50:
        raise ValueError(f"Quá ít dữ liệu ({n_total} dòng). Cần ít nhất 50 dòng.")
    if y.nunique() < 2:
        raise ValueError("Target chỉ có 1 giá trị — không thể train classifier.")

    # Cross-validation
    pipeline = build_universal_pipeline(numeric_features, categorical_features)
    cv_folds = min(5, max(2, n_total // 100))
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_auc = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    cv_f1  = cross_val_score(pipeline, X, y, cv=skf, scoring='f1', n_jobs=-1)

    # Train/test split để lấy metrics trên test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Feature importance
    clf = pipeline.named_steps['classifier']
    pre = pipeline.named_steps['preprocessor']
    fn  = pre.get_feature_names_out()
    imp = clf.feature_importances_

    # Gộp one-hot về feature gốc
    feature_importance_raw = {}
    for fname, fval in zip(fn, imp):
        base = fname.split('__', 1)[-1]  # bỏ 'num__' / 'cat__'
        # Với one-hot, base = "FeatureName_Value" → lấy phần trước dấu _
        root = base.rsplit('_', 1)[0] if '_' in base else base
        feature_importance_raw[root] = feature_importance_raw.get(root, 0) + fval

    fi_sorted = dict(sorted(feature_importance_raw.items(), key=lambda x: -x[1]))

    # Retrain trên toàn bộ data
    final_pipeline = build_universal_pipeline(numeric_features, categorical_features)
    final_pipeline.fit(X, y)

    # Model ID: hash tên cột + dataset_name
    col_sig = "|".join(sorted(feature_cols)) + "|" + target_col
    model_id = dataset_name + "_" + hashlib.md5(col_sig.encode()).hexdigest()[:8]
    model_path = os.path.join(CUSTOM_MODELS_DIR, f"{model_id}.pkl")
    meta_path  = os.path.join(CUSTOM_MODELS_DIR, f"{model_id}_meta.json")

    # Compute stats for UI (form defaults / recommendations)
    numeric_stats = {}
    for col in numeric_features:
        if col in df_work.columns:
            s = df_work[col].dropna()
            numeric_stats[col] = {
                'mean':  round(float(s.mean()), 3),
                'median': round(float(s.median()), 3),
                'q25':   round(float(s.quantile(0.25)), 3),
                'q75':   round(float(s.quantile(0.75)), 3),
                'min':   round(float(s.min()), 3),
                'max':   round(float(s.max()), 3),
            }

    categorical_samples = {}
    for col in categorical_features:
        if col in df_work.columns:
            categorical_samples[col] = sorted(df_work[col].dropna().unique().tolist()[:20], key=str)

    joblib.dump(final_pipeline, model_path, compress=3)

    metrics = {
        'cv_auc_mean':    round(float(cv_auc.mean()), 4),
        'cv_auc_std':     round(float(cv_auc.std()),  4),
        'cv_f1_mean':     round(float(cv_f1.mean()),  4),
        'cv_f1_std':      round(float(cv_f1.std()),   4),
        'test_accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
        'test_auc':       round(float(roc_auc_score(y_test, y_prob)), 4),
        'test_precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'test_recall':    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'test_f1':        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'train_size': int(len(X_train)),
        'test_size':  int(len(X_test)),
        'total_samples': int(n_total),
        'churn_rate': round(churn_rate, 4),
        'n_features': len(feature_cols),
        'cv_folds': cv_folds,
    }

    meta = {
        'model_id': model_id,
        'dataset_name': dataset_name,
        'trained_at': datetime.now().isoformat(),
        'target_col': target_col,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'feature_importance': {k: round(v, 4) for k, v in fi_sorted.items()},
        'numeric_stats': numeric_stats,
        'categorical_samples': categorical_samples,
        'metrics': metrics,
        'model_path': model_path,
    }

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        'model': final_pipeline,
        'model_id': model_id,
        'model_path': model_path,
        'meta_path': meta_path,
        'metrics': metrics,
        'feature_importance': fi_sorted,
        'meta': meta,
    }


# ─── Load custom model ────────────────────────────────────────────────────────

def load_custom_model(model_id: str):
    path = os.path.join(CUSTOM_MODELS_DIR, f"{model_id}.pkl")
    meta_path = os.path.join(CUSTOM_MODELS_DIR, f"{model_id}_meta.json")
    if not os.path.exists(path):
        return None, None
    model = joblib.load(path)
    meta = json.load(open(meta_path, encoding='utf-8')) if os.path.exists(meta_path) else {}
    return model, meta


def list_custom_models():
    """Liệt kê tất cả custom models đã train"""
    models = []
    for fname in os.listdir(CUSTOM_MODELS_DIR):
        if fname.endswith('_meta.json'):
            try:
                with open(os.path.join(CUSTOM_MODELS_DIR, fname), encoding='utf-8') as f:
                    meta = json.load(f)
                models.append(meta)
            except Exception:
                pass
    return sorted(models, key=lambda x: x.get('trained_at', ''), reverse=True)


# ─── Universal prediction ─────────────────────────────────────────────────────

def predict_universal(model, meta: dict, input_df: pd.DataFrame):
    """Dự đoán với bất kỳ model nào — thêm cột thiếu dưới dạng NaN"""
    numeric_features = meta.get('numeric_features', [])
    categorical_features = meta.get('categorical_features', [])
    feature_cols = numeric_features + categorical_features

    df_in = input_df.copy()

    # Thêm cột thiếu để pipeline có thể impute
    for col in numeric_features:
        if col not in df_in.columns:
            df_in[col] = np.nan
        else:
            df_in[col] = pd.to_numeric(df_in[col], errors='coerce')
    for col in categorical_features:
        if col not in df_in.columns:
            df_in[col] = np.nan

    proba = model.predict_proba(df_in[feature_cols])
    churn_prob = float(proba[0][1]) if proba.shape[0] == 1 else proba[:, 1].tolist()
    return churn_prob


# ─── Generic recommendations ─────────────────────────────────────────────────

def get_generic_recommendations(feature_importance: dict, input_features: dict,
                                 churn_prob: float, numeric_stats: dict = None) -> dict:
    """
    Tạo đề xuất dựa trên feature importance + giá trị thực tế của khách hàng.
    Hoạt động với BẤT KỲ dataset nào (không hardcode tên cột).
    """
    priority = 'high' if churn_prob > 0.7 else ('medium' if churn_prob > 0.4 else 'low')
    recommendations = []

    # Lấy top 3 features quan trọng nhất
    top_features = list(feature_importance.items())[:3]

    for feat, importance in top_features:
        val = input_features.get(feat)
        stats = (numeric_stats or {}).get(feat, {})

        if val is None:
            continue

        # Nếu là số → so sánh với trung bình
        try:
            val_num = float(val)
            mean_val = stats.get('mean', None)
            q75 = stats.get('q75', None)
            q25 = stats.get('q25', None)

            if mean_val is not None:
                if val_num > (q75 or mean_val * 1.3):
                    recommendations.append({
                        'action': f'⚠️ Chú ý: {feat} đang cao bất thường',
                        'detail': f'Giá trị hiện tại ({val_num:.1f}) cao hơn mức trung bình ({mean_val:.1f}). '
                                  f'Đây là yếu tố quan trọng ảnh hưởng đến quyết định rời đi.',
                        'impact': 'Cao' if importance > 0.15 else 'Trung bình',
                        'effort': 'Trung bình',
                        'feature': feat,
                    })
                elif val_num < (q25 or mean_val * 0.7):
                    recommendations.append({
                        'action': f'💡 Tăng engagement: {feat}',
                        'detail': f'Giá trị {feat} ({val_num:.1f}) thấp hơn bình thường ({mean_val:.1f}). '
                                  f'Có thể khách hàng chưa tận dụng hết dịch vụ.',
                        'impact': 'Trung bình',
                        'effort': 'Thấp',
                        'feature': feat,
                    })
        except (ValueError, TypeError):
            # Là categorical
            recommendations.append({
                'action': f'📋 Xem xét lại: {feat} = {val}',
                'detail': f'Đặc điểm "{feat}" của khách hàng là một trong những yếu tố dự báo churn quan trọng. '
                          f'Cân nhắc chính sách ưu đãi phù hợp với nhóm "{val}".',
                'impact': 'Cao' if importance > 0.15 else 'Trung bình',
                'effort': 'Thấp',
                'feature': feat,
            })

    # Thêm đề xuất chung theo mức độ
    if priority == 'high':
        recommendations.append({
            'action': '🚨 Liên hệ ngay trong 48 giờ',
            'detail': f'Xác suất rời đi {churn_prob*100:.0f}% — khách hàng cần được chăm sóc khẩn cấp. '
                      f'Gọi điện trực tiếp và đề xuất ưu đãi giữ chân.',
            'impact': 'Rất cao',
            'effort': 'Trung bình',
            'feature': None,
        })
    elif priority == 'medium':
        recommendations.append({
            'action': '📩 Gửi ưu đãi cá nhân hóa trong tuần này',
            'detail': f'Xác suất rời đi {churn_prob*100:.0f}% — theo dõi và gửi email/SMS với ưu đãi phù hợp.',
            'impact': 'Trung bình',
            'effort': 'Thấp',
            'feature': None,
        })
    else:
        recommendations.append({
            'action': '✅ Duy trì chất lượng dịch vụ',
            'detail': f'Xác suất rời đi thấp ({churn_prob*100:.0f}%). Tiếp tục giữ trải nghiệm tốt và '
                      f'khảo sát định kỳ để phát hiện thay đổi sớm.',
            'impact': 'Thấp',
            'effort': 'Rất thấp',
            'feature': None,
        })

    return {'priority': priority, 'recommendations': recommendations}
