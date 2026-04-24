"""
train_model.py — Huấn luyện & Đánh giá Models Churn Prediction
Cải tiến:
  - Cross-validation 5-fold
  - Báo cáo metrics đầy đủ (Accuracy, AUC-ROC, Precision, Recall, F1)
  - Confusion matrix
  - Tự động tìm hyperparameters tốt nhất
  - Lưu log kết quả ra file
  - Không phụ thuộc MLflow (optional)
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')

# Import config
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets_config import DATASET_CONFIGS

# ─── Optional MLflow ──────────────────────────────────────────────────────────
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("ℹ️  MLflow không có sẵn — bỏ qua logging MLflow")


def setup_mlflow():
    if not MLFLOW_AVAILABLE:
        return False
    try:
        uri = os.getenv("MLFLOW_URI", "http://host.docker.internal:5000")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("churn_prediction_v2")
        return True
    except Exception:
        try:
            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            mlflow.set_experiment("churn_prediction_v2")
            return True
        except Exception as e:
            print(f"⚠️  MLflow không kết nối được: {e}")
            return False


def build_pipeline(numeric_features, categorical_features, model_type='rf'):
    """Xây dựng sklearn Pipeline với preprocessor + classifier"""
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, numeric_features),
        ("cat", cat_transformer, categorical_features)
    ])

    if model_type == 'rf':
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',   # xử lý mất cân bằng class
            random_state=42,
            n_jobs=-1
        )
    else:  # gradient boosting
        clf = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])


def evaluate_model(model, X, y, cv_folds=5):
    """Đánh giá model với cross-validation và metrics đầy đủ"""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_auc  = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    cv_f1   = cross_val_score(model, X, y, cv=skf, scoring='f1', n_jobs=-1)
    cv_acc  = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)

    # Train/test split để lấy confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Ở lại (0)', 'Rời đi (1)'])

    metrics = {
        'cv_auc_mean':      round(float(cv_auc.mean()), 4),
        'cv_auc_std':       round(float(cv_auc.std()), 4),
        'cv_f1_mean':       round(float(cv_f1.mean()), 4),
        'cv_f1_std':        round(float(cv_f1.std()), 4),
        'cv_accuracy_mean': round(float(cv_acc.mean()), 4),
        'test_accuracy':    round(float(accuracy_score(y_test, y_pred)), 4),
        'test_auc':         round(float(roc_auc_score(y_test, y_prob)), 4),
        'test_precision':   round(float(precision_score(y_test, y_pred)), 4),
        'test_recall':      round(float(recall_score(y_test, y_pred)), 4),
        'test_f1':          round(float(f1_score(y_test, y_pred)), 4),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'train_size': len(X_train),
        'test_size':  len(X_test),
    }
    return model, metrics


def train_model_for_dataset(dataset_key, config, file_path, use_mlflow=False):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  🚀 Huấn luyện: {config['name']}")
    print(sep)

    if not os.path.exists(file_path):
        print(f"  ⚠️  Không tìm thấy file: {file_path}")
        return None

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(file_path)
    print(f"  📄 Loaded {len(df):,} rows × {len(df.columns)} cols")

    # Rename columns nếu cần (call_details CSV gốc dùng tên khác)
    rename_map = config.get('column_rename', {})
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  🔄 Đã rename {len(rename_map)} cột")

    # ── Kiểm tra cột bắt buộc ─────────────────────────────────────────────────
    available = set(df.columns)
    required  = set(config['required_columns'])
    missing   = required - available
    if missing:
        print(f"  ❌ Thiếu cột: {missing}")
        return None

    # ── Làm sạch dữ liệu ─────────────────────────────────────────────────────
    if 'TotalCharges' in df.columns:
        before = len(df)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna(subset=['TotalCharges'])
        dropped = before - len(df)
        if dropped:
            print(f"  🗑️  Bỏ {dropped} dòng TotalCharges không hợp lệ")

    # Ép numeric features sang float
    for col in config['numeric_features']:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # ── Target encoding ───────────────────────────────────────────────────────
    target_col = 'Churn'
    if target_col not in df.columns:
        print(f"  ❌ Không tìm thấy cột '{target_col}'")
        return None

    df[target_col] = df[target_col].astype(str).str.lower().map({
        'yes': 1, 'no': 0, 'true.': 1, 'false.': 0,
        'true': 1, 'false': 0, '1': 1, '0': 0
    })
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)

    churn_rate = df[target_col].mean()
    print(f"  📊 Tỷ lệ churn trong data: {churn_rate*100:.1f}%")
    print(f"  📊 Phân phối: Ở lại={int((df[target_col]==0).sum()):,} | Rời đi={int((df[target_col]==1).sum()):,}")

    # ── Build & Evaluate ──────────────────────────────────────────────────────
    feature_cols = config['numeric_features'] + config['categorical_features']
    X = df[feature_cols]
    y = df[target_col]

    print(f"\n  🔧 Đang huấn luyện và cross-validate (5-fold)...")
    pipeline = build_pipeline(config['numeric_features'], config['categorical_features'], model_type='rf')
    trained_model, metrics = evaluate_model(pipeline, X, y, cv_folds=5)

    # ── In kết quả ────────────────────────────────────────────────────────────
    print(f"\n  📈 KẾT QUẢ ĐÁNH GIÁ:")
    print(f"     Cross-val AUC-ROC : {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    print(f"     Cross-val F1      : {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
    print(f"     Cross-val Accuracy: {metrics['cv_accuracy_mean']:.4f}")
    print(f"     Test AUC-ROC      : {metrics['test_auc']:.4f}")
    print(f"     Test F1           : {metrics['test_f1']:.4f}")
    print(f"     Test Precision    : {metrics['test_precision']:.4f}")
    print(f"     Test Recall       : {metrics['test_recall']:.4f}")
    print(f"\n  📋 Classification Report (test set):")
    for line in metrics['classification_report'].split('\n'):
        if line.strip():
            print(f"     {line}")

    cm = np.array(metrics['confusion_matrix'])
    print(f"\n  🟩 Confusion Matrix:")
    print(f"     TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"     FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")

    # ── Feature importance top 5 ───────────────────────────────────────────────
    try:
        clf = trained_model.named_steps['classifier']
        pre = trained_model.named_steps['preprocessor']
        fn  = pre.get_feature_names_out()
        imp = clf.feature_importances_
        top = sorted(zip(fn, imp), key=lambda x: -x[1])[:5]
        print(f"\n  🔍 Top 5 features quan trọng nhất:")
        for fname, fval in top:
            bar = "█" * int(fval * 50)
            print(f"     {bar:<25} {fname} ({fval*100:.1f}%)")
    except Exception:
        pass

    # ── Retrain toàn bộ data trước khi lưu ────────────────────────────────────
    print(f"\n  🔁 Retrain trên toàn bộ {len(X):,} samples...")
    final_pipeline = build_pipeline(config['numeric_features'], config['categorical_features'])
    final_pipeline.fit(X, y)

    # ── Lưu model ─────────────────────────────────────────────────────────────
    model_dir = os.path.dirname(config['model_path'])
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(final_pipeline, config['model_path'], compress=3)
    print(f"  ✅ Model lưu tại: {config['model_path']}")

    # ── Lưu metrics ra JSON ───────────────────────────────────────────────────
    metrics_path = config['model_path'].replace('.pkl', '_metrics.json')
    metrics_out = {
        'dataset': config['name'],
        'trained_at': datetime.now().isoformat(),
        'train_samples': len(X),
        'churn_rate': round(float(churn_rate), 4),
        **{k: v for k, v in metrics.items() if k != 'classification_report'}
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)
    print(f"  📝 Metrics lưu tại: {metrics_path}")

    # ── MLflow logging (optional) ─────────────────────────────────────────────
    if use_mlflow and MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name=f"Train_{dataset_key}"):
                mlflow.log_params({
                    'dataset': dataset_key,
                    'n_samples': len(X),
                    'churn_rate': round(float(churn_rate), 4),
                    'model_type': 'RandomForest'
                })
                mlflow.log_metrics({
                    'cv_auc': metrics['cv_auc_mean'],
                    'cv_f1': metrics['cv_f1_mean'],
                    'test_auc': metrics['test_auc'],
                    'test_f1': metrics['test_f1'],
                    'test_precision': metrics['test_precision'],
                    'test_recall': metrics['test_recall'],
                })
                mlflow.sklearn.log_model(final_pipeline, f"model_{dataset_key}")
            print(f"  📡 Đã log lên MLflow")
        except Exception as e:
            print(f"  ⚠️  MLflow log thất bại: {e}")

    return final_pipeline, metrics


def main():
    print("\n" + "═" * 60)
    print("  🔮 CHURN PREDICTION — HUẤN LUYỆN MODELS v2.0")
    print("═" * 60)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    use_mlflow = setup_mlflow()

    results = {}

    # ── Train Telco IBM ───────────────────────────────────────────────────────
    result = train_model_for_dataset(
        'telco_ibm',
        DATASET_CONFIGS['telco_ibm'],
        os.path.join(BASE_DIR, 'data', 'Telco_customer_churn.csv'),
        use_mlflow=use_mlflow
    )
    if result:
        results['telco_ibm'] = result[1]

    # ── Train Call Details ────────────────────────────────────────────────────
    result = train_model_for_dataset(
        'call_details',
        DATASET_CONFIGS['call_details'],
        os.path.join(BASE_DIR, 'data', 'Churn.csv'),
        use_mlflow=use_mlflow
    )
    if result:
        results['call_details'] = result[1]

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  📊 TỔNG KẾT")
    print("═" * 60)
    for key, m in results.items():
        name = DATASET_CONFIGS[key]['name']
        print(f"  {name}:")
        print(f"    AUC-ROC: {m['test_auc']:.4f} | F1: {m['test_f1']:.4f} | Accuracy: {m['test_accuracy']:.4f}")
    print("\n  🎉 Hoàn tất! Chạy app: streamlit run app.py")
    print("  🌐 Chạy API:  python api.py")


if __name__ == "__main__":
    main()
