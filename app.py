"""
Churn Prediction System v2.0
Giao diện Streamlit cải tiến: charts, UX tốt hơn, đề xuất giải pháp chi tiết
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import json
from io import StringIO
import traceback

warnings.filterwarnings('ignore')

# ─── Import universal trainer ────────────────────────────────────────────────
from universal_trainer import (
    analyze_columns, suggest_mapping, train_universal,
    list_custom_models, load_custom_model, predict_universal,
    get_generic_recommendations, encode_target
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ — Dự Đoán Khách Hàng Rời Đi",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS tùy chỉnh ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }

/* Header gradient */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99,179,237,0.2);
}
.main-header h1 { color: #e2e8f0; margin:0; font-size:1.8rem; font-weight:700; }
.main-header p  { color: #94a3b8; margin:.4rem 0 0; font-size:.95rem; }

/* Risk badge */
.risk-high   { background:#fee2e2; color:#991b1b; padding:.3rem .9rem; border-radius:20px; font-weight:600; font-size:.85rem; }
.risk-medium { background:#fef3c7; color:#92400e; padding:.3rem .9rem; border-radius:20px; font-weight:600; font-size:.85rem; }
.risk-low    { background:#d1fae5; color:#065f46; padding:.3rem .9rem; border-radius:20px; font-weight:600; font-size:.85rem; }

/* Rec cards */
.rec-card {
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: .85rem 1rem;
    margin: .5rem 0;
}
.rec-card.high  { border-color: #ef4444; }
.rec-card.medium{ border-color: #f59e0b; }
.rec-card.low   { border-color: #10b981; }
.rec-title { font-weight:600; color:#1e293b; font-size:.95rem; }
.rec-detail { color:#475569; font-size:.88rem; margin-top:.25rem; }
.rec-meta   { display:flex; gap:.75rem; margin-top:.4rem; }
.tag { font-size:.75rem; padding:.15rem .5rem; border-radius:10px; background:#e2e8f0; color:#475569; }

/* Metric override */
[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: .75rem 1rem !important;
}

/* Gauge */
.gauge-wrap { text-align:center; }
.gauge-num  { font-size:3rem; font-weight:700; }
.gauge-high   { color:#ef4444; }
.gauge-medium { color:#f59e0b; }
.gauge-low    { color:#10b981; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label { color: #94a3b8 !important; font-size:.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Import config ────────────────────────────────────────────────────────────
from datasets_config import DATASET_CONFIGS, detect_dataset_type, get_retention_recommendations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải models...")
def load_models():
    models = {}
    for key, config in DATASET_CONFIGS.items():
        path = config['model_path']
        if os.path.exists(path):
            try:
                models[key] = {'model': joblib.load(path), 'config': config}
            except Exception as e:
                st.warning(f"⚠️ Không load được model {key}: {e}")
    return models

models_dict = load_models()

# ─── Helper functions ─────────────────────────────────────────────────────────
def get_feature_importances(model_entry, input_df):
    """Trả về dict {tên feature gốc: importance}"""
    try:
        model = model_entry['model']
        clf = model.named_steps['classifier']
        pre = model.named_steps['preprocessor']
        fn = pre.get_feature_names_out()
        imp = clf.feature_importances_
        result = {}
        for fname, fval in zip(fn, imp):
            clean = fname.split('__', 1)[-1]
            # Gộp one-hot về feature gốc
            base = clean.split('_')[0] if '_' in clean else clean
            result[clean] = round(float(fval), 4)
        return result
    except Exception:
        return {}

def run_prediction(model_entry, input_df):
    model = model_entry['model']
    config = model_entry['config']
    feature_cols = config['numeric_features'] + config['categorical_features']
    proba = model.predict_proba(input_df[feature_cols])[0]
    raw = model.predict(input_df[feature_cols])[0]
    churn_prob = float(proba[1])
    is_churn = bool(raw == 1 or str(raw).lower() in ('yes', 'true', 'true.'))
    return churn_prob, is_churn

def risk_badge(prob):
    if prob > 0.7:
        return '<span class="risk-high">🔴 Nguy cơ CAO</span>'
    elif prob > 0.4:
        return '<span class="risk-medium">🟡 Nguy cơ TRUNG BÌNH</span>'
    return '<span class="risk-low">🟢 Nguy cơ THẤP</span>'

def render_gauge(prob):
    pct = prob * 100
    cls = "gauge-high" if prob > 0.7 else ("gauge-medium" if prob > 0.4 else "gauge-low")
    return f"""
    <div class="gauge-wrap">
        <div class="gauge-num {cls}">{pct:.1f}%</div>
        <div style="color:#64748b;font-size:.9rem;margin-top:.25rem;">Xác suất rời đi</div>
    </div>"""

def render_recs(recs_data):
    priority = recs_data['priority']
    recs = recs_data['recommendations']
    impact_color = {'Rất cao': 'high', 'Cao': 'high', 'Trung bình': 'medium', 'Thấp': 'low', 'Rất thấp': 'low'}
    html = ""
    for r in recs:
        cls = impact_color.get(r.get('impact', 'Thấp'), 'low')
        html += f"""
        <div class="rec-card {cls}">
            <div class="rec-title">{r['action']}</div>
            <div class="rec-detail">{r['detail']}</div>
            <div class="rec-meta">
                <span class="tag">Impact: {r.get('impact','?')}</span>
                <span class="tag">Effort: {r.get('effort','?')}</span>
            </div>
        </div>"""
    return html


# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🔮 ChurnIQ — Hệ Thống Dự Đoán Khách Hàng Rời Đi</h1>
    <p>Phân tích nguy cơ churn & đề xuất giải pháp giữ chân khách hàng theo thời gian thực</p>
</div>
""", unsafe_allow_html=True)

if not models_dict:
    st.error("❌ Không tìm thấy model nào. Vui lòng chạy `python train_model.py` trước.")
    st.stop()

# ─── Model status bar ────────────────────────────────────────────────────────
cols_status = st.columns(len(models_dict) + 1)
with cols_status[0]:
    st.markdown(f"**{len(models_dict)} models sẵn sàng**")
for i, (key, info) in enumerate(models_dict.items()):
    with cols_status[i + 1]:
        st.success(f"✅ {info['config']['name']}")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_single, tab_batch, tab_custom, tab_saved, tab_insight = st.tabs([
    "🎯 Dự Đoán Đơn Lẻ",
    "📦 Dự Đoán Hàng Loạt",
    "🧩 Dataset Tùy Chỉnh",
    "💾 Models Đã Lưu",
    "📊 Phân Tích & Insights"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: DỰ ĐOÁN ĐƠN LẺ
# ─────────────────────────────────────────────────────────────────────────────
with tab_single:
    col_form, col_result = st.columns([1, 1.2], gap="large")

    with col_form:
        st.subheader("📝 Nhập thông tin khách hàng")

        selected_type = st.selectbox(
            "Loại dataset",
            options=list(models_dict.keys()),
            format_func=lambda x: DATASET_CONFIGS[x]['name'],
            key="single_type"
        )
        config = models_dict[selected_type]['config']
        labels = config.get('feature_labels', {})
        options_map = config.get('categorical_options', {})
        defaults_num = config.get('numeric_defaults', {})

        inputs = {}
        with st.container():
            st.markdown("**📐 Thông số số**")
            ncols = st.columns(min(3, len(config['numeric_features'])))
            for i, feat in enumerate(config['numeric_features']):
                with ncols[i % len(ncols)]:
                    label = labels.get(feat, feat)
                    default = defaults_num.get(feat, 0.0)
                    step = 0.1 if any(k in feat.lower() for k in ['charges', 'minutes']) else 1.0
                    inputs[feat] = st.number_input(label, value=default, step=step, key=f"num_{feat}")

        with st.container():
            st.markdown("**🏷️ Thông số phân loại**")
            ccols = st.columns(min(2, len(config['categorical_features'])))
            for i, feat in enumerate(config['categorical_features']):
                with ccols[i % len(ccols)]:
                    label = labels.get(feat, feat)
                    opts = options_map.get(feat, ['No', 'Yes'])
                    inputs[feat] = st.selectbox(label, opts, key=f"cat_{feat}")

        predict_btn = st.button("🔮 Dự Đoán Ngay", type="primary", use_container_width=True)

    with col_result:
        st.subheader("📊 Kết Quả Phân Tích")

        if predict_btn:
            try:
                model_entry = models_dict[selected_type]
                input_df = pd.DataFrame([inputs])
                for feat in config['numeric_features']:
                    input_df[feat] = pd.to_numeric(input_df[feat], errors='coerce')

                churn_prob, is_churn = run_prediction(model_entry, input_df)
                recs_data = get_retention_recommendations(selected_type, inputs, churn_prob)

                # Gauge
                st.markdown(render_gauge(churn_prob), unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center;margin:.5rem 0'>{risk_badge(churn_prob)}</div>", unsafe_allow_html=True)
                st.markdown("")

                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("Dự đoán", "⚠️ Rời đi" if is_churn else "✅ Ở lại")
                m2.metric("Xác suất", f"{churn_prob*100:.1f}%")
                m3.metric("Mức độ ưu tiên", recs_data['priority'].upper())

                # Feature importance bar chart
                fi = get_feature_importances(model_entry, input_df)
                if fi:
                    st.markdown("**🔍 Yếu tố ảnh hưởng nhiều nhất**")
                    # Top 6 features
                    top_fi = dict(sorted(fi.items(), key=lambda x: -x[1])[:6])
                    fi_df = pd.DataFrame({'Feature': list(top_fi.keys()), 'Importance': list(top_fi.values())})
                    fi_df = fi_df.sort_values('Importance')
                    st.bar_chart(fi_df.set_index('Feature'))

                # Recommendations
                st.markdown("**💡 Đề xuất giải pháp giữ chân**")
                st.markdown(render_recs(recs_data), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Lỗi dự đoán: {e}")
                with st.expander("Chi tiết lỗi"):
                    st.code(traceback.format_exc())
        else:
            st.info("👈 Điền thông tin khách hàng và nhấn **Dự Đoán Ngay**")
            st.markdown("""
            **Hướng dẫn sử dụng:**
            - Chọn loại dataset phù hợp với dữ liệu của bạn
            - Nhập thông số khách hàng cần kiểm tra
            - Xem kết quả xác suất churn & đề xuất giải pháp
            """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: DỰ ĐOÁN HÀNG LOẠT
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("📤 Tải lên file CSV để dự đoán hàng loạt")

    col_up, col_fmt = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Kéo thả file CSV vào đây", type=["csv"], label_visibility="collapsed")

    with col_fmt:
        with st.expander("📖 Định dạng CSV hỗ trợ"):
            st.markdown("""
**Telco IBM:** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `PaymentMethod`, `InternetService`, `TechSupport`, `OnlineSecurity`

**Call Details:** `Account length`, `Total day minutes`, `Total eve minutes`, `Total night minutes`, `Total intl minutes`, `Number vmail messages`, `Customer service calls`, `International plan`, `Voice mail plan`
            """)

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            st.markdown(f"**Preview:** {len(df_raw):,} dòng × {len(df_raw.columns)} cột")
            st.dataframe(df_raw.head(3), use_container_width=True, hide_index=True)

            dataset_type = detect_dataset_type(df_raw)
            if not dataset_type:
                st.error("❌ Không nhận diện được định dạng dataset!")
                st.stop()

            st.success(f"✅ Phát hiện: **{DATASET_CONFIGS[dataset_type]['name']}**")

            if dataset_type not in models_dict:
                st.error("❌ Chưa có model cho dataset này!")
                st.stop()

            if st.button("🚀 Bắt đầu dự đoán hàng loạt", type="primary", use_container_width=True):
                with st.spinner("Đang xử lý..."):
                    model_entry = models_dict[dataset_type]
                    config = model_entry['config']
                    model = model_entry['model']

                    df = df_raw.copy()
                    rename_map = config.get('column_rename', {})
                    df = df.rename(columns=rename_map)

                    for col in config['numeric_features']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    if 'TotalCharges' in df.columns:
                        dropped = df['TotalCharges'].isna().sum()
                        df = df.dropna(subset=['TotalCharges'])
                        if dropped:
                            st.warning(f"⚠️ Bỏ {dropped} dòng do TotalCharges không hợp lệ")

                    feature_cols = config['numeric_features'] + config['categorical_features']
                    missing = [c for c in feature_cols if c not in df.columns]
                    if missing:
                        st.error(f"❌ File thiếu cột: {missing}")
                        st.stop()

                    proba = model.predict_proba(df[feature_cols])
                    df["Churn_Probability"] = proba[:, 1]
                    raw_preds = model.predict(df[feature_cols])
                    if hasattr(raw_preds, 'dtype') and raw_preds.dtype == object:
                        df["Churn_Prediction"] = pd.Series(raw_preds).map(
                            {'Yes': 1, 'No': 0, 'True.': 1, 'False.': 0, 'true': 1, 'false': 0}
                        ).fillna(0).astype(int).values
                    else:
                        df["Churn_Prediction"] = raw_preds.astype(int)

                    total = len(df)
                    churn_cnt = int(df["Churn_Prediction"].sum())
                    high_risk = int((df["Churn_Probability"] > 0.75).sum())
                    med_risk  = int(((df["Churn_Probability"] > 0.4) & (df["Churn_Probability"] <= 0.75)).sum())
                    low_risk  = total - high_risk - med_risk

                # ── Summary metrics ──
                st.markdown("### 📊 Tổng Quan Kết Quả")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Tổng khách hàng", f"{total:,}")
                c2.metric("Dự đoán rời đi", f"{churn_cnt:,}", f"{churn_cnt/total*100:.1f}%")
                c3.metric("🔴 Nguy cơ cao (>75%)", f"{high_risk:,}")
                c4.metric("🟡 Nguy cơ TB (40-75%)", f"{med_risk:,}")

                # ── Charts ──
                chart_c1, chart_c2 = st.columns(2)

                with chart_c1:
                    st.markdown("**Phân phối xác suất churn**")
                    hist_data = pd.cut(
                        df["Churn_Probability"],
                        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
                    ).value_counts().sort_index()
                    st.bar_chart(hist_data)

                with chart_c2:
                    st.markdown("**Phân tầng rủi ro**")
                    risk_df = pd.DataFrame({
                        'Mức độ rủi ro': ['🔴 Cao (>75%)', '🟡 Trung bình (40-75%)', '🟢 Thấp (<40%)'],
                        'Số khách hàng': [high_risk, med_risk, low_risk]
                    })
                    st.bar_chart(risk_df.set_index('Mức độ rủi ro'))

                # ── High-risk table ──
                if high_risk > 0:
                    st.markdown(f"### 🚨 Top {min(20, high_risk)} Khách Hàng Nguy Cơ Cao Nhất")
                    high_df = df.nlargest(min(20, high_risk), "Churn_Probability")

                    id_col = next((c for c in ['customerID', 'CustomerID', 'Account length', 'State'] if c in high_df.columns), None)
                    show_cols = ([id_col] if id_col else []) + feature_cols[:4] + ["Churn_Probability"]
                    show_cols = [c for c in show_cols if c in high_df.columns]

                    st.dataframe(
                        high_df[show_cols].style.format({"Churn_Probability": "{:.1%}"})
                            .background_gradient(subset=["Churn_Probability"], cmap="RdYlGn_r"),
                        use_container_width=True,
                        hide_index=True
                    )

                # ── Download ──
                csv_out = df.to_csv(index=False)
                st.download_button(
                    "💾 Tải kết quả đầy đủ (CSV)",
                    data=csv_out,
                    file_name=f"churn_predictions_{dataset_type}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
            with st.expander("Chi tiết lỗi"):
                st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: PHÂN TÍCH & INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_insight:
    st.subheader("📊 Phân Tích Dữ Liệu Mẫu & Feature Importance")

    ins_type = st.selectbox(
        "Chọn dataset để phân tích",
        list(models_dict.keys()),
        format_func=lambda x: DATASET_CONFIGS[x]['name'],
        key="insight_type"
    )

    config = models_dict[ins_type]['config']
    data_path = os.path.join(BASE_DIR, 'data',
        'Telco_customer_churn.csv' if ins_type == 'telco_ibm' else 'Churn.csv')

    if os.path.exists(data_path):
        @st.cache_data
        def load_sample(path, dtype):
            df = pd.read_csv(path)
            if dtype == 'call_details':
                rename_map = DATASET_CONFIGS['call_details'].get('column_rename', {})
                df = df.rename(columns=rename_map)
            return df

        df_sample = load_sample(data_path, ins_type)

        # Detect target column
        target_col = 'Churn'
        if target_col in df_sample.columns:
            df_sample[target_col] = df_sample[target_col].astype(str).str.lower().map(
                {'yes': 1, 'no': 0, 'true.': 1, 'false.': 0, 'true': 1, 'false': 0, '1': 1, '0': 0}
            ).fillna(0)

        st.markdown(f"**Dataset:** {len(df_sample):,} records | {len(df_sample.columns)} cột")

        # ── Churn rate overview ──
        if target_col in df_sample.columns:
            churn_rate = df_sample[target_col].mean()
            c1, c2, c3 = st.columns(3)
            c1.metric("Tỷ lệ churn thực tế", f"{churn_rate*100:.1f}%")
            c2.metric("Số khách hàng churn", f"{int(df_sample[target_col].sum()):,}")
            c3.metric("Số khách hàng ở lại", f"{int((df_sample[target_col]==0).sum()):,}")

        # ── Feature Importance từ model ──
        st.markdown("### 🔍 Tầm Quan Trọng Của Từng Đặc Trưng (Model)")
        model_obj = models_dict[ins_type]['model']
        try:
            clf = model_obj.named_steps['classifier']
            pre = model_obj.named_steps['preprocessor']
            fn = pre.get_feature_names_out()
            imp = clf.feature_importances_

            fi_df = pd.DataFrame({'Feature': fn, 'Importance': imp})
            fi_df['Feature_Clean'] = fi_df['Feature'].str.split('__').str[-1]
            fi_df = fi_df.sort_values('Importance', ascending=False).head(12)

            st.bar_chart(fi_df.set_index('Feature_Clean')['Importance'])

            st.markdown("**Top yếu tố ảnh hưởng đến churn:**")
            for _, row in fi_df.head(5).iterrows():
                label = config.get('feature_labels', {}).get(row['Feature_Clean'].split('_')[0], row['Feature_Clean'])
                bar = "█" * int(row['Importance'] * 100)
                st.markdown(f"`{bar:<20}` **{row['Feature_Clean']}** — {row['Importance']*100:.1f}%")
        except Exception as e:
            st.warning(f"Không lấy được feature importance: {e}")

        # ── Distribution charts ──
        if target_col in df_sample.columns:
            st.markdown("### 📈 Phân Tích Đặc Trưng Theo Churn")
            num_feats = config['numeric_features']
            cols = st.columns(min(3, len(num_feats)))
            for i, feat in enumerate(num_feats[:3]):
                if feat in df_sample.columns:
                    with cols[i]:
                        st.markdown(f"**{feat}**")
                        grp = df_sample.groupby(target_col)[feat].mean().reset_index()
                        grp[target_col] = grp[target_col].map({0: 'Ở lại', 1: 'Rời đi'})
                        st.bar_chart(grp.set_index(target_col)[feat])

            # Categorical breakdown
            cat_feats = config['categorical_features']
            if cat_feats:
                st.markdown("### 📊 Tỷ Lệ Churn Theo Đặc Trưng Phân Loại")
                cat_cols = st.columns(min(2, len(cat_feats)))
                for i, feat in enumerate(cat_feats[:4]):
                    if feat in df_sample.columns:
                        with cat_cols[i % 2]:
                            st.markdown(f"**{feat}**")
                            grp = df_sample.groupby(feat)[target_col].mean().sort_values(ascending=False)
                            grp_df = grp.reset_index()
                            grp_df.columns = [feat, 'Churn Rate']
                            grp_df['Churn Rate'] = grp_df['Churn Rate'] * 100
                            st.bar_chart(grp_df.set_index(feat)['Churn Rate'])

    else:
        st.warning(f"Không tìm thấy file dữ liệu mẫu tại: {data_path}")
        st.info("Đặt file CSV vào thư mục `data/` để xem phân tích.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: DATASET TÙY CHỈNH — train model từ bất kỳ CSV nào
# ═════════════════════════════════════════════════════════════════════════════
with tab_custom:
    st.subheader("🧩 Tải Dataset Bất Kỳ & Train Model")
    st.markdown("Upload bất kỳ file CSV nào có cột churn/attrition — hệ thống tự phân tích, gợi ý mapping, và train model mới.")

    up_col, info_col = st.columns([2, 1])

    with up_col:
        custom_file = st.file_uploader(
            "Kéo thả file CSV của bạn vào đây",
            type=["csv"],
            key="custom_upload",
            label_visibility="collapsed"
        )
    with info_col:
        st.info("**Yêu cầu tối thiểu:**\n- Ít nhất 50 dòng\n- Có cột target (Yes/No, 0/1, True/False)\n- Có ít nhất 2 features\n- Encoding UTF-8")

    if custom_file:
        try:
            # Thử đọc với nhiều encoding
            raw = custom_file.read()
            for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    df_custom = pd.read_csv(pd.io.common.BytesIO(raw), encoding=enc)
                    break
                except Exception:
                    continue

            st.markdown(f"**✅ Đọc thành công:** `{custom_file.name}` — {len(df_custom):,} dòng × {len(df_custom.columns)} cột")

            # Preview
            with st.expander("👁️ Xem trước dữ liệu (5 dòng đầu)"):
                st.dataframe(df_custom.head(), use_container_width=True, hide_index=True)

            # Phân tích cột
            with st.spinner("🔍 Đang phân tích cột..."):
                col_analysis = analyze_columns(df_custom)
                auto_mapping  = suggest_mapping(col_analysis)

            # ── Bảng phân tích cột ──────────────────────────────────────────
            st.markdown("### 🔬 Phân Tích Cột Tự Động")

            role_emoji = {
                'numeric': '🔢', 'categorical': '🏷️', 'target_candidate': '🎯',
                'id': '🆔', 'datetime': '📅', 'text': '📝', 'leakage': '⚠️',
            }
            role_color = {
                'numeric': '#dbeafe', 'categorical': '#d1fae5', 'target_candidate': '#fef3c7',
                'id': '#f1f5f9', 'datetime': '#ede9fe', 'text': '#fce7f3', 'leakage': '#fee2e2',
            }

            analysis_rows = []
            for col, info in col_analysis.items():
                role = info['role'] or 'unknown'
                analysis_rows.append({
                    'Cột': col,
                    'Kiểu dữ liệu': info['dtype'],
                    'Loại': f"{role_emoji.get(role, '❓')} {role}",
                    'Unique': info['n_unique'],
                    'Thiếu (%)': f"{info['missing_pct']}%",
                    'Gợi ý': info['suggestion'] or '',
                })

            st.dataframe(
                pd.DataFrame(analysis_rows),
                use_container_width=True,
                hide_index=True,
                height=min(400, len(analysis_rows) * 38 + 40),
            )

            st.markdown("---")

            # ── Column Mapping UI ────────────────────────────────────────────
            st.markdown("### 🗺️ Cấu Hình Mapping Cột")
            st.markdown("Hệ thống đã tự gợi ý. Bạn có thể điều chỉnh bên dưới:")

            all_cols = list(df_custom.columns)
            none_opt = ["(bỏ qua)"]

            map_col1, map_col2 = st.columns(2)

            with map_col1:
                # Target selection
                st.markdown("**🎯 Cột Target (Churn)**")
                target_default = auto_mapping.get('target')
                target_idx = all_cols.index(target_default) if target_default and target_default in all_cols else 0
                chosen_target = st.selectbox(
                    "Chọn cột churn/attrition",
                    all_cols,
                    index=target_idx,
                    key="custom_target"
                )

                # Dataset name
                st.markdown("**📛 Tên Model**")
                suggested_name = custom_file.name.replace('.csv', '').replace(' ', '_')[:30]
                model_name = st.text_input("Đặt tên nhận biết", value=suggested_name, key="custom_name")

            with map_col2:
                st.markdown("**ℹ️ Phân phối Target**")
                try:
                    from universal_trainer import encode_target as _enc
                    target_enc = _enc(df_custom[chosen_target])
                    churn_rate_preview = target_enc.mean()
                    c1p, c2p = st.columns(2)
                    c1p.metric("Tỷ lệ Churn", f"{churn_rate_preview*100:.1f}%")
                    c2p.metric("Tổng records", f"{len(df_custom):,}")
                    st.markdown(f"Ở lại: **{int((target_enc==0).sum()):,}** | Rời đi: **{int((target_enc==1).sum()):,}**")
                except Exception:
                    st.info("Chọn cột target hợp lệ để xem phân phối")

            st.markdown("---")
            st.markdown("**🔢 Chọn Features Numeric** (giữ Ctrl/Cmd để chọn nhiều)")

            auto_num = auto_mapping.get('numeric', [])
            available_non_target = [c for c in all_cols if c != chosen_target]

            chosen_numeric = st.multiselect(
                "Features số (liên tục)",
                available_non_target,
                default=[c for c in auto_num if c in available_non_target],
                key="custom_numeric"
            )

            st.markdown("**🏷️ Chọn Features Categorical**")
            auto_cat = auto_mapping.get('categorical', [])
            remaining = [c for c in available_non_target if c not in chosen_numeric]

            chosen_categorical = st.multiselect(
                "Features phân loại",
                remaining,
                default=[c for c in auto_cat if c in remaining],
                key="custom_categorical"
            )

            # ── Summary trước khi train ──────────────────────────────────────
            total_features = len(chosen_numeric) + len(chosen_categorical)
            if total_features > 0:
                st.markdown("---")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Features numeric", len(chosen_numeric))
                sc2.metric("Features categorical", len(chosen_categorical))
                sc3.metric("Tổng features", total_features)
                sc4.metric("Rows dùng để train", f"{len(df_custom):,}")

            # ── Train button ─────────────────────────────────────────────────
            can_train = chosen_target and total_features > 0
            if not can_train:
                st.warning("⚠️ Chọn ít nhất 1 cột target và 1 feature để bắt đầu train.")

            if st.button("🚀 Train Model Ngay", type="primary", use_container_width=True,
                         disabled=not can_train, key="btn_train_custom"):
                mapping_custom = {
                    'target': chosen_target,
                    'numeric': chosen_numeric,
                    'categorical': chosen_categorical,
                    'ignore': [],
                }

                progress = st.progress(0, text="Đang chuẩn bị dữ liệu...")
                try:
                    progress.progress(20, "Đang tiền xử lý...")
                    import time; time.sleep(0.3)
                    progress.progress(40, "Đang cross-validate (5-fold)...")
                    result = train_universal(df_custom, mapping_custom, model_name)
                    progress.progress(85, "Đang lưu model...")
                    time.sleep(0.2)
                    progress.progress(100, "✅ Hoàn tất!")

                    m = result['metrics']

                    st.success(f"🎉 Train thành công! Model ID: `{result['model_id']}`")

                    # Metrics
                    st.markdown("### 📊 Kết Quả Đánh Giá Model")
                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.metric("AUC-ROC", f"{m['test_auc']:.4f}",
                               help="Khả năng phân biệt churn vs không churn. >0.8 là tốt.")
                    mc2.metric("F1 Score", f"{m['test_f1']:.4f}",
                               help="Cân bằng giữa Precision và Recall. >0.7 là tốt.")
                    mc3.metric("Accuracy", f"{m['test_accuracy']:.4f}")
                    mc4.metric("Precision", f"{m['test_precision']:.4f}",
                               help="Trong số dự đoán là churn, bao nhiêu % đúng")
                    mc5.metric("Recall", f"{m['test_recall']:.4f}",
                               help="Tìm được bao nhiêu % khách hàng churn thực sự")

                    # Cross-val
                    cv_c1, cv_c2 = st.columns(2)
                    cv_c1.metric(f"CV AUC ({m['cv_folds']}-fold)",
                                 f"{m['cv_auc_mean']:.4f} ± {m['cv_auc_std']:.4f}")
                    cv_c2.metric(f"CV F1 ({m['cv_folds']}-fold)",
                                 f"{m['cv_f1_mean']:.4f} ± {m['cv_f1_std']:.4f}")

                    # Feature importance chart
                    fi = result['feature_importance']
                    if fi:
                        st.markdown("### 🔍 Tầm Quan Trọng Đặc Trưng")
                        fi_df = pd.DataFrame(list(fi.items())[:12], columns=['Feature', 'Importance'])
                        fi_df = fi_df.sort_values('Importance')
                        st.bar_chart(fi_df.set_index('Feature'))

                    # Confusion matrix
                    cm = m['confusion_matrix']
                    st.markdown("### 🧮 Confusion Matrix (test set)")
                    cm_df = pd.DataFrame(
                        cm,
                        index=['Thực tế: Ở lại', 'Thực tế: Rời đi'],
                        columns=['Dự đoán: Ở lại', 'Dự đoán: Rời đi']
                    )
                    st.dataframe(cm_df.style.background_gradient(cmap='Blues'), use_container_width=False)

                    st.info(f"💾 Model đã được lưu tại `models/custom/{result['model_id']}.pkl`\n\n"
                            f"Chuyển sang tab **💾 Models Đã Lưu** để dự đoán với model này.")

                except Exception as e:
                    progress.empty()
                    st.error(f"❌ Lỗi khi train: {e}")
                    with st.expander("Chi tiết lỗi"):
                        st.code(traceback.format_exc())

        except Exception as e:
            st.error(f"❌ Không đọc được file: {e}")
            with st.expander("Chi tiết lỗi"):
                st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: MODELS ĐÃ LƯU — dự đoán với custom model
# ═════════════════════════════════════════════════════════════════════════════
with tab_saved:
    st.subheader("💾 Models Tùy Chỉnh Đã Train")

    saved_models = list_custom_models()

    if not saved_models:
        st.info("Chưa có model tùy chỉnh nào. Hãy upload CSV ở tab **🧩 Dataset Tùy Chỉnh** để train!")
    else:
        st.markdown(f"**{len(saved_models)} model** đã được train:")

        # Model selector
        model_labels = {
            m['model_id']: f"{m['dataset_name']} (AUC {m['metrics']['test_auc']:.3f}, "
                           f"F1 {m['metrics']['test_f1']:.3f}) — {m['trained_at'][:10]}"
            for m in saved_models
        }
        chosen_model_id = st.selectbox(
            "Chọn model để dự đoán",
            list(model_labels.keys()),
            format_func=lambda x: model_labels[x],
            key="saved_model_select"
        )

        if chosen_model_id:
            saved_model, saved_meta = load_custom_model(chosen_model_id)
            if not saved_model:
                st.error("Không load được model!")
            else:
                m_metrics = saved_meta.get('metrics', {})
                fi_saved  = saved_meta.get('feature_importance', {})
                num_feats  = saved_meta.get('numeric_features', [])
                cat_feats  = saved_meta.get('categorical_features', [])

                # Model summary card
                with st.expander("ℹ️ Thông tin model", expanded=False):
                    info_c1, info_c2 = st.columns(2)
                    with info_c1:
                        st.markdown(f"**Dataset:** {saved_meta.get('dataset_name')}")
                        st.markdown(f"**Target:** `{saved_meta.get('target_col')}`")
                        st.markdown(f"**Trained:** {saved_meta.get('trained_at','')[:19]}")
                        st.markdown(f"**Samples:** {m_metrics.get('total_samples',0):,} | Churn: {m_metrics.get('churn_rate',0)*100:.1f}%")
                    with info_c2:
                        st.markdown(f"**AUC-ROC:** {m_metrics.get('test_auc','?')}")
                        st.markdown(f"**F1:** {m_metrics.get('test_f1','?')}")
                        st.markdown(f"**Features:** {len(num_feats)} numeric + {len(cat_feats)} categorical")

                st.markdown("---")
                saved_mode = st.radio(
                    "Chế độ dự đoán",
                    ["🎯 Đơn lẻ (nhập tay)", "📦 Hàng loạt (upload CSV)"],
                    horizontal=True,
                    key="saved_mode"
                )

                # ── Dự đoán đơn lẻ ──────────────────────────────────────────
                if saved_mode == "🎯 Đơn lẻ (nhập tay)":
                    sv_col1, sv_col2 = st.columns([1, 1.2], gap="large")

                    with sv_col1:
                        st.markdown("**📝 Nhập giá trị từng cột**")
                        sv_inputs = {}

                        if num_feats:
                            st.markdown("*Numeric features:*")
                            nc = st.columns(min(3, len(num_feats)))
                            for i, feat in enumerate(num_feats):
                                with nc[i % len(nc)]:
                                    sv_inputs[feat] = st.number_input(feat, value=0.0, key=f"sv_num_{feat}_{chosen_model_id}")

                        if cat_feats:
                            st.markdown("*Categorical features:*")
                            cc = st.columns(min(2, len(cat_feats)))
                            for i, feat in enumerate(cat_feats):
                                with cc[i % len(cc)]:
                                    # Lấy danh sách giá trị từ training data nếu có
                                    sample_vals = saved_meta.get('categorical_samples', {}).get(feat, ['No', 'Yes'])
                                    sv_inputs[feat] = st.selectbox(feat, sample_vals, key=f"sv_cat_{feat}_{chosen_model_id}")

                        sv_predict_btn = st.button("🔮 Dự Đoán", type="primary",
                                                   use_container_width=True, key=f"sv_btn_{chosen_model_id}")

                    with sv_col2:
                        if sv_predict_btn:
                            try:
                                sv_df = pd.DataFrame([sv_inputs])
                                churn_prob = predict_universal(saved_model, saved_meta, sv_df)

                                # Lấy stats từ metadata nếu có
                                numeric_stats = saved_meta.get('numeric_stats', {})
                                recs = get_generic_recommendations(fi_saved, sv_inputs, churn_prob, numeric_stats)

                                st.markdown(render_gauge(churn_prob), unsafe_allow_html=True)
                                st.markdown(f"<div style='text-align:center;margin:.5rem 0'>{risk_badge(churn_prob)}</div>",
                                            unsafe_allow_html=True)
                                m1, m2 = st.columns(2)
                                m1.metric("Xác suất churn", f"{churn_prob*100:.1f}%")
                                m2.metric("Mức ưu tiên", recs['priority'].upper())

                                if fi_saved:
                                    st.markdown("**🔍 Top features quan trọng**")
                                    top_fi = dict(list(fi_saved.items())[:6])
                                    fi_chart = pd.DataFrame(list(top_fi.items()), columns=['Feature', 'Importance'])
                                    st.bar_chart(fi_chart.set_index('Feature'))

                                st.markdown("**💡 Đề xuất giữ chân**")
                                st.markdown(render_recs(recs), unsafe_allow_html=True)

                            except Exception as e:
                                st.error(f"❌ Lỗi: {e}")
                                with st.expander("Chi tiết"):
                                    st.code(traceback.format_exc())
                        else:
                            st.info("Nhập thông tin khách hàng và nhấn **Dự Đoán**")

                # ── Dự đoán hàng loạt ────────────────────────────────────────
                else:
                    sv_batch_file = st.file_uploader(
                        "Upload CSV (không cần cột target)",
                        type=["csv"],
                        key=f"sv_batch_{chosen_model_id}"
                    )

                    if sv_batch_file:
                        try:
                            sv_batch_df = pd.read_csv(sv_batch_file)
                            st.markdown(f"**{len(sv_batch_df):,} records**")
                            st.dataframe(sv_batch_df.head(3), use_container_width=True, hide_index=True)

                            # Kiểm tra cột
                            feature_cols = num_feats + cat_feats
                            missing_cols = [c for c in feature_cols if c not in sv_batch_df.columns]

                            if missing_cols:
                                st.error(f"❌ File thiếu cột: {missing_cols}")
                                st.markdown("**Cột cần có:** " + ", ".join(f"`{c}`" for c in feature_cols))
                            else:
                                if st.button("🚀 Dự Đoán Hàng Loạt", type="primary", key=f"sv_batch_btn_{chosen_model_id}"):
                                    with st.spinner("Đang xử lý..."):
                                        for col in num_feats:
                                            sv_batch_df[col] = pd.to_numeric(sv_batch_df[col], errors='coerce')

                                        proba_all = saved_model.predict_proba(sv_batch_df[feature_cols])
                                        sv_batch_df['Churn_Probability'] = proba_all[:, 1]
                                        sv_batch_df['Risk_Level'] = sv_batch_df['Churn_Probability'].apply(
                                            lambda p: '🔴 Cao' if p > 0.7 else ('🟡 TB' if p > 0.4 else '🟢 Thấp')
                                        )

                                    total = len(sv_batch_df)
                                    high  = int((sv_batch_df['Churn_Probability'] > 0.7).sum())
                                    med   = int(((sv_batch_df['Churn_Probability'] > 0.4) & (sv_batch_df['Churn_Probability'] <= 0.7)).sum())

                                    bc1, bc2, bc3 = st.columns(3)
                                    bc1.metric("Tổng", f"{total:,}")
                                    bc2.metric("🔴 Nguy cơ cao", f"{high:,}", f"{high/total*100:.1f}%")
                                    bc3.metric("🟡 Nguy cơ TB", f"{med:,}", f"{med/total*100:.1f}%")

                                    # Risk distribution
                                    st.markdown("**Phân tầng rủi ro**")
                                    risk_counts = sv_batch_df['Risk_Level'].value_counts()
                                    st.bar_chart(risk_counts)

                                    # Top high-risk
                                    top_n = min(20, high if high > 0 else 10)
                                    st.markdown(f"**Top {top_n} nguy cơ cao nhất**")
                                    st.dataframe(
                                        sv_batch_df.nlargest(top_n, 'Churn_Probability')
                                            .style.format({'Churn_Probability': '{:.1%}'}),
                                        use_container_width=True,
                                        hide_index=True
                                    )

                                    csv_out = sv_batch_df.to_csv(index=False)
                                    st.download_button(
                                        "💾 Tải kết quả CSV",
                                        csv_out,
                                        f"predictions_{chosen_model_id}.csv",
                                        "text/csv",
                                        use_container_width=True
                                    )

                        except Exception as e:
                            st.error(f"❌ {e}")
                            with st.expander("Chi tiết"):
                                st.code(traceback.format_exc())

        # ── Xóa model ────────────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("🗑️ Xóa model này"):
            if st.button("Xác nhận xóa", type="secondary", key=f"del_{chosen_model_id}"):
                try:
                    import glob
                    for f in glob.glob(os.path.join(BASE_DIR, "models", "custom", f"{chosen_model_id}*")):
                        os.remove(f)
                    st.success("Đã xóa model!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi xóa: {e}")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-size:.8rem'>"
    "🔮 ChurnIQ v3.0 | Powered by RandomForest + Streamlit | "
    "REST API: <code>python api.py</code>"
    "</div>",
    unsafe_allow_html=True
)
