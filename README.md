# 🔮 ChurnIQ — Hệ Thống Dự Đoán Khách Hàng Rời Đi v2.0

Hệ thống dự đoán churn đa dataset với giao diện Streamlit hiện đại, REST API đầy đủ, và đề xuất giải pháp giữ chân khách hàng thông minh.

---

## 📁 Cấu Trúc Dự Án

```
churn-webapp/
├── app.py              # Giao diện Streamlit (3 tabs)
├── api.py              # REST API (Flask)
├── train_model.py      # Huấn luyện model + đánh giá
├── datasets_config.py  # Cấu hình datasets + đề xuất giải pháp
├── app.yaml            # Databricks app config
├── requirements.txt
├── models/
│   ├── churn_model_telco.pkl
│   ├── churn_model_calls.pkl
│   ├── churn_model_telco_metrics.json   # (sinh ra sau khi train)
│   └── churn_model_calls_metrics.json
└── data/
    ├── Telco_customer_churn.csv
    └── Churn.csv
```

---

## 🚀 Chạy Ứng Dụng

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models (nếu chưa có)
```bash
python train_model.py
```

### 3. Chạy giao diện Streamlit
```bash
streamlit run app.py
```

### 4. Chạy REST API (cổng 8001)
```bash
python api.py
# hoặc đổi cổng:
API_PORT=9000 python api.py
```

---

## 🌐 REST API Endpoints

### Health Check
```
GET /health
```

### Danh sách models
```
GET /api/models
```

### Dự đoán đơn lẻ
```
POST /api/predict
Content-Type: application/json

{
  "dataset_type": "telco_ibm",
  "features": {
    "tenure": 2,
    "MonthlyCharges": 70.7,
    "TotalCharges": 151.65,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "InternetService": "Fiber optic",
    "TechSupport": "No",
    "OnlineSecurity": "No"
  }
}
```

**Response:**
```json
{
  "churn_probability": 0.88,
  "churn_probability_pct": "88.0%",
  "is_churn": true,
  "risk_level": "HIGH",
  "feature_importance": { "TotalCharges": 0.28, "MonthlyCharges": 0.25, ... },
  "retention": {
    "priority": "high",
    "recommendations": [
      {
        "action": "📋 Nâng cấp lên hợp đồng dài hạn",
        "detail": "Đề xuất hợp đồng 1-2 năm với ưu đãi giảm 10-15%",
        "impact": "Cao",
        "effort": "Thấp"
      }
    ]
  }
}
```

### Dự đoán hàng loạt (JSON)
```
POST /api/predict/batch
Content-Type: application/json

{
  "dataset_type": "telco_ibm",
  "records": [ {...}, {...} ]
}
```

### Dự đoán từ file CSV
```
POST /api/predict/csv
Content-Type: multipart/form-data
file: <your_file.csv>
```

---

## 📊 Datasets Hỗ Trợ

### Telco IBM (`telco_ibm`)
| Cột | Kiểu | Mô tả |
|-----|------|--------|
| tenure | số | Số tháng sử dụng dịch vụ |
| MonthlyCharges | số | Phí hàng tháng ($) |
| TotalCharges | số | Tổng phí đã trả ($) |
| Contract | phân loại | Month-to-month / One year / Two year |
| PaymentMethod | phân loại | Electronic check / Mailed check / ... |
| InternetService | phân loại | DSL / Fiber optic / No |
| TechSupport | phân loại | No / Yes / No internet service |
| OnlineSecurity | phân loại | No / Yes / No internet service |

### Call Details BigML (`call_details`)
| Cột | Kiểu | Mô tả |
|-----|------|--------|
| Account length | số | Số ngày có tài khoản |
| Total day minutes | số | Phút gọi ban ngày |
| Total eve minutes | số | Phút gọi buổi tối |
| Total night minutes | số | Phút gọi ban đêm |
| Total intl minutes | số | Phút gọi quốc tế |
| Number vmail messages | số | Số tin nhắn voicemail |
| Customer service calls | số | Số lần gọi CSKH |
| International plan | phân loại | No / Yes |
| Voice mail plan | phân loại | No / Yes |

---

## 🔄 Cải Tiến So Với v1.0

| Vấn đề cũ | Giải pháp v2.0 |
|------------|----------------|
| Không có API | REST API đầy đủ (Flask) với 4 endpoints |
| Giao diện thô sơ | 3 tabs: Đơn lẻ / Hàng loạt / Analytics |
| Không có charts | Bar charts feature importance, distribution, risk segmentation |
| Đề xuất chung chung | 10+ đề xuất cụ thể theo từng trường hợp |
| Model path relative | Absolute path — hoạt động đúng trên Databricks |
| CSV gốc lỗi cột | Auto-rename columns (AccountLength → Account length) |
| Không có metrics | Cross-val 5-fold, AUC-ROC, F1, Precision, Recall, Confusion Matrix |
| RandomForest mặc định | class_weight='balanced' xử lý mất cân bằng class |

---

## 🏗️ Deploy Trên Databricks

1. Upload toàn bộ folder lên Databricks workspace
2. Cài requirements: `%pip install -r requirements.txt`
3. Train model (nếu cần): `%run train_model`
4. Tạo Databricks App với `app.yaml`
5. Truy cập URL được cấp bởi Databricks

---

## 📝 License
MIT
