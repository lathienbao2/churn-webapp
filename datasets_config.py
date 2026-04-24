"""
Cấu hình dataset + helper functions cho Churn Prediction System
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_CONFIGS = {
    'telco_ibm': {
        'name': 'Telco Customer Churn (IBM)',
        'required_columns': [
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'Contract', 'PaymentMethod', 'InternetService',
            'TechSupport', 'OnlineSecurity'
        ],
        'numeric_features': ['tenure', 'MonthlyCharges', 'TotalCharges'],
        'categorical_features': ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport', 'OnlineSecurity'],
        'model_path': os.path.join(BASE_DIR, 'models', 'churn_model_telco.pkl'),
        'feature_labels': {
            'tenure': 'Thời gian sử dụng (tháng)',
            'MonthlyCharges': 'Phí hàng tháng ($)',
            'TotalCharges': 'Tổng phí ($)',
            'Contract': 'Loại hợp đồng',
            'PaymentMethod': 'Phương thức thanh toán',
            'InternetService': 'Dịch vụ Internet',
            'TechSupport': 'Hỗ trợ kỹ thuật',
            'OnlineSecurity': 'Bảo mật trực tuyến',
        },
        'categorical_options': {
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'TechSupport': ['No', 'Yes', 'No internet service'],
            'OnlineSecurity': ['No', 'Yes', 'No internet service'],
        },
        'numeric_defaults': {
            'tenure': 24.0, 'MonthlyCharges': 65.0, 'TotalCharges': 1500.0
        }
    },
    'call_details': {
        'name': 'Call Details Churn (BigML)',
        'required_columns': [
            'Account length', 'Total day minutes', 'Total eve minutes',
            'Total night minutes', 'Total intl minutes', 'Number vmail messages',
            'Customer service calls', 'International plan', 'Voice mail plan'
        ],
        'numeric_features': [
            'Account length', 'Total day minutes', 'Total eve minutes',
            'Total night minutes', 'Total intl minutes', 'Number vmail messages',
            'Customer service calls'
        ],
        'categorical_features': ['International plan', 'Voice mail plan'],
        'model_path': os.path.join(BASE_DIR, 'models', 'churn_model_calls.pkl'),
        'feature_labels': {
            'Account length': 'Thời gian tài khoản (ngày)',
            'Total day minutes': 'Phút gọi ban ngày',
            'Total eve minutes': 'Phút gọi buổi tối',
            'Total night minutes': 'Phút gọi ban đêm',
            'Total intl minutes': 'Phút gọi quốc tế',
            'Number vmail messages': 'Số tin nhắn voicemail',
            'Customer service calls': 'Số lần gọi CSKH',
            'International plan': 'Gói quốc tế',
            'Voice mail plan': 'Gói voicemail',
        },
        'categorical_options': {
            'International plan': ['No', 'Yes'],
            'Voice mail plan': ['No', 'Yes'],
        },
        'column_rename': {
            'AccountLength': 'Account length',
            'VMailMessage': 'Number vmail messages',
            'DayMins': 'Total day minutes',
            'EveMins': 'Total eve minutes',
            'NightMins': 'Total night minutes',
            'IntlMins': 'Total intl minutes',
            'CustServCalls': 'Customer service calls',
            'IntlPlan': 'International plan',
            'VMailPlan': 'Voice mail plan'
        },
        'numeric_defaults': {
            'Account length': 100.0, 'Total day minutes': 200.0, 'Total eve minutes': 180.0,
            'Total night minutes': 150.0, 'Total intl minutes': 10.0,
            'Number vmail messages': 5.0, 'Customer service calls': 2.0
        }
    }
}


def detect_dataset_type(df):
    """Tự động phát hiện loại dataset từ các cột"""
    columns = set(df.columns)
    if 'tenure' in columns and 'MonthlyCharges' in columns:
        return 'telco_ibm'
    if 'Account length' in columns and 'Total day minutes' in columns:
        return 'call_details'
    # Try after rename
    renamed = set()
    for old, new in DATASET_CONFIGS['call_details']['column_rename'].items():
        if old in columns:
            renamed.add(new)
    if 'Account length' in renamed and 'Total day minutes' in renamed:
        return 'call_details'
    return None


def get_retention_recommendations(dataset_type, features, churn_prob):
    """
    Tạo đề xuất giữ chân khách hàng dựa trên đặc điểm và xác suất churn
    """
    recommendations = []
    priority = "high" if churn_prob > 0.7 else ("medium" if churn_prob > 0.4 else "low")

    if dataset_type == 'telco_ibm':
        contract = features.get('Contract', '')
        tenure = float(features.get('tenure', 0))
        monthly = float(features.get('MonthlyCharges', 0))
        security = features.get('OnlineSecurity', '')
        tech = features.get('TechSupport', '')
        internet = features.get('InternetService', '')

        if contract == 'Month-to-month':
            recommendations.append({
                'action': '📋 Nâng cấp lên hợp đồng dài hạn',
                'detail': 'Đề xuất hợp đồng 1-2 năm với ưu đãi giảm 10-15% phí hàng tháng',
                'impact': 'Cao',
                'effort': 'Thấp'
            })
        if tenure < 12:
            recommendations.append({
                'action': '🎁 Chương trình loyalty cho khách hàng mới',
                'detail': 'Tặng tháng miễn phí hoặc dịch vụ bổ sung để tăng gắn kết trong 12 tháng đầu',
                'impact': 'Cao',
                'effort': 'Thấp'
            })
        if monthly > 80:
            recommendations.append({
                'action': '💰 Tối ưu hóa gói cước',
                'detail': f'Xem xét lại gói cước phù hợp hơn. Hiện tại: ${monthly:.0f}/tháng — có thể tiết kiệm 10-20%',
                'impact': 'Trung bình',
                'effort': 'Thấp'
            })
        if security == 'No' and internet != 'No':
            recommendations.append({
                'action': '🔒 Kích hoạt Online Security miễn phí 3 tháng',
                'detail': 'Khách hàng dùng Internet mà chưa có bảo mật — tặng thử để tăng giá trị gói',
                'impact': 'Trung bình',
                'effort': 'Thấp'
            })
        if tech == 'No':
            recommendations.append({
                'action': '🛠️ Hỗ trợ kỹ thuật ưu tiên',
                'detail': 'Cung cấp dịch vụ TechSupport miễn phí 1 tháng để giảm friction kỹ thuật',
                'impact': 'Trung bình',
                'effort': 'Thấp'
            })

    elif dataset_type == 'call_details':
        csc = float(features.get('Customer service calls', 0))
        day_mins = float(features.get('Total day minutes', 0))
        intl_plan = features.get('International plan', 'No')
        intl_mins = float(features.get('Total intl minutes', 0))

        if csc >= 4:
            recommendations.append({
                'action': '📞 Chăm sóc khách hàng chủ động',
                'detail': f'Khách hàng đã gọi CSKH {csc:.0f} lần — cần được liên hệ proactively để giải quyết triệt để vấn đề',
                'impact': 'Rất cao',
                'effort': 'Trung bình'
            })
        if day_mins > 250:
            recommendations.append({
                'action': '📱 Nâng cấp gói gọi ban ngày',
                'detail': f'Sử dụng {day_mins:.0f} phút/ngày — đề xuất gói không giới hạn để tránh phí phụ thu',
                'impact': 'Cao',
                'effort': 'Thấp'
            })
        if intl_plan == 'No' and intl_mins > 15:
            recommendations.append({
                'action': '🌍 Đăng ký gói quốc tế',
                'detail': f'Gọi quốc tế {intl_mins:.0f} phút nhưng chưa có gói — tiết kiệm chi phí đáng kể khi đăng ký',
                'impact': 'Cao',
                'effort': 'Rất thấp'
            })
        if csc == 0:
            recommendations.append({
                'action': '📊 Chủ động check-in định kỳ',
                'detail': 'Khách hàng chưa từng gọi CSKH — gửi survey hài lòng để phát hiện vấn đề tiềm ẩn',
                'impact': 'Thấp',
                'effort': 'Thấp'
            })

    if not recommendations:
        recommendations.append({
            'action': '💌 Chương trình chăm sóc tiêu chuẩn',
            'detail': 'Gửi email/SMS cảm ơn và khảo sát mức độ hài lòng định kỳ hàng quý',
            'impact': 'Thấp',
            'effort': 'Rất thấp'
        })

    return {'priority': priority, 'recommendations': recommendations}
