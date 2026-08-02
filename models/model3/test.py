import datetime
import os
import streamlit as st
import pandas as pd
import clickhouse_connect
from dotenv import load_dotenv

load_dotenv() # Tải các biến môi trường từ file .env

from src.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_FEATURES_TABLE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_SECURE,
    CLICKHOUSE_USERNAME,
    MODEL_PATH
)
from src.predict import load_saved_model, predict_latest_signal

st.set_page_config(page_title="Test Model 3", page_icon="📈", layout="wide")

@st.cache_resource
def load_model():
    """Tải và cache model để không phải load lại mỗi lần đổi ngày."""
    return load_saved_model(MODEL_PATH)

def fetch_data_by_date(target_date: str, db_password: str) -> pd.DataFrame:
    """Truy vấn dữ liệu từ ClickHouse cho một ngày được chọn."""
    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USERNAME,
        password=db_password,
        database=CLICKHOUSE_DATABASE,
        secure=CLICKHOUSE_SECURE,
    )
    query = f"""
        SELECT * 
        FROM `{CLICKHOUSE_DATABASE}`.`{CLICKHOUSE_FEATURES_TABLE}`
        WHERE toDate(trading_date) = '{target_date}'
    """
    df = client.query_df(query)
    if not df.empty:
        df.columns = df.columns.str.strip()
    return df

st.title("🧪 Kiểm thử Kết Quả Mô Hình (Model 3)")
st.markdown("Ứng dụng kiểm thử tín hiệu giao dịch. Dữ liệu đầu vào sẽ được lọc sạch (đảm bảo phải đủ features) trước khi được đưa vào dự đoán.")

db_password = st.text_input(
    "🔑 Mật khẩu ClickHouse:", 
    type="password", 
    value=os.environ.get("CLICKHOUSE_PASSWORD", CLICKHOUSE_PASSWORD),
    help="Nếu có file .env chuẩn, mật khẩu sẽ tự động được điền."
)

# 1. Input: Chọn ngày giao dịch
selected_date = st.date_input(
    "📅 Chọn ngày giao dịch:",
    value=datetime.date(2026, 4, 30)
)

if st.button("🚀 Chạy dự đoán"):
    if not db_password:
        st.error("⚠️ Vui lòng nhập mật khẩu ClickHouse.")
        st.stop()
        
    with st.spinner("Đang tải dữ liệu và mô hình..."):
        try:
            model, features, signal_labels = load_model()
            
            date_str = selected_date.strftime("%Y-%m-%d")
            df = fetch_data_by_date(date_str, db_password)
            
            if df.empty:
                st.warning(f"⚠️ Không tìm thấy dữ liệu giao dịch cho ngày {date_str}.")
            else:
                st.info(f"📊 Đã lấy được {len(df)} mã cổ phiếu (dữ liệu thô) cho ngày {date_str}.")
                
                # Yêu cầu: "đầu vào chuẩn là phải đầy đủ feature" -> Drop toàn bộ hàng chứa NaN ở features
                df_clean = df.dropna(subset=features).copy()
                dropped_count = len(df) - len(df_clean)
                
                if dropped_count > 0:
                    st.warning(f"🧹 Đã loại bỏ {dropped_count} mã cổ phiếu do thiếu dữ liệu features (không đạt chuẩn).")
                
                if df_clean.empty:
                    st.error("❌ Sau khi lọc các mã thiếu features, không còn dữ liệu hợp lệ để dự đoán.")
                else:
                    st.success(f"✅ Bắt đầu dự đoán trên {len(df_clean)} mã cổ phiếu đạt chuẩn đầy đủ features.")
                    
                    # 2. Output: Dự đoán và xuất kết quả
                    predictions_df = predict_latest_signal(
                        df=df_clean, model=model, features=features, signal_labels=signal_labels
                    )
                    
                    display_cols = ["symbol", "close", "adjusted_signal", "buy_probability", "sell_probability", "hold_probability"]
                    available_cols = [col for col in display_cols if col in predictions_df.columns]
                    
                    st.dataframe(predictions_df[available_cols].sort_values("buy_probability", ascending=False), use_container_width=True)
                    
        except Exception as e:
            st.error(f"🚨 Đã xảy ra lỗi hệ thống: {e}")