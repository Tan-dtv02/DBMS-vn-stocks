import streamlit as st
import pandas as pd
import joblib
import clickhouse_connect
from pathlib import Path

# =========================
# CONFIG
# =========================
CLICKHOUSE_HOST = "cvzq3t560s.ap-southeast-1.aws.clickhouse.cloud"
CLICKHOUSE_PORT = 8443
CLICKHOUSE_USER = "default"
CLICKHOUSE_PASSWORD = "K5clN_57i9pu6"
CLICKHOUSE_DATABASE = "stock"
CLICKHOUSE_SECURE = True

TABLE_NAME = "features_all"

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "future_return_lgbm.pkl"

DATE_COL = "trading_date"
SYMBOL_COL = "symbol"
TARGET_COL = "future_return_5d"

DROP_COLS = [
    DATE_COL,
    SYMBOL_COL,
    TARGET_COL,
    "companyname",
    "sector"
]

# LOAD MODEL
@st.cache_resource
def load_model():
  return joblib.load(MODEL_PATH)


@st.cache_resource
def get_client():
  return clickhouse_connect.get_client(
    host=CLICKHOUSE_HOST,
    port=CLICKHOUSE_PORT,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASSWORD,
    database=CLICKHOUSE_DATABASE,
    secure=CLICKHOUSE_SECURE
  )


# QUERY DATA
def get_feature_by_symbol_date(symbol, trading_date):
  client = get_client()

  query = f"""
    SELECT *
    FROM {TABLE_NAME}
    WHERE upper(trim(symbol)) = upper(trim('{symbol}'))
      AND toDate({DATE_COL}) = toDate('{trading_date}')
    LIMIT 1
  """

  return client.query_df(query)


def get_latest_dates(symbol):
  client = get_client()

  query = f"""
    SELECT DISTINCT {DATE_COL}
    FROM {TABLE_NAME}
    WHERE upper(trim(symbol)) = upper(trim('{symbol}'))
    ORDER BY {DATE_COL} DESC
    LIMIT 10
  """

  return client.query_df(query)

def get_actual_return_5_sessions(symbol, trading_date):
  client = get_client()

  query = f"""
    SELECT
      trading_date,
      close
    FROM {TABLE_NAME}
    WHERE upper(trim(symbol)) = upper(trim('{symbol}'))
      AND toDate(trading_date) >= toDate('{trading_date}')
    ORDER BY trading_date ASC
    LIMIT 6
  """

  df_price = client.query_df(query)

  # Cần đủ: phiên hiện tại + 5 phiên sau = 6 dòng
  if len(df_price) < 6:
    return None, None

  close_current = float(df_price["close"].iloc[0])
  close_after_5_sessions = float(df_price["close"].iloc[5])
  actual_date = df_price["trading_date"].iloc[5]

  actual_return = close_after_5_sessions / close_current - 1

  return actual_return, actual_date

# =========================
# PREPARE INPUT
# =========================
def prepare_input(df):
  drop_cols = [
    DATE_COL,
    SYMBOL_COL,
    TARGET_COL,
    "companyname",
    "sector"
  ]

  X = df.drop(
    columns=[col for col in drop_cols if col in df.columns],
    errors="ignore"
  )

  X = X.select_dtypes(include=["int64", "float64", "int32", "float32"])
  X = X.fillna(0)

  return X


def get_signal(predicted_return):
  if predicted_return > 0.03:
    return "Kỳ vọng tăng mạnh"
  elif predicted_return > 0:
    return "Kỳ vọng tăng nhẹ"
  elif predicted_return > -0.03:
    return "Trung tính / giảm nhẹ"
  else:
    return "Rủi ro giảm mạnh"


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
  page_title="Model2 - Future Return 5 phiên",
  layout="wide"
)

st.title("Demo Model2 - Dự đoán lợi suất cổ phiếu sau 5 phiên giao dịch")

st.markdown("""
Model2 nhận đầu vào là **mã cổ phiếu** và **ngày giao dịch**.  
Hệ thống sẽ lấy feature tương ứng từ ClickHouse và dự đoán `future_return_5d`.

Lưu ý: `future_return_5d` nghĩa là **lợi suất sau 5 phiên giao dịch tiếp theo**, không phải 5 ngày lịch.
""")

col1, col2 = st.columns(2)

with col1:
  symbol = st.text_input("Nhập mã cổ phiếu", value="ACB")

with col2:
  trading_date = st.date_input("Chọn ngày giao dịch")


if st.button("Dự đoán"):
  symbol = symbol.upper().strip()
  date_str = trading_date.strftime("%Y-%m-%d")

  model = load_model()
  df = get_feature_by_symbol_date(symbol, date_str)

  if df.empty:
    st.error("Không tìm thấy dữ liệu feature cho symbol và ngày đã chọn.")

    latest_dates = get_latest_dates(symbol)

    if not latest_dates.empty:
      st.warning("Các ngày gần nhất có dữ liệu của mã này:")
      st.dataframe(latest_dates)
    else:
      st.warning("Không tìm thấy mã cổ phiếu này trong bảng features_all.")

  else:
      X = prepare_input(df)

      predicted_return = model.predict(X)[0]

      st.subheader("Kết quả dự đoán Model2")

      col1, col2, col3 = st.columns(3)

      with col1:
        st.metric("Mã cổ phiếu", symbol)

      with col2:
        st.metric("Ngày dự đoán", date_str)

      with col3:
        st.metric(
          "Predicted Return sau 5 phiên",
          f"{predicted_return * 100:.2f}%"
        )

      # =========================
      # ACTUAL RETURN
      # =========================
      actual_return, actual_date = get_actual_return_5_sessions(symbol, date_str)

      if actual_return is not None:
          st.subheader("So sánh với kết quả thực tế sau 5 phiên")

          error = predicted_return - actual_return
          abs_error = abs(error)

          col1, col2, col3, col4 = st.columns(4)

          with col1:
            st.metric(
              "Actual Return sau 5 phiên",
              f"{actual_return * 100:.2f}%"
            )

          with col2:
            st.metric(
              "Ngày phiên thứ 5",
              str(actual_date)
            )

          with col3:
            st.metric(
              "Sai số",
              f"{error * 100:.2f}%"
            )

          with col4:
            st.metric(
              "Sai số tuyệt đối",
              f"{abs_error * 100:.2f}%"
            )

          if predicted_return * actual_return > 0:
            st.success("Model dự đoán đúng chiều tăng/giảm.")
          else:
            st.warning("Model dự đoán sai chiều tăng/giảm.")

      else:
        st.info(
          "Ngày này chưa đủ 5 phiên giao dịch sau đó trong dữ liệu, "
          "nên chỉ hiển thị kết quả dự đoán của model."
        )

      # =========================
      # SIGNAL
      # =========================
      signal = get_signal(predicted_return)
      st.info(f"Nhận định: {signal}")

      with st.expander("Xem feature đầu vào model"):
        st.dataframe(X)

      with st.expander("Xem dữ liệu gốc lấy từ ClickHouse"):
        st.dataframe(df)