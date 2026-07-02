
import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import calendar
from datetime import datetime

# 1. Cấu hình giao diện Web
st.set_page_config(page_title="Superstore Sales Forecast Dashboard", layout="wide")

st.title("📊 Hệ Thống Dự Báo Doanh Thu Siêu Thị Toàn Diện")
st.subheader("Mô hình nâng cấp: SARIMAX & Bộ chọn thời gian tương lai thông minh")
st.markdown("Ứng dụng tự động huấn luyện dữ liệu gốc và phân tách doanh thu đến từng ngày bất kỳ bạn chọn.")

# 2. Đọc trực tiếp và tiền xử lý dữ liệu gốc
try:
    df_raw = pd.read_csv('superstore_sales.csv')
    df_raw['Order Date'] = pd.to_datetime(df_raw['Order Date'], dayfirst=True)
    monthly_sales = df_raw.resample('MS', on='Order Date')['Sales'].sum().reset_index()
    monthly_sales.columns = ['ds', 'y']
    
    history_dates = pd.to_datetime(monthly_sales['ds'])
    history_sales = monthly_sales['y'].values
    
    # Mốc thời gian cuối cùng của dữ liệu lịch sử thô (Tháng 12/2018)
    last_date = history_dates.iloc[-1]
    min_future_date = last_date + pd.offsets.MonthBegin(1)

    # =========================================================
    # GIAO DIỆN THANH BÊN (SIDEBAR) - CHỌN NGÀY/THÁNG/NĂM TÁCH RỜI
    # =========================================================
    st.sidebar.header("⚙️ Chọn Thời Gian Dự Báo")
    st.sidebar.write(f"📅 *Dữ liệu lịch sử kết thúc: {last_date.strftime('%d-%m-%Y')}*")
    
    # Tạo danh sách các năm từ năm kế tiếp đến năm 2040 để chọn nhanh
    list_years = list(range(min_future_date.year, 2041))
    chosen_year = st.sidebar.selectbox("1️⃣ Chọn Năm:", list_years, index=2) # Mặc định chọn năm thứ 3 trong list
    
    # Chọn tháng
    list_months = list(range(1, 13))
    chosen_month = st.sidebar.selectbox("2️⃣ Chọn Tháng:", list_months, index=datetime.now().month - 1)
    
    # Tính số ngày tối đa của tháng/năm đã chọn để tạo danh sách ngày hợp lệ
    max_days = calendar.monthrange(chosen_year, chosen_month)[1]
    list_days = list(range(1, max_days + 1))
    chosen_day = st.sidebar.selectbox("3️⃣ Chọn Ngày:", list_days, index=0)

    # Khởi tạo datetime từ các giá trị đã chọn
    chosen_datetime = pd.to_datetime(f"{chosen_year}-{chosen_month}-{chosen_day}")
    
    # Kiểm tra nếu ngày chọn nhỏ hơn ngày bắt đầu dự báo hợp lệ
    if chosen_datetime < min_future_date:
        st.sidebar.warning(f"⚠️ Vui lòng chọn mốc thời gian sau tháng {last_date.strftime('%m-%Y')}")
        chosen_datetime = min_future_date
        chosen_year, chosen_month, chosen_day = min_future_date.year, min_future_date.month, min_future_date.day

    # Tính toán chính xác số tháng cần dự báo từ mốc cuối (2018) đến thời gian được chọn
    thoi_gian_du_bao = ((chosen_datetime.year - last_date.year) * 12 + (chosen_datetime.month - last_date.month))
    if thoi_gian_du_bao <= 0:
        thoi_gian_du_bao = 1

    # 3. Huấn luyện thuật toán SARIMAX chạy dài hạn
    with st.spinner(f"Mô hình SARIMAX đang tính toán chuỗi ngày đến năm {chosen_datetime.year}..."):
        model_sarimax = sm.tsa.statespace.SARIMAX(
            history_sales,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model_sarimax.fit(disp=False)
        
        # Dự báo toàn bộ các tháng tương lai kéo dài tới mốc đã chọn
        future_pred = results.forecast(steps=thoi_gian_du_bao)
        future_dates = pd.date_range(start=min_future_date, periods=thoi_gian_du_bao, freq='MS')

        # Tạo bảng kết quả lưu trữ
        forecast_df = pd.DataFrame({
            'Month_Raw': future_dates,
            'Tháng / Năm': future_dates.strftime('%m-%Y'),
            'Doanh Thu Tháng ($)': future_pred
        })

    # =========================================================
    # TRÍCH XUẤT KẾT QUẢ DỰ BÁO RIÊNG CHO NGÀY ĐƯỢC CHỌN
    # =========================================================
    target_month_data = forecast_df[
        (forecast_df['Month_Raw'].dt.year == chosen_datetime.year) & 
        (forecast_df['Month_Raw'].dt.month == chosen_datetime.month)
    ]
    
    st.markdown("---")
    st.markdown("### 🎯 Kết Quả Dự Báo Đặc Biệt")
    
    if not target_month_data.empty:
        total_month_sales = target_month_data['Doanh Thu Tháng ($)' ].values[0]
        days_in_month = calendar.monthrange(chosen_datetime.year, chosen_datetime.month)[1]
        predicted_day_sales = total_month_sales / days_in_month
        
        # Hiển thị số liệu dạng thẻ lớn
        c_kpi1, c_kpi2 = st.columns(2)
        with c_kpi1:
            st.metric(
                label=f"🔮 Doanh thu ước tính riêng ngày: {chosen_datetime.strftime('%d-%m-%Y')}", 
                value=f"${predicted_day_sales:,.2f}"
            )
        with c_kpi2:
            st.metric(
                label=f"🏢 Tổng doanh thu cả tháng {chosen_datetime.strftime('%m-%Y')}", 
                value=f"${total_month_sales:,.2f}"
            )
    
    st.markdown("---")

    # 4. Hiển thị đồ thị trực quan và bảng số liệu
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write(f"### 📈 Biểu đồ dữ liệu & Đường dự báo đến năm {chosen_datetime.year}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history_dates, history_sales, label='Dữ liệu quá khứ (Historical)', color='#0047AB', linewidth=2)
        ax.plot(future_dates, future_pred, label='Đường dự báo tương lai (SARIMAX)', color='#FF0000', linestyle='--', linewidth=2)
        ax.scatter(chosen_datetime, total_month_sales, color='black', s=100, zorder=5, label='Mốc thời gian bạn chọn')
        
        ax.set_xlabel('Thời gian (Năm)')
        ax.set_ylabel('Doanh thu ($)')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    with col2:
        st.write("### 📋 Lịch trình doanh thu tương lai ($)")
        df_display = forecast_df.copy()
        df_display['Doanh Thu Tháng ($)'] = df_display['Doanh Thu Tháng ($)'].map(lambda x: f"${x:,.2f}")
        st.dataframe(df_display[['Tháng / Năm', 'Doanh Thu Tháng ($)']], use_container_width=True, height=400)

except FileNotFoundError:
    st.error("Không tìm thấy file 'superstore_sales.csv' trong thư mục hiện tại. Hãy kiểm tra lại tên file!")