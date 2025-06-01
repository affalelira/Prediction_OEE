import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------- KONFIGURASI --------------------
st.set_page_config(page_title="Sistem OEE", layout="wide")
APPS_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbxVMwkBqOj9v97hRd-gGXCPK9ODs65DNtu7OvCQVzh7kQTtCCOAwXWjNWbCtkUC6c8tEA/exec"

# -------------------- FUNGSI API --------------------
def fetch_data():
    try:
        response = requests.get(APPS_SCRIPT_URL, params={"action": "get_data"}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                header = data[0]
                rows = data[1:]
                df = pd.DataFrame(rows, columns=header)
                df.columns = df.columns.str.strip()
                return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
        return pd.DataFrame()

def add_data(payload):
    try:
        response = requests.post(APPS_SCRIPT_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": str(e)}

# -------------------- MODEL & ENCODER --------------------
model = joblib.load("model_risk_oee.joblib")
encoder = joblib.load("encoder_mesin.joblib")

# -------------------- SIDEBAR TABS --------------------
tab1, tab2 = st.tabs(["📝 Input Prediksi OEE", "📈 Dashboard Business Intelligence"])

# -------------------- TAB 1: FORM INPUT --------------------
with tab1:
    st.markdown(
        "<div style='background-color: #117864; padding: 1.5rem; border-radius: 10px; text-align: center;'>"
        "<h1 style='color: white;'>📊 Sistem Prediksi Risiko Penurunan Efisiensi Produksi</h1>"
        "<p style='color: white; font-size: 16px;'>Gunakan form di bawah ini untuk menghitung OEE dan mengirim data ke spreadsheet</p>"
        "</div>", unsafe_allow_html=True)

    if "prediksi_siap" not in st.session_state:
        st.session_state.prediksi_siap = False
        st.session_state.prediksi = {}
        st.session_state.input_data = {}
    
    with st.form("form_prediksi_oee"):    
        col1, col2, col3 = st.columns(3)
        with col1:
            tanggal = st.date_input('📅 Tanggal Produksi', value=date.today())
            shift = st.selectbox("🔄 Shift", [1, 2, 3], help="Pilih shift produksi")
            CT = st.number_input("⏱️ Cycle Time (CT)", min_value=0.0, help="Waktu siklus mesin (dalam detik)")
            Sch_Loss = st.number_input("📉 Schedule Loss", min_value=0.0)
            MPT = st.number_input("🧰 MPT (Minor Planned Time)", min_value=0.0)
            PDT = st.number_input("⚙️ PDT (Planned Downtime)", min_value=0.0)

        with col2:
            UPDT = st.number_input("⛔ UPDT (Unplanned Downtime)", min_value=0.0)
            AT = st.number_input("⌛ Available Time (AT)", min_value=0.0)
            Std_Output = st.number_input("📦 Standard Output", min_value=0.0)
            Act_Output = st.number_input("📈 Actual Output", min_value=0.0)
            Good_Out = st.number_input("✅ Good Output", min_value=0.0)

        with col3:
            BS = st.number_input("📉  Below Standart (BS)", min_value=0.0)
            mesin = st.selectbox("🏭 Nama Mesin", encoder.classes_, help="Pilih mesin yang digunakan")
            st.markdown("---")

        # Tombol submit
        prediksi_btn = st.form_submit_button("🔍 Prediksi Risiko OEE")

    if prediksi_btn:
        mesin_encoded = encoder.transform([mesin])[0]
        input_data = [[shift, CT, Sch_Loss, MPT, PDT, UPDT, AT, Std_Output, Act_Output, Good_Out, BS, mesin_encoded]]
        prediction = model.predict(input_data)[0]

        availability = round(max(0, min(100, ((AT / MPT * 100)))), 2) if AT else 0.0
        performance = round(max(0, min(100, (Act_Output / Std_Output * 100))), 2) if Std_Output else 0.0
        quality = round(max(0, min(100, (Good_Out / Act_Output * 100))), 2) if Act_Output else 0.0
        oee = round((availability * performance * quality) / 10000, 2)

        status_text = "⚠️ Risiko Penurunan Efisiensi" if prediction == 1 else "✅ Aman"
        status_color = "#E74C3C" if prediction == 1 else "#2ECC71"

        st.session_state.prediksi_siap = True
        st.session_state.prediksi = {
            "availability": availability,
            "performance": performance,
            "quality": quality,
            "oee": oee,
            "status_text": status_text,
            "status_color": status_color,
            "prediction": prediction
        }
        st.session_state.input_data = {
            "Tanggal": tanggal.strftime("%d/%m/%Y"),
            "Shift": shift,
            "Mesin": mesin,
            "CT": CT,
            "Sch_Loss": Sch_Loss,
            "MPT": MPT,
            "PDT": PDT,
            "UPDT": UPDT,
            "AT": AT,
            "Std_Output": Std_Output,
            "Act_Output": Act_Output,
            "Good_Out": Good_Out,
            "BS": BS
        }

    if st.session_state.prediksi_siap:
        pred = st.session_state.prediksi
        st.markdown("---")
        st.markdown(
            f"<div style='padding: 1rem; border-left: 5px solid {pred['status_color']}; background-color: #f8f9fa; border-radius: 6px;'>"
            f"<h3 style='color:{pred['status_color']}; margin-bottom: 0;'>{pred['status_text']}</h3>"
            f"<p style='margin-top: 5px;'>Berikut adalah komponen OEE hasil perhitungan dari input yang Anda berikan.</p>"
            "</div>", unsafe_allow_html=True)

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Availability", f"{pred['availability']:.2f} %")
        colB.metric("Performance", f"{pred['performance']:.2f} %")
        colC.metric("Quality", f"{pred['quality']:.2f} %")
        colD.metric("OEE", f"{pred['oee']:.2f} %")

        if st.button("📤 Simpan ke Spreadsheet"):
            payload = {
                "action": "add_data",
                "data": {
                    **st.session_state.input_data,
                    "AR": f"{pred['availability']:.2f}%",
                    "PR": f"{pred['performance']:.2f}%",
                    "QR": f"{pred['quality']:.2f}%",
                    "OEE": f"{pred['oee']:.2f}%",
                    "Status": "Beresiko" if pred['prediction'] == 1 else "Aman"
                }
            }
            result = add_data(payload)
            if result.get("status") == "success":
                st.success("✅ Data berhasil dikirim dan disimpan.")
                st.session_state.prediksi_siap = False
            else:
                st.warning(f"⚠️ Gagal menyimpan: {result.get('error', 'Tidak diketahui')}")

# -------------------- TAB 2: DASHBOARD --------------------
with tab2:
    st.markdown(
        "<div style='background-color: #1A5276; padding: 1.5rem; border-radius: 10px; text-align: center;'>"
        "<h1 style='color: white;'>📈 Business Intelligence Dashboard OEE Produksi</h1>"
        "<p style='color: white; font-size: 16px;'>Analisis visual efisiensi produksi berdasarkan data OEE</p>"
        "</div>", unsafe_allow_html=True)

    df = fetch_data()

    if not df.empty:
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], dayfirst=True)
        for col in ['%AR', '%PR', '%QR', '%OEE']:
            df[col] = df[col].str.replace('%', '', regex=False).str.replace(',', '.', regex=False).astype(float)

        num_cols = ['CT', 'Sch Loss', 'MPT', 'PDT', 'UPDT', 'AT', 'Std Output', 'Act Output', 'Good Out', 'BS']
        for col in num_cols:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.rename(columns={
            'TANGGAL': 'Tanggal',
            'SHIFT': 'Shift',
            'MESIN': 'Mesin',
            '%AR': 'AR',
            '%PR': 'PR',
            '%QR': 'QR',
            '%OEE': 'OEE',
            'STATUS': 'STATUS'
        })

        df['Tanggal_str'] = df['Tanggal'].dt.strftime('%d/%m/%Y')
        df = df.sort_values(by='Tanggal', ascending=False)
        cols = df.columns.tolist()
        cols.remove('Tanggal')
        cols.remove('Tanggal_str')
        new_cols = ['Tanggal_str'] + cols

        st.subheader("📄 Data Detail")
        st.dataframe(df[new_cols].rename(columns={'Tanggal_str': 'Tanggal'}), use_container_width=True)

        st.markdown("### 📅 Filter Data")
        with st.expander("Filter Berdasarkan Tanggal dan Mesin"):
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Dari", df['Tanggal'].min().date())
                end_date = st.date_input("Sampai", df['Tanggal'].max().date())
            with col2:
                mesin_filter = st.multiselect("Pilih Mesin", df["Mesin"].unique(), default=list(df["Mesin"].unique()))

        df_filtered = df[
            (df['Tanggal'].dt.date >= start_date) &
            (df['Tanggal'].dt.date <= end_date) &
            (df['Mesin'].isin(mesin_filter))
        ]

        if df_filtered.empty:
            st.warning("Data kosong untuk filter yang dipilih.")
        else:
            st.markdown("### 🔍 Rangkuman Statistika OEE")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rata-rata Availability", f"{df_filtered['AR'].mean():.2f} %")
            col2.metric("Rata-rata Performance", f"{df_filtered['PR'].mean():.2f} %")
            col3.metric("Rata-rata Quality", f"{df_filtered['QR'].mean():.2f} %")
            col4.metric("Rata-rata OEE", f"{df_filtered['OEE'].mean():.2f} %")

            st.markdown("### 📊 Visualisasi Data")
            df_filtered['Minggu'] = df_filtered['Tanggal'].dt.to_period('W').apply(lambda r: r.start_time)

            df_weekly = df_filtered.groupby(['Minggu', 'Mesin'], as_index=False)['OEE'].mean()

            col1, col2 = st.columns(2)
            with col1:
                fig_oee_bar = px.bar(df_weekly, x='Minggu', y='OEE', color='Mesin', barmode='group',
                                     title="📊 Rata-rata OEE Mingguan per Mesin")
                st.plotly_chart(fig_oee_bar, use_container_width=True)

            with col2:
                df_pivot = df_filtered.pivot_table(index='Mesin', columns='Tanggal', values='OEE')
                text = df_pivot.round(2).astype(str).values
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=df_pivot.values,
                    x=df_pivot.columns.strftime('%d-%m-%Y'),
                    y=df_pivot.index,
                    colorscale='Viridis',
                    colorbar=dict(title='OEE'),
                    text=text,
                    texttemplate="%{text}",
                    textfont={"size": 12, "color": "white"},
                ))
                fig_heatmap.update_layout(
                    title="📊 Heatmap OEE per Mesin dan Tanggal",
                    xaxis_title="Tanggal",
                    yaxis_title="Mesin",
                    yaxis_autorange='reversed'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

            col3, col4 = st.columns(2)
           
            with col3:
                df_quality = df_filtered.groupby('Mesin', as_index=False).agg({
                    'Good Out': 'sum',
                    'BS': 'sum'
                })
                df_quality['Total'] = df_quality['Good Out'] + df_quality['BS']
                df_quality['Persen Good'] = df_quality['Good Out'] / df_quality['Total'] * 100
                df_quality['Persen BS'] = df_quality['BS'] / df_quality['Total'] * 100

                fig_donut = px.pie(df_quality, names='Mesin', values='Persen Good',
                                title='🍬 Proporsi Produk Bagus per Mesin',
                                hole=0.4)
                st.plotly_chart(fig_donut, use_container_width=True)

            with col4:
                status_mesin = df_filtered.groupby(['Mesin', 'STATUS']).size().reset_index(name='Jumlah')
                color_map = {
                    "Aman": "green",
                    "Beresiko": "red"
                }
                fig_bar_status = px.bar(status_mesin, x='Mesin', y='Jumlah', color='STATUS',
                                        title="📉 Risiko Penurunan Efisiensi per Mesin",
                                        labels={"Jumlah": "Jumlah Status", "Mesin": "Mesin", "STATUS": "Status Risiko"},
                                        barmode='stack',
                                        color_discrete_map=color_map)
                st.plotly_chart(fig_bar_status, use_container_width=True)

            with st.container(): 
                loss_cols = ['PDT', 'UPDT', 'Sch Loss', 'MPT']
                loss_sum = df_filtered[loss_cols].sum().sort_values(ascending=False)

                df_pareto = loss_sum.reset_index()
                df_pareto.columns = ['Loss Type', 'Total Duration']
                df_pareto['Cumulative'] = df_pareto['Total Duration'].cumsum()
                df_pareto['Cumulative %'] = df_pareto['Cumulative'] / df_pareto['Total Duration'].sum() * 100

                fig_pareto = go.Figure()

                # Bar chart
                fig_pareto.add_trace(go.Bar(
                    x=df_pareto['Loss Type'],
                    y=df_pareto['Total Duration'],
                    name='Durasi Kerugian',
                    marker=dict(color='indianred')
                ))

                # Garis kumulatif
                fig_pareto.add_trace(go.Scatter(
                    x=df_pareto['Loss Type'],
                    y=df_pareto['Cumulative %'],
                    name='Garis Pareto (%)',
                    yaxis='y2',
                    mode='lines+markers',
                    line=dict(color='darkblue')
                ))

                # Garis horizontal di 80%
                fig_pareto.add_shape(
                    type='line',
                    x0=-0.5,
                    x1=len(df_pareto)-0.5,
                    y0=80,
                    y1=80,
                    yref='y2',
                    line=dict(color='gray', width=1.5, dash='dash')
                )

                # Anotasi threshold 80%
                fig_pareto.add_annotation(
                    x=0,
                    y=82,
                    yref='y2',
                    text="80% Threshold",
                    showarrow=False,
                    font=dict(size=12, color="gray")
                )

                # Layout
                fig_pareto.update_layout(
                    title='📉 Pareto Kerugian Waktu Produksi',
                    xaxis=dict(title='Kategori Loss'),
                    yaxis=dict(title='Durasi (menit)'),
                    yaxis2=dict(title='Akumulasi (%)', overlaying='y', side='right', range=[0, 110]),
                    legend=dict(x=0.01, y=0.99),
                    margin=dict(t=40, l=40, r=40, b=40)
                )

                st.plotly_chart(fig_pareto, use_container_width=True)
            
    else:
        st.info("Belum ada data yang tersedia.")
