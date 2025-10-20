import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Belajar Klasifikasi Lemon",
    page_icon=":lemon:")

model = joblib.load("model_klasifikasi_lemon.joblib")

st.title(":lemon: Belajar Klasifikasi Lemon")
st.markdown("Aplikasi machine learning classification untuk memprediksi kualitas lemon")

diameter = st.slider("Diameter", 4.0, 10.0, 6.5)
berat = st.slider("Berat", 100.0, 250.0, 210.0)
tebal_kulit = st.slider("Tebal Kulit", 0.2, 1.0, 0.8)
kadar_gula = st.slider("Kadar Gula", 8.0, 14.0, 12.0)
asal_daerah = st.pills("Asal Daerah", ["California", "Malang", "Medan"], default="California")
warna = st.pills("Warna", ["Hijau pekat", "Kuning kehijauan", "Kuning cerah"], default="Hijau pekat")
musim_panen = st.pills("Musim Panen", ["awal", "puncak", "akhir"], default="awal")

if st.button("Prediksi", type="primary"):
    data_baru = pd.DataFrame(
        [[diameter, berat, tebal_kulit, kadar_gula, asal_daerah, musim_panen, warna]],
        columns=["diameter", "berat", "tebal_kulit", "kadar_gula", "asal_daerah", "musim_panen", "warna"]
    )
    prediksi = model.predict(data_baru)[0]
    presentase = max(model.predict_proba(data_baru)[0])
    st.success(f"Model memprediksi *{prediksi}* dengan tingkat keyakinan *{presentase*100:.2f}%*")
    st.balloons()
    st.snow()

st.divider()
st.caption("Dibuat dengan :lemon: oleh *Toni Kurniawan*")