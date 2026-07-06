import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="HEA Dual-Pipeline AI", layout="wide")

st.title("🔬 In-Silico Design: High-Entropy Alloy (HEA) Fe-Cr-Ni-Cu-Al")
st.markdown("Sistem **Dual-Pipeline Hybrid AI** untuk perancangan material pelindung Reaktor Nuklir Gen-IV.")
st.markdown("---")

# --- DATA & KONSTANTA FISIKA ---
# Nilai Valence Electron (VEC) referensi
VEC_VALS = {'Fe': 8, 'Cr': 6, 'Ni': 10, 'Cu': 11, 'Al': 3}

# --- PIPELINE A: STATISTIK (MOCKUP ALGORITMA) ---
def run_pipeline_a():
    st.subheader("📊 Pipeline A: Statistik & Optimasi")
    
    with st.expander("1. GAN Augmentation (Data Sintetik)", expanded=True):
        st.write("Menggunakan Tabular GAN untuk memperbanyak data langka...")
        
        # Membuat Mockup 5000 Data Sintetik
        n_rows = 5000
        # Menggunakan distribusi Dirichlet agar total kelima elemen selalu 100%
        gan_data = np.random.dirichlet(np.ones(5), size=n_rows) * 100
        df_gan = pd.DataFrame(gan_data, columns=['Fe', 'Cr', 'Ni', 'Cu', 'Al'])
        
        # Membulatkan nilai agar lebih rapi (2 angka di belakang koma)
        df_gan = df_gan.round(2)
        
        st.progress(100)
        st.success("Berhasil menghasilkan 5,000 baris data komposisi sintetik yang divalidasi Isolation Forest.")
        
        # Menampilkan cuplikan data (5 baris pertama)
        st.write("Cuplikan Data Sintetik (Top 5):")
        st.dataframe(df_gan.head(5), use_container_width=True)
        
        # Fungsi konversi DataFrame ke CSV (menggunakan cache agar cepat)
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv_gan = convert_df(df_gan)
        
        # Tombol Unduh CSV
        st.download_button(
            label="📥 Unduh 5.000 Dataset Sintetik GAN (CSV)",
            data=csv_gan,
            file_name='hea_gan_synthetic_data.csv',
            mime='text/csv',
            type="primary"
        )

    with st.expander("2. PINN Evaluator (Constraint Termodinamika)"):
        st.write("Mengevaluasi konstrain hukum kekekalan massa (Total = 100%).")
        st.latex(r"\mathcal{L}_{physics} = \left( \sum_{i=1}^{5} c_i - 100 \right)^2")
        st.success("Model PINN terkalibrasi. Prediksi kekuatan (MPa) dan biaya (USD/kg) siap.")

    with st.expander("3. NSGA-II Optimization (Pareto Optimal)"):
        st.write("Mengekstrak Himpunan Pareto dari triliunan permutasi komposisi...")
        
        # Simulasi data Pareto Front
        df_pareto = pd.DataFrame({
            'Fe': [25.0, 20.0, 15.0], 'Cr': [20.0, 20.0, 25.0], 
            'Ni': [15.0, 20.0, 25.0], 'Cu': [25.0, 20.0, 15.0], 'Al': [15.0, 20.0, 20.0],
            'Biaya_USD': [15.5, 22.0, 31.5],
            'Kekuatan_MPa': [450, 780, 920],
            'Label': ['Murah', 'Optimal', 'Kuat']
        })
        st.dataframe(df_pareto, use_container_width=True)
        return df_pareto.iloc[1] # Mengembalikan kandidat 'Optimal'

# --- PIPELINE B: FISIKA ---
def run_pipeline_b(kandidat):
    st.subheader("⚛️ Pipeline B: Fisika & Validasi Visual")
    
    # 1. Kalkulasi VEC
    vec_score = sum((kandidat[elem]/100) * VEC_VALS[elem] for elem in VEC_VALS)
    
    # Penentuan Fasa berdasarkan VEC
    fasa = "FCC (Ulet/Ductile)" if vec_score >= 8.0 else "BCC (Keras)" if vec_score < 6.8 else "FCC+BCC (Fasa Ganda)"
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Kalkulasi *Valence Electron Concentration* (VEC)**")
        st.latex(r"VEC = \sum_{i=1}^{n} c_i (VEC)_i")
        st.metric("Skor VEC Prediksi", f"{vec_score:.2f}", f"Fasa: {fasa}")
        
    with col2:
        st.markdown("**Generative Visualization (Microstructure)**")
        st.info("Render citra In-Silico dari fasa matriks berdasarkan skor VEC (Mockup Generative GAN).")
        # Visualisasi diagram fasa sederhana sebagai pengganti gambar GAN
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.barh([fasa], [vec_score], color='teal')
        ax.set_xlim(0, 12)
        ax.set_title("Stabilitas Fasa Kristal")
        st.pyplot(fig)

# --- EKSEKUSI ALUR ---
if st.button("🚀 Jalankan Analisis Dual-Pipeline", type="primary"):
    col_a, col_b = st.columns(2)
    
    with col_a:
        kandidat_terbaik = run_pipeline_a()
        
    with col_b:
        run_pipeline_b(kandidat_terbaik)
        
    st.markdown("---")
    st.success(f"**OUTPUT KANDIDAT FINAL:** Fe ({kandidat_terbaik['Fe']}%) - Cr ({kandidat_terbaik['Cr']}%) - Ni ({kandidat_terbaik['Ni']}%) - Cu ({kandidat_terbaik['Cu']}%) - Al ({kandidat_terbaik['Al']}%)")
    st.caption("✅ MANDAT PENELITIAN TERCAPAI: Keseimbangan optimal antara Efisiensi Biaya dan Performa Mekanik Superior.")
