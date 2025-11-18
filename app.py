import streamlit as st
import pandas as pd
import joblib

# ============================================================
# 🔹 KONSTANTA DAN LOAD FILE
# ============================================================

MODEL_PATH = "model_random_forest_sehat2.pkl"
DATA_PATH = "cleaned_ingredients.csv"
PREDICTION_COLUMN = "Prediksi (Sehat/Tidak)"

try:
    # 1. Load model
    model = joblib.load(MODEL_PATH)
    FITUR_MODEL = list(model.feature_names_in_)
except FileNotFoundError:
    st.error(f"Error: File model tidak ditemukan di jalur: {MODEL_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()

try:
    # 2. Load dataset
    df_bahan = pd.read_csv(DATA_PATH)
    # Ubah nama kolom 'Descrip' menjadi 'Nama Bahan' untuk UI yang lebih baik
    df_bahan = df_bahan.rename(columns={'Descrip': 'Nama Bahan'})
    # Kolom nutrisi yang akan dihitung totalnya
    KOLOM_NUTRISI = [col for col in df_bahan.columns if col not in ['NDB_No', 'Nama Bahan']]
    
    # Konversi kolom 'Nama Bahan' menjadi huruf kecil untuk pencarian yang tidak sensitif huruf besar/kecil
    df_bahan['Nama Bahan Lower'] = df_bahan['Nama Bahan'].str.lower()
except FileNotFoundError:
    st.error(f"Error: File dataset tidak ditemukan di jalur: {DATA_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Error saat memuat dataset: {e}")
    st.stop()

# ============================================================
# 🔹 FUNGSI PREDIKSI
# ============================================================

def sesuaikan_input(data_input: dict):
    """
    Menyesuaikan input data nutrisi dengan fitur yang digunakan model saat training.
    """
    df_input = pd.DataFrame([data_input])

    # Hapus kolom yang tidak ada dalam fitur model
    df_input = df_input[[col for col in df_input.columns if col in FITUR_MODEL]]

    # Tambahkan kolom yang hilang dan isi dengan 0
    for kolom in FITUR_MODEL:
        if kolom not in df_input.columns:
            df_input[kolom] = 0.0

    # Urutkan kolom agar sesuai dengan urutan fitur model
    df_input = df_input[FITUR_MODEL]
    return df_input

def prediksi_kesehatan(data_input: dict):
    """
    Melakukan prediksi kesehatan berdasarkan total nutrisi yang diinput.
    """
    df_fixed = sesuaikan_input(data_input)

    # Pastikan data memiliki setidaknya satu kolom nutrisi yang valid
    if df_fixed.empty:
        return {"Prediksi": "Data Kosong", "Prob_Tidak_Sehat (%)": 0, "Prob_Sehat (%)": 0}

    pred = model.predict(df_fixed)[0]
    proba = model.predict_proba(df_fixed)[0]

    hasil = {
        "Prediksi": "Sehat (1)" if pred == 1 else "Tidak Sehat (0)",
        "Prob_Tidak_Sehat (%)": round(proba[0] * 100, 2),
        "Prob_Sehat (%)": round(proba[1] * 100, 2)
    }
    return hasil

# ============================================================
# 🔹 KONFIGURASI HALAMAN STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Prediksi Kesehatan Nutrisi Menu",
    layout="wide",
    page_icon="🍲",
)

st.title("🍲 Prediksi Kesehatan Nutrisi Menu")
st.markdown("Pilih bahan makanan harianmu dari ribuan data, dan AI akan memprediksi apakah total nutrisinya cenderung **Sehat** atau **Tidak Sehat**.")

st.markdown("---")

# ============================================================
# 🔹 INISIALISASI SESSION STATE
# ============================================================

if "selected_ingredients" not in st.session_state:
    st.session_state.selected_ingredients = []

# ============================================================
# 🔹 FITUR 1: PENCARIAN & PENAMBAHAN BAHAN
# ============================================================

st.subheader("🔎 Pilih Bahan Makanan")

# UI untuk Pencarian Bahan
search_query = st.text_input("Ketik nama bahan makanan untuk mencari (e.g., butter, catla, fish):", key="search_input")

# Inisialisasi daftar opsi kosong
bahan_options = []

if search_query:
    # Filter DataFrame berdasarkan input pencarian (menggunakan kolom lower-case untuk case-insensitivity)
    # Batasi hasil pencarian hingga 50 data agar UI tidak terlalu panjang
    search_lower = search_query.lower()
    filtered_df = df_bahan[df_bahan['Nama Bahan Lower'].str.contains(search_lower)].head(50)
    bahan_options = filtered_df['Nama Bahan'].tolist()
    
    if not bahan_options:
        st.warning("Tidak ditemukan bahan yang cocok dengan kata kunci.")
else:
    # Tampilkan pesan bantuan saat kolom pencarian kosong
    st.info("Silakan ketik nama bahan makanan di atas untuk memulai pencarian.")

# Kolom untuk memilih dan menambahkan
col_select, col_weight, col_add = st.columns([4, 1, 1])

# Selectbox untuk memilih bahan dari hasil filter
food_to_add = col_select.selectbox("Pilih bahan dari hasil pencarian:", options=bahan_options, disabled=not bool(bahan_options))

# Input jumlah (berat) bahan dalam gram
input_grams = col_weight.number_input("Berat (g)", min_value=1, value=100, step=1, key="gram_input")

if col_add.button("➕ Tambahkan ke Menu", use_container_width=True, disabled=not bool(food_to_add)):
    if food_to_add:
        # Ambil data nutrisi per 100g dari bahan terpilih
        # Gunakan kolom 'Nama Bahan' untuk pencarian final, karena ini yang digunakan untuk selectbox
        selected_row = df_bahan[df_bahan["Nama Bahan"] == food_to_add].iloc[0].copy()

        # Hitung nutrisi untuk berat yang diinput
        ratio = input_grams / 100.0
        for col in KOLOM_NUTRISI:
            # Pastikan perhitungan hanya untuk kolom numerik (nutrisi)
            if pd.api.types.is_numeric_dtype(selected_row[col]):
                selected_row[col] *= ratio

        # Tambahkan kolom untuk berat yang dipilih
        selected_row['Berat (g)'] = input_grams

        # Tambahkan ke session state
        st.session_state.selected_ingredients.append(selected_row.to_dict())
        st.success(f"**{food_to_add} ({input_grams} g)** berhasil ditambahkan!")

st.markdown("---")

# ============================================================
# 🔹 TAMPILKAN DAFTAR BAHAN TERPILIH & TOTAL NUTRISI
# ============================================================

if st.session_state.selected_ingredients:
    st.subheader("🧾 Menu Makanan Harian Anda")

    # Konversi list of dicts menjadi DataFrame
    selected_df = pd.DataFrame(st.session_state.selected_ingredients)
    
    # Hapus kolom tambahan (seperti NDB_No dan Nama Bahan Lower) dari tampilan
    display_cols = ['Nama Bahan', 'Berat (g)', 'Energy_kcal', 'Protein_g', 'Fat_g', 'Sodium_mg']
    
    st.dataframe(
        selected_df[[col for col in display_cols if col in selected_df.columns]].round(2),
        use_container_width=True,
        hide_index=True
    )

    # Hitung total nutrisi yang akan diinput ke model
    total_nutrisi = selected_df[KOLOM_NUTRISI].sum().to_dict()

    st.markdown("### 🔢 Total Nutrisi Harian Anda")

    # Tampilkan total nutrisi utama dalam metric boxes
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Energi (kcal)", f"{total_nutrisi.get('Energy_kcal', 0):.1f}")
    col2.metric("Protein (g)", f"{total_nutrisi.get('Protein_g', 0):.1f}")
    col3.metric("Lemak Total (g)", f"{total_nutrisi.get('Fat_g', 0):.1f}")
    col4.metric("Natrium (mg)", f"{total_nutrisi.get('Sodium_mg', 0):.1f}")

    st.markdown("---")

    # ============================================================
    # 🔹 PREDIKSI KESEHATAN
    # ============================================================

    st.subheader("💡 Hasil Prediksi Kesehatan")

    if st.button("🔮 Prediksi Kesehatan Menu Ini", type="primary"):
        # Lakukan prediksi
        hasil_prediksi = prediksi_kesehatan(total_nutrisi)
        
        # Tampilkan Hasil
        st.metric(
            label=PREDICTION_COLUMN,
            value=hasil_prediksi['Prediksi'],
        )
        
        # Tampilkan probabilitas dalam progress bar dan keterangan
        prob_sehat = hasil_prediksi["Prob_Sehat (%)"]
        
        st.markdown(f"**Tingkat Keyakinan Prediksi:**")
        st.progress(prob_sehat / 100.0, text=f"Sehat: {prob_sehat}% | Tidak Sehat: {hasil_prediksi['Prob_Tidak_Sehat (%)']}%")

        if prob_sehat > 70:
            st.balloons()
            st.success("🎉 Berdasarkan komposisi nutrisinya, menu ini diprediksi **Sangat Sehat**!")
        elif prob_sehat > 50:
             st.info("👍 Berdasarkan komposisi nutrisinya, menu ini diprediksi **Cenderung Sehat**.")
        else:
            st.warning("⚠️ Berdasarkan komposisi nutrisinya, menu ini diprediksi **Cenderung Tidak Sehat**.")

    # Tombol untuk mereset menu
    if st.button("🗑️ Hapus Semua Bahan dari Menu"):
        st.session_state.selected_ingredients = []
        st.rerun()

# ============================================================
# 🔹 FOOTER
# ============================================================

st.markdown("---")
st.caption("Aplikasi Prediksi Kesehatan Nutrisi | Dibuat dengan Streamlit & Model Random Forest")