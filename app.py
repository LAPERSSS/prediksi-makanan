# ============================================================
# 🥗 STREAMLIT APP: Prediksi Kesehatan untuk Kombinasi Bahan
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from deep_translator import GoogleTranslator


# ============================================================
# CONFIG & PATHS
# ============================================================

MODEL_PATH = "model_random_forest_sehat2.pkl"
DATA_PATH = "cleaned_ingredients.csv"

st.set_page_config(page_title="Prediksi Kesehatan - Kombinasi Menu", layout="wide", page_icon="🥗")

# ============================================================
# LOAD MODEL & DATA
# ============================================================

model = joblib.load(MODEL_PATH)

# load dataset
df = pd.read_csv(DATA_PATH)

# normalize column names to lowercase
df.columns = df.columns.str.strip().str.lower()

# find descriptive column
desc_col = "descrip"
possible = [c for c in df.columns if "desc" in c or "name" in c]
if desc_col not in df.columns:
    desc_col = possible[0] if possible else df.columns[0]

# numeric nutrient columns in cleaned_ingredients.csv
exclude_cols = {desc_col, "ndb_no"}
numeric_cols = [c for c in df.columns if c not in exclude_cols and
                pd.api.types.is_numeric_dtype(df[c])]

# features expected by the model
fitur_model = list(model.feature_names_in_)

# ============================================================
# 🔧 FIX: MAPPING KOLOM cleaned_ingredients → training_data
# ============================================================

# training_data punya kolom:
# ['additives_n', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g',
#  'sugars_100g', 'fiber_100g', 'proteins_100g', 'sodium_100g',
#  'nutrition-score-uk_100g', 'healthy_label']

MAPPING = {
    "fat_100g": ["fat_g", "fat", "total_fat_g"],
    "saturated-fat_100g": ["saturated_fat_g", "sat_fat_g"],
    "carbohydrates_100g": ["carb_g", "carbohydrate_g"],
    "sugars_100g": ["sugar_g"],
    "fiber_100g": ["fiber_g"],
    "proteins_100g": ["protein_g", "proteins_g"],
    "sodium_100g": ["sodium_mg", "salt_mg"],  # mg → g otomatis nanti
    "additives_n": ["additives", "additives_n"],
    # kalau tidak ada nutrution_score_uk di CSV, isi default 0
}

def map_totals_ke_fitur_model(totals):
    model_input = {}

    for feat in fitur_model:
        if feat in MAPPING:
            sumber_list = MAPPING[feat]
            nilai = None
            for s in sumber_list:
                if s in totals:
                    nilai = totals[s]
                    break

            if nilai is None:
                # fallback: isi 0 kalau sumber tidak ada
                nilai = 0

            # konversi jika sodium mg → sodium_100g (g)
            if feat == "sodium_100g":
                nilai = nilai / 1000.0

            model_input[feat] = nilai

        else:
            # kolom seperti nutrition-score-uk_100g jika tidak ada → isi 0
            model_input[feat] = totals.get(feat, 0)

    return model_input

# ============================================================
# HELPERS
# ============================================================

def sesuaikan_input(data_input: dict):
    df_input = pd.DataFrame([data_input])
    for kol in fitur_model:
        if kol not in df_input.columns:
            df_input[kol] = 0
    df_input = df_input[fitur_model]
    return df_input

def prediksi_kesehatan(data_input: dict):
    df_fix = sesuaikan_input(data_input)
    pred = model.predict(df_fix)[0]
    proba = model.predict_proba(df_fix)[0]
    return {
        "Prediksi": "Sehat (1)" if pred == 1 else "Tidak Sehat (0)",
        "Prob_Tidak_Sehat (%)": round(proba[0] * 100, 2),
        "Prob_Sehat (%)": round(proba[1] * 100, 2),
    }

# ============================================================
# SESSION STATE
# ============================================================

if "selected_foods" not in st.session_state:
    st.session_state.selected_foods = []

# ============================================================
# UI (tidak diubah)
# ============================================================

st.title("🥗 Prediksi Kesehatan untuk Kombinasi Bahan (Menu)")
st.markdown(
    "Pilih beberapa bahan dari database, atur jumlah porsi (servings), lalu tekan **Prediksi Kesehatan Menu**."
)

st.sidebar.header("Tips")
st.sidebar.write("- Gunakan search untuk menemukan bahan.")
st.sidebar.write("- Masukkan jumlah porsi.")
st.sidebar.write("- Kamu bisa menghapus item terpilih.")

# ============================================================
# Panel kiri
# ============================================================

with st.container():
    st.subheader("🔍 Cari & Tambah Bahan")
    col1, col2 = st.columns([3,1])
    with col1:
        search_q = st.text_input("Ketik nama bahan (Indonesia / Inggris):", "").strip()

    if search_q:
        # translate Indo → English
        try:
            terjemahan = GoogleTranslator(source="id", target="en").translate(search_q)
        except:
            terjemahan = search_q


        # gabungkan dua pencarian: original + hasil translate
        filtered = df[
            df[desc_col].str.contains(search_q, case=False, na=False) |
            df[desc_col].str.contains(terjemahan, case=False, na=False)
        ]
    else:
        filtered = df

    with col2:
        options = filtered[desc_col].unique().tolist()
        selected_name = st.selectbox("Pilih bahan:", options) if len(options) > 0 else None
        servings = st.number_input("Porsi:", min_value=0.1, value=1.0, step=0.5)

        if st.button("➕ Tambah ke Daftar"):
            if selected_name:
                row = filtered[filtered[desc_col] == selected_name].iloc[0]
                entry = {desc_col: selected_name, "servings": float(servings)}
                for col in numeric_cols:
                    val = row[col] if pd.notna(row[col]) else 0.0
                    entry[col] = float(val) * float(servings)
                st.session_state.selected_foods.append(entry)
                st.success(f"{selected_name} ditambahkan!")

# ============================================================
# Panel tengah
# ============================================================

st.subheader("🧾 Daftar Bahan yang Dipilih")

if st.session_state.selected_foods:
    sel_df = pd.DataFrame(st.session_state.selected_foods)
    st.dataframe(sel_df[[desc_col, "servings"] + numeric_cols[:8]])

    remove_choices = st.multiselect("Hapus item:", sel_df[desc_col].tolist())
    if st.button("🗑 Hapus terpilih"):
        st.session_state.selected_foods = [e for e in st.session_state.selected_foods if e[desc_col] not in remove_choices]

    if st.button("🧹 Bersihkan Semua"):
        st.session_state.selected_foods = []
else:
    st.info("Belum ada bahan.")

# ============================================================
# Panel kanan + prediksi
# ============================================================

st.subheader("🔢 Ringkasan & Prediksi")

if st.session_state.selected_foods:
    total_df = pd.DataFrame(st.session_state.selected_foods).fillna(0)
    totals = {"servings": total_df["servings"].sum()}
    for col in numeric_cols:
        totals[col] = total_df[col].sum()

    st.markdown("**Total nutrisi:**")
    st.dataframe(pd.DataFrame([totals]))

    # 🔥 FIX UTAMA: mapping totals → fitur model
    model_input = map_totals_ke_fitur_model(totals)

    if st.button("🔮 Prediksi Kesehatan Menu"):
        hasil = prediksi_kesehatan(model_input)
        st.success(f"**{hasil['Prediksi']}**")
        c1, c2 = st.columns(2)
        c1.metric("Prob Tidak Sehat", f"{hasil['Prob_Tidak_Sehat (%)']}%")
        c2.metric("Prob Sehat", f"{hasil['Prob_Sehat (%)']}%")
        st.progress(hasil['Prob_Sehat (%)'] / 100)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("💡 Versi perbaikan mapping kolom CSV → model")
