import streamlit as st
import pandas as pd
import numpy as np
import torch
import whisper
import tempfile
import os
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---------------- PAGE CONFIGURATION ---------------- #
st.set_page_config(page_title="Sahayak", page_icon="🚀", layout="wide")

# ---------------- CUSTOM STYLING (Updated CSS) ---------------- #
st.markdown(
    """
    <style>
        /* Background for main page */
        [data-testid="stAppViewContainer"] {
            background-color: #F4F4F8;
        }
        
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #E3E3E8;
        }

        /* Sidebar text color */
        [data-testid="stSidebar"] * {
            color: #222 !important;
        }

        /* Button Styling */
        div.stButton > button {
            background-color: #007BFF !important;
            color: white !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            padding: 10px !important;
            transition: background-color 0.3s, transform 0.2s;
        }
        
        div.stButton > button:hover {
            background-color: #0056b3 !important;
            transform: scale(1.05);
        }

        /* Input & Select Box Styling */
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            color: black !important;
            border-radius: 5px !important;
            padding: 8px !important;
        }

        /* DataFrame Table Styling */
        .stDataFrame {
            background-color: #FFFFFF !important;
            color: black !important;
        }

        /* Headings & Titles */
        h1, h2, h3, h4 {
            color: #222 !important;
        }

        /* Generated Text Styling */
        .generated-text {
            color: white !important;
            background-color: #007BFF;
            padding: 10px;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ---------------- #
data = pd.read_csv('naukri.csv')

drop_cols = ['jobdescription', 'jobid', 'site_name', 'uniq_id', 'jobtitle', 'postdate']
data = data.drop(columns=drop_cols)

data.experience.fillna("2 - 7 yrs", inplace=True)
year_experience = data.experience.str.split(' ')
data['min_year_exp'] = year_experience.apply(lambda x: x[0])
data['max_year_exp'] = year_experience.apply(lambda x: x[2] if len(x) > 2 else x[0])
data['min_year_exp'] = np.where(data.min_year_exp == 'Not', '2', data.min_year_exp)
data['max_year_exp'] = np.where(data.max_year_exp == 'Not', '7', data.max_year_exp)
data['max_year_exp'] = np.where(data.max_year_exp == '-1', '1', data.max_year_exp)
data['min_year_exp'] = data['min_year_exp'].astype(int)
data['max_year_exp'] = data['max_year_exp'].astype(int)

data.education.fillna('UG: Any Graduate - Any Specialization', inplace=True)
data["degree"] = data.education.str.split(" PG:").str[0].str.replace("UG: ", "")

data.industry.fillna(data.industry.mode()[0], inplace=True)
data.joblocation_address.fillna('Not Mentioned', inplace=True)
data['loc'] = data.joblocation_address.apply(lambda x: [location.strip() for location in x.split(",")])

# ---------------- SIDEBAR NAVIGATION ---------------- #
st.sidebar.title("🚀 Sahayak")
menu = st.sidebar.radio("Choose an Option", ["📌 Job Recommendation", "🖼️ Image Captioning", "🎙️ Audio Transcription", "🌍 Text Translation"])

# ---------------- JOB RECOMMENDATION SYSTEM ---------------- #
if menu == "📌 Job Recommendation":
    st.title("🔍 Open Jobs Recommendation System")
    min_exp = st.sidebar.slider("Minimum Experience (Years)", 0, 25, 0)
    max_exp = st.sidebar.slider("Maximum Experience (Years)", 0, 27, 10)
    degree = st.sidebar.selectbox("🎓 Select Degree", data.degree.unique())
    skills = st.sidebar.selectbox("🛠️ Select Skills", data.skills.unique())
    loc_options = data['loc'].explode().unique()
    selected_loc = st.sidebar.multiselect("📍 Select Location", loc_options)
    
    loc_mask = data['loc'].apply(lambda x: any(l in selected_loc for l in x) if selected_loc else True)
    filtered_data = data[(data["min_year_exp"] >= min_exp) & 
                         (data["max_year_exp"] <= max_exp) & 
                         (data["degree"] == degree) & 
                         (data["skills"] == skills) & loc_mask]

    st.write("### 📌 Recommended Jobs")
    st.dataframe(filtered_data[['company', 'loc', 'industry', 'degree', 'experience', 'skills', 'numberofpositions', 'payrate']])

# ---------------- IMAGE CAPTIONING ---------------- #
elif menu == "🖼️ Image Captioning":
    st.title("🖼️ AI Image Captioning")
    uploaded_img = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="📷 Uploaded Image", use_column_width=True)
        caption = "🚀 A beautiful scenic view with mountains and a sunset!"  # Replace with AI-generated caption
        st.markdown(f'<p class="generated-text">📜 Caption: {caption}</p>', unsafe_allow_html=True)

# ---------------- AUDIO TRANSCRIPTION ---------------- #
elif menu == "🎙️ Audio Transcription":
    st.title("🎙️ AI Audio Transcription")
    uploaded_audio = st.file_uploader("🎵 Upload an Audio File", type=["mp3", "wav", "m4a"])
    if uploaded_audio:
        st.audio(uploaded_audio, format="audio/wav")
        if st.button("📝 Transcribe Audio"):
            result = "Hello! This is a sample transcription generated by AI."  # Replace with AI transcription
            st.markdown(f'<p class="generated-text">🗣️ Transcription: {result}</p>', unsafe_allow_html=True)

# ---------------- TEXT TRANSLATION ---------------- #
elif menu == "🌍 Text Translation":
    st.title("🌍 AI-Powered Text Translation")
    input_text = st.text_area("✍️ Enter text:", "This is a boy")
    source_lang = st.selectbox("🌐 Translate From:", ["English", "Hindi", "French", "German"])
    target_lang = st.selectbox("🔁 Translate To:", ["Hindi", "English", "French", "Spanish"])
    if st.button("🌎 Translate"):
        translation = "🚀 यह एक लड़का है"  # Replace with AI translation
        st.markdown(f'<p class="generated-text">✅ AI Translation: {translation}</p>', unsafe_allow_html=True)
