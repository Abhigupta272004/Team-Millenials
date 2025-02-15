import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import scipy.sparse

# Download necessary NLTK data
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk_resources()

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')

    # Drop unnecessary columns
    df = df[['v1', 'v2']]
    df = df.rename(columns={'v1': 'type', 'v2': 'message'})

    # Ensure 'type' column is formatted correctly
    df['type'] = df['type'].astype(str).str.lower().str.strip()

    # Convert labels to numerical format
    df = df[df['type'].isin(['ham', 'spam'])]  # Filter only valid labels
    y = df['type'].map({'ham': 0, 'spam': 1}).astype(int)

    # Text cleaning function
    def clean_text(text):
        text = str(text).lower()  # Convert to string and lowercase
        text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()
        cleaned = [ps.stem(word) for word in tokens if word not in stop_words]
        return " ".join(cleaned)

    df['cleaned_message'] = df['message'].apply(clean_text)

    return df, y

# Load the data
file_path = 'spam.csv'
df, y = load_and_preprocess_data(file_path)

# Feature extraction - Cache TFIDF Vectorizer
@st.cache_resource
def get_tfidf_vectorizer():
    return TfidfVectorizer()

tfidf = get_tfidf_vectorizer()
X = tfidf.fit_transform(df['cleaned_message'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model with cache fix
@st.cache_resource(hash_funcs={scipy.sparse.csr_matrix: lambda _: None})
def train_model(_X_train, _y_train):
    model = SVC(kernel='linear')
    model.fit(_X_train.toarray(), _y_train)  # Convert sparse matrix to dense
    return model

model = train_model(X_train, y_train)

# Streamlit UI with Enhanced Design
st.markdown(
    "<h1 style='text-align: center; color: #2E8B57;'>📩 Spam SMS Classifier</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Detect whether a text message is Spam or Not using Machine Learning</p>",
    unsafe_allow_html=True
)

# Create a layout with columns
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 🔍 Enter a text message:")
    user_input = st.text_area("", height=100, placeholder="Type your message here...")

with col2:
    st.markdown("### ⚡ Quick Guide:")
    st.markdown("- Enter a text message")
    st.markdown("- Click **Predict** to check if it's spam")
    st.markdown("- Model is trained on SMS spam dataset")

# Prediction function
def predict_spam(text):
    cleaned_text = re.sub(r'[^a-zA-Z]', ' ', text.lower())  # Basic cleaning
    text_vector = tfidf.transform([cleaned_text])
    prediction = model.predict(text_vector.toarray())[0]  # Convert sparse matrix to dense
    return "Spam 🚨" if prediction == 1 else "Not Spam ✅"

# Predict Button with Styling
st.markdown("<br>", unsafe_allow_html=True)
if st.button('🚀 Predict', use_container_width=True):
    if user_input.strip():
        prediction = predict_spam(user_input)
        st.markdown(
            f"<div style='text-align: center; padding: 15px; border-radius: 10px; background-color: #f4f4f4; font-size: 20px;'><b>{prediction}</b></div>",
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Please enter a message before predicting.")


