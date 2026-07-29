import streamlit as st
import joblib
import pandas as pd
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image
# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

model = joblib.load("models/new_sentiment_model.pkl")
vectorizer = joblib.load("models/new_tfidf_vectorizer.pkl")

# -------------------------------------------------
# NLP SETUP
# -------------------------------------------------

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -------------------------------------------------
# TEXT CLEANING FUNCTION
# -------------------------------------------------

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------

def predict_sentiment(review):

    cleaned = clean_text(review)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = probabilities.max() * 100

    return prediction, confidence, probabilities

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.title("🛈 Project Information")

    st.markdown("---")

    st.write("### Model")

    st.success("Logistic Regression")

    st.write("### Feature Extraction")

    st.info("TF-IDF Vectorizer")

    st.write("### NLP")

    st.write("✔ Lowercase")

    st.write("✔ Remove URLs")

    st.write("✔ Remove Punctuation")

    st.write("✔ Stopword Removal")

    st.write("✔ Lemmatization")

    st.markdown("---")

    st.caption("Developed by Roshan Jadhao")

# -------------------------------------------------
# HEADER WITH LOGO
# -------------------------------------------------

col1, col2 = st.columns([1, 5])

with col1:
    logo = Image.open("logo2.png")
    st.image(logo, width=90)

with col2:
    st.markdown(
        """
        <h1 style='color:#2874F0; margin-bottom:0px;'>
        Flipkart Product Review Sentiment Analysis
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.caption("Analyze customer reviews using Machine Learning")

st.markdown("---")

# -------------------------------------------------
# INPUT
# -------------------------------------------------

review = st.text_area(
    "Enter Flipkart Product Review: ",
    height=180,
    placeholder="Example:\nThis phone is amazing. Battery backup is excellent and camera quality is superb."
)

predict_btn = st.button(
    "🔍 Analyze Review",
    use_container_width=True
)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if predict_btn:

    if review.strip() == "":

        st.warning("Please enter a review.")

    else:

        prediction, confidence, probabilities = predict_sentiment(review)

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == "Positive":

            st.success(f"😊 **{prediction}**")

        elif prediction == "Negative":

            st.error(f"😠 **{prediction}**")

        else:

            st.info(f"😐 **{prediction}**")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.subheader("Prediction Probability")

        probability_df = pd.DataFrame({

            "Sentiment": model.classes_,

            "Probability (%)": probabilities * 100

        })

        st.bar_chart(
            probability_df.set_index("Sentiment")
        )

        st.dataframe(
            probability_df.style.format({
                "Probability (%)": "{:.2f}"
            }),
            use_container_width=True
        )

# -------------------------------------------------
# SAMPLE REVIEWS
# -------------------------------------------------

st.markdown("---")

st.subheader("Try These Sample Reviews")

col1, col2 = st.columns(2)

with col1:

    st.success("""
**Positive Review**

Excellent phone.


Battery backup is amazing.

Camera quality is outstanding.
""")

with col2:

    st.error("""
**Negative Review**

Worst phone.

Battery drains very quickly.

Waste of money.
""")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.markdown(
"""
<div style="text-align:center">

Made with ❤️ using

<b>Python</b> |
<b>Scikit-Learn</b> |
<b>NLTK</b> |
<b>Streamlit</b>

</div>
""",
unsafe_allow_html=True
)
