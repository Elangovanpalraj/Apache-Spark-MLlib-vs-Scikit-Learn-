import streamlit as st
import pandas as pd
import joblib

# Load model and vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("review_sentiment_model.pkl")

# -------------------------------
# Title
st.title("📦 Amazon Review Sentiment Classifier")
st.write("இந்த app மூலம் உங்கள் விமர்சனத்தின் nature (நல்லதா / மோசமா) என்பதை கணிக்கலாம்.")

# -------------------------------
# Single Review Prediction
st.header("✍️ தனிப்பட்ட விமர்சனக் கணிப்பு")

user_input = st.text_area("📝 உங்கள் விமர்சனத்தை உள்ளிடுங்கள்:")

if st.button("🔍 கணிக்க"):
    if user_input.strip() == "":
        st.warning("⛔ விமர்சனத்தை உள்ளிடவும்.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        if prediction == 1:
            st.success("✅ இது ஒரு நல்ல விமர்சனம் (Positive Review).")
        else:
            st.error("❌ இது ஒரு மோசமான விமர்சனம் (Negative Review).")

# -------------------------------
# Bulk CSV Upload & Prediction
st.header("📁 CSV விமர்சனங்களை கையாளல் (Bulk Prediction)")

uploaded_file = st.file_uploader("CSV கோப்பை பதிவேற்றுக (column name: 'reviewText')", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if 'reviewText' not in df.columns:
            st.error("❌ 'reviewText' column இல்லை. CSV column name சரிபார்க்கவும்.")
        else:
            st.success(f"✅ {len(df)} விமர்சனங்கள் ஏற்றப்பட்டன.")
            input_vecs = vectorizer.transform(df['reviewText'].fillna(""))
            preds = model.predict(input_vecs)
            df['Prediction'] = preds
            df['Sentiment'] = df['Prediction'].apply(lambda x: 'Positive' if x == 1 else 'Negative')

            st.dataframe(df[['reviewText', 'Sentiment']])

            csv_download = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 முடிவுகளைப் பதிவிறக்கு",
                data=csv_download,
                file_name='predicted_reviews.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"⚠️ பிழை ஏற்பட்டது: {e}")
