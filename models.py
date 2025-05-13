# === Modules ===
import zipfile, os, time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# === 1. Extract ZIP and Load Data ===
zip_path = r"D:\Apache Spark MLlib vs Scikit-Learn\amazon_reviews.zip"
extract_path = "./amazon_reviews"

# Extract only if not already extracted
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

csv_file = os.path.join(extract_path, "amazon_reviews.csv")
df = pd.read_csv(csv_file)

# === 2. Clean and Prepare Data ===
if 'reviewText' not in df.columns or 'rating' not in df.columns:
    raise KeyError("Missing 'reviewText' or 'rating' in CSV.")

df = df[['reviewText', 'rating']].dropna()
df['label'] = df['rating'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else None))
df = df[df['label'].notnull()]  # Remove neutral ratings (e.g., rating == 3)

# === 3. Train-Test Split and Vectorization ===
X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === 4. Train Model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)

print("🔄 Training Scikit-learn model...")
start_time = time.time()
model.fit(X_train_vec, y_train)
train_time = time.time() - start_time

# === 5. Evaluate ===
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.4f}")
print(f"⏱ Training Time: {train_time:.2f} seconds")
print("\n🧾 Classification Report:")
print(classification_report(y_test, y_pred))

# === 6. Plot (Optional) ===
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy], color='blue')
plt.ylabel("Accuracy")
plt.title("Scikit-learn Model Accuracy")
plt.ylim(0, 1)
plt.show()

# ==================================================================================================
import numpy as np

# === 7. Show Top Features ===
print("\n📊 Top 20 Important Words in Classification:")
feature_names = vectorizer.get_feature_names_out()
importances = model.feature_importances_

# Get top 20 features
indices = np.argsort(importances)[-20:][::-1]
top_words = [feature_names[i] for i in indices]
top_scores = importances[indices]

# Print
for word, score in zip(top_words, top_scores):
    print(f"{word}: {score:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.barh(top_words[::-1], top_scores[::-1], color='purple')
plt.xlabel("Importance Score")
plt.title("Top 20 Important Words (TF-IDF + Random Forest)")
plt.tight_layout()
plt.show()
# ==========================================================================================================
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Predictions
y_pred = model.predict(X_test_vec)

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative (0)", "Positive (1)"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
# ===========================================================================================================
import joblib

# Save the vectorizer and model
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(model, "review_sentiment_model.pkl")
print("✅ Model மற்றும் vectorizer save செய்யப்பட்டது.")
# ================================================================================================
# Load model and vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("review_sentiment_model.pkl")

# Take input
user_input = input("✍️ உங்கள் விமர்சனத்தை உள்ளிடுங்கள்: ")

# Preprocess and predict
input_vec = vectorizer.transform([user_input])
prediction = model.predict(input_vec)[0]

# Output result
if prediction == 1:
    print("✅ இது ஒரு நல்ல விமர்சனம் (Positive Review).")
else:
    print("❌ இது ஒரு மோசமான விமர்சனம் (Negative Review).")
# ==============================================================================================
