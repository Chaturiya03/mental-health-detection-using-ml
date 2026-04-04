import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from preprocess import clean_text

print("STEP 1: Creating folders...")
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("STEP 2: Loading dataset...")
df = pd.read_csv("data/mental_health_data.csv", low_memory=False)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# OPTIONAL: use only first 5000 rows for faster testing
df = df.sample(n=5000, random_state=42)

print("STEP 3: Combining title and body...")
df['text'] = df['title'].fillna('') + " " + df['body'].fillna('')

print("STEP 4: Cleaning text...")
df['clean_text'] = df['text'].apply(clean_text)

print("STEP 5: Preparing features and labels...")
X_text = df['clean_text']
y = df['label']

print("STEP 6: Applying TF-IDF...")
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(X_text)

print("STEP 7: Splitting train and test data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("STEP 8: Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("STEP 9: Making predictions...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n===== MODEL RESULTS =====")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(report)

print("STEP 10: Saving classification report...")
with open("outputs/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print("STEP 11: Saving confusion matrix image...")
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("STEP 12: Saving model and vectorizer...")
joblib.dump(model, "models/depression_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\n✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("Saved:")
print("- models/depression_model.pkl")
print("- models/tfidf_vectorizer.pkl")
print("- outputs/confusion_matrix.png")
print("- outputs/classification_report.txt")