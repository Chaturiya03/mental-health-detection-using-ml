import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Load dataset
df = pd.read_csv("data/mental_health_data.csv", low_memory=False)
# Combine title and body
df['text'] = df['title'].fillna('') + " " + df['body'].fillna('')
# Initialize tools
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# Text cleaning function
def clean_text(text):
    text = text.lower()                          # lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)      # remove punctuation & numbers
    words = text.split()
    words = [w for w in words if w not in stop_words]   # remove stopwords
    words = [lemmatizer.lemmatize(w) for w in words]    # lemmatize
    return " ".join(words)
# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)
print("TEXT CLEANING COMPLETE\n")
print("Original text:\n")
print(df['text'].iloc[0])
print("\nCleaned text:\n")
print(df['clean_text'].iloc[0])