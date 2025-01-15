import os

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import swifter  # For fast Pandas apply


csv1_path = os.path.join("train.csv")  # Example: training CSV
csv2_path = os.path.join("test.csv")  # Example: testing CSV

column_names = ["label", "Review summary", "Review text"]

# Load CSV files into DataFrames
train_df = pd.read_csv(csv1_path, names=column_names, header=None)
test_df = pd.read_csv(csv2_path, names=column_names, header=None)
""
df = train_df
df = df.sample(frac=0.1, random_state=42)
df = df[df['label'] != 3]
df['text'] = df['Review summary'] + " " + df['Review text']
df.dropna(inplace=True)
df['label'] = df['label'].map({1: 0, 2: 0, 4: 1, 5: 1})
test_df = test_df.sample(frac=0.2, random_state=42)
test_df = test_df[test_df['label'] != 3]
test_df['text'] = test_df['Review summary'] + " " + test_df['Review text']
test_df.dropna(inplace=True)
test_df['label'] = test_df['label'].map({1: 0, 2: 0, 4: 1, 5: 1})



# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    tokens = nltk.word_tokenize(text)  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords and lemmatize
    return ' '.join(tokens)

# Apply preprocessing in parallel using Swifter
df['processed_review'] = df['text'].swifter.apply(preprocess_text)
test_df['processed_review'] = test_df['text'].swifter.apply(preprocess_text)

# Encode Sentiment Labels

# Split Data into Train and Test Sets
X = df['processed_review']  # Features (processed reviews)
y = df['label']  # Labels (encoded sentiment)

X_test_new = test_df['processed_review']
y_test_new = test_df['label']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
X_test_tfidf_new = vectorizer.transform(X_test_new)

# Output processed data shapes
print("Train TF-IDF Shape:", X_train_tfidf.shape)
print("Test TF-IDF Shape:", X_test_tfidf.shape)
print("Test TF-IDF Shape:", X_test_tfidf_new.shape)

# Save processed DataFrame
df.to_csv("processed_reviews_swifter.csv", index=False)

# Optional: Save vectorized data (if needed for later use)
np.save("X_train_tfidf.npy", X_train_tfidf.toarray())
np.save("X_val_tfidf.npy", X_test_tfidf.toarray())
np.save("y_train_tfidf.npy", y_train.to_numpy())
np.save("y_val_tfidf.npy", y_test.to_numpy())
np.save("X_test_tfidf_new.npy", X_test_tfidf_new.toarray())
np.save("y_test_new.npy", y_test_new.to_numpy())
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
