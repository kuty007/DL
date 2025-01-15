import os
import pandas as pd
import numpy as np
import re
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CSV files into DataFrames
csv1_path = os.path.join("train.csv")  # Example: training CSV
csv2_path = os.path.join("test.csv")  # Example: testing CSV

column_names = ["label", "Review summary", "Review text"]

train_df = pd.read_csv(csv1_path, names=column_names, header=None)
test_df = pd.read_csv(csv2_path, names=column_names, header=None)
train_df = train_df.sample(frac=0.1, random_state=42)
train_df = train_df[train_df['label'] != 3]
test_df = test_df.sample(frac=0.2, random_state=42)
test_df = test_df[test_df['label'] != 3]
# Combine summary and text for both train and test data
train_df['text'] = train_df['Review summary'] + " " + train_df['Review text']
test_df['text'] = test_df['Review summary'] + " " + test_df['Review text']

# Drop NaNs and filter out label 3 (neutral)
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Map labels: 1 -> 0 (negative), 2 -> 0, 4 -> 1 (positive), 5 -> 1
train_df['label'] = train_df['label'].map({1: 0, 2: 0, 4: 1, 5: 1})
test_df['label'] = test_df['label'].map({1: 0, 2: 0, 4: 1, 5: 1})

# Download BERT tokenizer and model
nltk.download('punkt')
nltk.download('wordnet')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Preprocessing function using BERT
def preprocess_text_with_bert(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

    with torch.no_grad():
        outputs = bert_model(**encoded_input)

    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


# Process train data
train_embeddings = torch.cat([preprocess_text_with_bert(text) for text in train_df['text']])

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_embeddings, train_df['label'], test_size=0.2, random_state=42)

# Process test data
test_embeddings = torch.cat([preprocess_text_with_bert(text) for text in test_df['text']])

# Save processed data
np.save("X_train_bert.npy", X_train.cpu().numpy())
np.save("X_val_bert.npy", X_val.cpu().numpy())
np.save("y_train.npy", y_train.to_numpy())
np.save("y_val.npy", y_val.to_numpy())
np.save("X_test_bert.npy", test_embeddings.cpu().numpy())
np.save("y_test.npy", test_df['label'].to_numpy())

# Save tokenizer
tokenizer.save_pretrained("bert_tokenizer")

# Print the shapes of the processed data
print(f"Train BERT Embeddings Shape: {X_train.shape}")
print(f"Validation BERT Embeddings Shape: {X_val.shape}")
print(f"Test BERT Embeddings Shape: {test_embeddings.shape}")
