# train_model.py
import os # For file path handling
import pandas as pd # For data manipulation
import re # For text preprocessing
from sklearn.feature_extraction.text import CountVectorizer # For text vectorization
from sklearn.model_selection import train_test_split # For splitting the dataset
from sklearn.linear_model import LogisticRegression # For logistic regression model
from sklearn.metrics import classification_report   # For model evaluation
import joblib # For saving the model and vectorizer
import json  # Used to store model performance metrics

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the manual test data
manual_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "testdata.manual.2009.06.14.csv"),
    header=None,
    names=["sentiment", "id", "date", "query", "user", "text"]
)
manual_df = manual_df[["sentiment", "text"]]
manual_df = manual_df[manual_df["sentiment"].isin([0, 2, 4])]

# Load the large Sentiment140 dataset
large_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "training.1600000.processed.noemoticon.csv"),
    encoding="latin-1",
    header=None,
    names=["sentiment", "id", "date", "query", "user", "text"]
)
large_df = large_df[["sentiment", "text"]]
large_df = large_df[large_df["sentiment"].isin([0, 4])]

# Preprocessing function
def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

# Clean both datasets
manual_df["text"] = manual_df["text"].apply(clean_text)
large_df["text"] = large_df["text"].apply(clean_text)

# Merge the datasets
combined_df = pd.concat([manual_df, large_df], ignore_index=True)

# Features and labels
X = combined_df["text"]
y = combined_df["sentiment"]

# Vectorization
vectorizer = CountVectorizer(stop_words="english", max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
report = classification_report(
    y_test, 
    y_pred, 
    target_names=["Negative", "Neutral", "Positive"], 
    output_dict=True, 
    zero_division=0  # Handle undefined metrics by setting them to 0
)
print(classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"], zero_division=0))

# Extract metrics
metrics = {
    "Accuracy": report["accuracy"],
    "Precision": {
        "Negative": report["Negative"]["precision"],
        "Neutral": report["Neutral"]["precision"],
        "Positive": report["Positive"]["precision"],
    },
    "Recall": {
        "Negative": report["Negative"]["recall"],
        "Neutral": report["Neutral"]["recall"],
        "Positive": report["Positive"]["recall"],
    },
}

# Save the metrics to a JSON file
metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved successfully to {metrics_path}")

# Save the model and the vectorizer
joblib.dump(model, os.path.join(BASE_DIR, "models", "sentiment_model.pkl"))
joblib.dump(vectorizer, os.path.join(BASE_DIR, "models", "vectorizer.pkl"))
print("Model and vectorizer saved successfully!")