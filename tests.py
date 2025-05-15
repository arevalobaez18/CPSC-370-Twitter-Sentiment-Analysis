# tests.py
import os
import joblib

# Define the path to the models folder dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load the trained model and vectorizer
model = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))

# Sentiment mapping
sentiment_map = {0: "Negative", 2: "Neutral", 4: "Positive"}

# Path to the test cases file
testcases_path = os.path.join(BASE_DIR, "test", "TestCases.txt")

# Read test cases from the file
test_cases = []
with open(testcases_path, "r", encoding="utf-8") as f:
    for line in f:
        text = line.strip().strip('“”"')
        if text:
            test_cases.append(text)

print("Sentiment Analysis Test Results:\n")
for text in test_cases:
    input_vectorized = vectorizer.transform([text])
    prediction = model.predict(input_vectorized)[0]
    sentiment = sentiment_map.get(prediction, "Unknown")
    print(f"Input: {text}\nPredicted Sentiment: {sentiment}\n")