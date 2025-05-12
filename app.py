# app.py
import streamlit as st # For creating the web app
import joblib # For loading the model and vectorizer
import json # Used to store model performance metrics
import os # For file path handling

# Define the path to the models folder dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load the trained model and vectorizer (the files saved after training)
model = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))

# Sentiment mapping - Map the sentiment labels to their respective categories
sentiment_map = {0: "Negative", 2: "Neutral", 4: "Positive"}

# Streamlit app title
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet below to predict its sentiment (Positive, Neutral, or Negative).")

# Text input field
user_input = st.text_input("Enter your text here:")

# Real-time sentiment prediction
# Check if the input is not empty
if user_input.strip():
    # Vectorize the input text
    input_vectorized = vectorizer.transform([user_input])
    
    # Predict the sentiment
    prediction = model.predict(input_vectorized)[0]
    sentiment = sentiment_map[prediction]
    
    # Define color based on sentiment for display purposes
    color = "green" if sentiment == "Positive" else "blue" if sentiment == "Neutral" else "red"
    
    # Display the result for the user
    st.markdown(
        f"<h3>Predicted Sentiment: <span style='color: {color};'>{sentiment}</span></h3>",
        unsafe_allow_html=True
    )
else:   # If the input is empty, prompt the user
    st.write("Start typing to see the sentiment prediction.")

# Display model performance metrics
st.write("---")
st.subheader("Model Performance Metrics")

# Load the metrics from the JSON file
metrics_path = os.path.join(MODELS_DIR, "metrics.json")
with open(metrics_path, "r") as f:
    model_metrics = json.load(f)

# Display accuracy
st.markdown(f"### **Overall Accuracy:** `{model_metrics['Accuracy'] * 100:.2f}%`")

# Display precision and recall in a table format
st.markdown("### **Class-wise Metrics**")
metrics_table = """
| Class      | Precision | Recall |
|------------|-----------|--------|
| Negative   | {:.2f}    | {:.2f} |
| Neutral    | {:.2f}    | {:.2f} |
| Positive   | {:.2f}    | {:.2f} |
""".format(
    model_metrics["Precision"]["Negative"], model_metrics["Recall"]["Negative"],
    model_metrics["Precision"]["Neutral"], model_metrics["Recall"]["Neutral"],
    model_metrics["Precision"]["Positive"], model_metrics["Recall"]["Positive"]
)
st.markdown(metrics_table)