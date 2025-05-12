# Twitter Sentiment Analysis Application

This project is a real-time sentiment analysis tool for text data, built using Python, scikit-learn, and Streamlit. It leverages the Sentiment140 dataset to train a machine learning model that classifies input text as **Negative**, **Neutral**, or **Positive**.

## Features

- **Real-time Sentiment Prediction**: Enter any text and instantly see its predicted sentiment.
- **Pre-trained Model**: Uses a logistic regression model trained on the Sentiment140 dataset.
- **Web Interface**: Simple and interactive UI powered by Streamlit.
- **Model Performance Metrics**: The app dynamically displays the model's accuracy, precision, and recall for transparency.

## Project Structure

```
.
├── app.py                      # Streamlit web application
├── train_model.py              # Script to train and save the sentiment model
├── models/                     # Directory containing the saved model, vectorizer, and metrics
│   ├── sentiment_model.pkl     # Saved logistic regression model
│   ├── vectorizer.pkl          # Saved CountVectorizer
│   └── metrics.json            # Model performance metrics (accuracy, precision, recall)
├── data/                       # Directory for datasets
│   ├── testdata.manual.2009.06.14.csv  # Manual test dataset
│   └── training.1600000.processed.noemoticon.csv  # Sentiment140 dataset (not included)
└── README.md                   # Project documentation
```

## Scripts Overview

### `train_model.py`

This script is responsible for training the sentiment analysis model. It performs the following tasks:

1. Loads and preprocesses the datasets:
   - **Manual Test Dataset**: `testdata.manual.2009.06.14.csv`
   - **Sentiment140 Dataset**: `training.1600000.processed.noemoticon.csv`
2. Cleans the text data by removing URLs, mentions, hashtags, and non-alphabetic characters.
3. Combines the datasets and vectorizes the text using `CountVectorizer`.
4. Splits the data into training and testing sets.
5. Trains a logistic regression model on the training data.
6. Evaluates the model on the test data and generates performance metrics (accuracy, precision, recall).
7. Saves the following to the `models/` directory:
   - `sentiment_model.pkl`: The trained logistic regression model.
   - `vectorizer.pkl`: The fitted `CountVectorizer`.
   - `metrics.json`: A JSON file containing the model's performance metrics.
   - `metadata.json`: A JSON file containing metadata about the trained model (e.g., training date, dataset size, vectorizer details).

### `app.py`

This script runs the Streamlit web application for real-time sentiment analysis. It performs the following tasks:

1. Loads the pre-trained model (`sentiment_model.pkl`) and vectorizer (`vectorizer.pkl`) from the `models/` directory.
2. Dynamically loads the model's performance metrics from `metrics.json`.
3. Dynamically loads model metadata from `metadata.json`.
4. Provides a user-friendly interface where users can input text to predict its sentiment.
5. Displays the predicted sentiment as **Negative**, **Neutral**, or **Positive** in real time.
6. Shows the model's performance metrics (accuracy, precision, recall) for transparency.

## Model Evaluation & Performance Metrics

- The logistic regression model is trained on the Sentiment140 dataset.
- After training, the model is evaluated on a held-out validation set (20% of the training data).
- The following metrics are calculated and saved to `metrics.json`:
  - **Accuracy**
  - **Precision** (for Negative, Neutral, and Positive classes)
  - **Recall** (for Negative, Neutral, and Positive classes)
- These metrics are dynamically loaded and displayed in the app.

## Snapshots

- Below are example screenshots of the running web application:

### Input and Prediction

![Input and Prediction](images/input_prediction.png)

### Model Metrics

![Model Metrics](images/model_metrics.png)

## Acknowledgments

- [Streamlit](https://streamlit.io/): An open-source app framework for Machine Learning and Data Science projects, enabling the creation of interactive web applications with minimal effort.
- [scikit-learn](https://scikit-learn.org/): A powerful Python library for machine learning, providing tools for data preprocessing, model training, and evaluation.
- [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140): A dataset containing 1.6 million tweets, labeled for sentiment analysis, used to train the model.
- [Vectorization in Machine Learning](https://www.comet.com/site/blog/vectorization-in-machine-learning/): A blog post explaining the importance of vectorization in machine learning.
- [CountVectorizer Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html): Official scikit-learn documentation for `CountVectorizer`.
- [Linear Regression Using scikit-learn](https://www.geeksforgeeks.org/python-linear-regression-using-sklearn/): A tutorial on implementing linear regression using scikit-learn.
- [Saving Model Metadata](https://www.geeksforgeeks.org/saving-a-machine-learning-model/): A guide on saving and loading machine learning models in python.
