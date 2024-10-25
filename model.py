import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load dataset (replace with your file path)
file_path = '/content/drive/MyDrive/restaurant_feedback.csv'
df = pd.read_csv(file_path)

# Data preview and basic info
print("Data Preview:")
print(df.head())
print("\nData Info:")
print(df.info())

# Check for missing values
df.dropna(subset=['Feedback'], inplace=True)

# Initialize VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(feedback):
    blob = TextBlob(feedback)
    vader_score = vader.polarity_scores(feedback)['compound']
    textblob_score = blob.sentiment.polarity
    return vader_score, textblob_score

# Apply sentiment analysis and add features
df[['Vader_Sentiment', 'TextBlob_Sentiment']] = df['Feedback'].apply(lambda x: pd.Series(analyze_sentiment(x)))
df['Sentiment_Average'] = (df['Vader_Sentiment'] + df['TextBlob_Sentiment']) / 2
df['Sentiment_Label'] = df['Sentiment_Average'].apply(lambda x: 'Positive' if x > 0 else ('Neutral' if x == 0 else 'Negative'))

# Plot sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Sentiment_Label', palette='viridis')
plt.title('Customer Feedback Sentiment Distribution')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Reviews')
plt.show()

# Prepare text features using TF-IDF
tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
X = tfidf.fit_transform(df['Feedback']).toarray()
y = df['Sentiment_Label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate model
y_pred = rf_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:]
features = tfidf.get_feature_names_out()

plt.figure(figsize=(10, 6))
plt.title("Top 10 Important Features")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# Average sentiment score
average_sentiment = df['Sentiment_Average'].mean()
print("\nAverage Sentiment Score:", average_sentiment)
