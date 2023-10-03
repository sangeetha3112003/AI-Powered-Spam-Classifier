# AI-Powered-Spam-Classifier
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load your labeled dataset (assuming you have a CSV file with 'text' and 'label' columns)
data = pd.read_csv('spam_data.csv')

# Data preprocessing: Tokenization and feature extraction
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(data['text'])
encoder = LabelEncoder()
y = encoder.fit_transform(data['label'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a machine learning model (Random Forest in this example)
spam_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
spam_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = spam_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{confusion}')
print(f'Classification Report:\n{classification_rep}')

# Now you can use this trained model to classify new messages
new_messages = ["Win a free iPhone now!", "Let's meet for lunch tomorrow."]
X_new = vectorizer.transform(new_messages)
predictions = spam_classifier.predict(X_new)

for message, prediction in zip(new_messages, predictions):
    predicted_label = encoder.inverse_transform([prediction])[0]
    print(f"Message: {message}")
    print(f"Predicted Label: {predicted_label}")
    print()
