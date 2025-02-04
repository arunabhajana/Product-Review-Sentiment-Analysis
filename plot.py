import torch
from transformers import pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Load the model and tokenizer
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
emotion_classifier = pipeline("text-classification", model=model_name)

# Sample data (replace with your dataset)
texts = [
    "I am so happy with my purchase!",
    "This is the worst product I have ever bought.",
    "I feel great after using this.",
    "I am very angry about this product.",
    "It is an okay product, nothing special."
]

# True labels (based on your dataset, these should match the emotions you're predicting)
true_labels = ["joy", "anger", "joy", "anger", "neutral"]

# Predict the emotions
predictions = [emotion_classifier(text)[0]['label'] for text in texts]

# Print the predictions to check what labels are returned
print("Predicted labels:", predictions)

# Update the emotion_map based on the model's return format
emotion_map = {
    "LABEL_0": "anger",
    "LABEL_1": "fear",
    "LABEL_2": "joy",
    "LABEL_3": "love",
    "LABEL_4": "sadness",
    "LABEL_5": "surprise",
    "LABEL_6": "neutral"
}

# If the predictions are returned as labels (e.g., "joy" directly), no need to map them
# Check if the labels are already in the form you expect:
if isinstance(predictions[0], str):
    predicted_labels = predictions  # If it's already in label format like 'joy', 'anger'
else:
    # If they are numeric (like 'LABEL_0'), we need to map them
    predicted_labels = [emotion_map[pred] for pred in predictions]

# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=["anger", "fear", "joy", "love", "sadness", "surprise", "neutral"])

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["anger", "fear", "joy", "love", "sadness", "surprise", "neutral"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Emotion Classification")
plt.show()
