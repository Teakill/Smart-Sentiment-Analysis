import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch
from transformers import AutoTokenizer
from utils.preprocessors import Preprocessing
from models.PhoBertClassifierV1 import CustomPhoBERTModel
from utils.scoring import get_score_modified


import json

# Load configurations
with open("config.json", "r") as file:
    config = json.load(file)

device = torch.device(config["device"] if torch.cuda.is_available() else 'cpu')


preprocessor = Preprocessing('data/external/vietnamese-stopwords-dash.txt',
                                 'libs/vncorenlp/VnCoreNLP-1.2.jar')

# Load the test data
test_data_path = r"C:\Users\broke\PycharmProjects\Smart-Sentiment-Analysis\data\preprocessed_data\test_data.csv"
test_df = pd.read_csv(test_data_path)

# Initialize the tokenizer and device
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Load your trained model
model = CustomPhoBERTModel()
model.load_state_dict(torch.load('models/PhoBertTuneV1 weights.pth', map_location=torch.device('cpu')))
model.to(device)
model.eval()


# Function to get model's predictions
def predict(text):
    text = preprocessor.process(text)
    input_ids = tokenizer.encode(text, return_tensors="pt", padding='max_length', max_length=256, truncation=True)
    attention_mask = (input_ids != tokenizer.pad_token_id).int()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    predicted_probs = [F.softmax(output, dim=1) for output in outputs]
    predicted_classes = [torch.argmax(output, dim=1).item() for output in predicted_probs]
    return np.array(predicted_classes)


# Lists to store true labels and predictions
true_labels = test_df.iloc[:, 1:].values.tolist()  # Convert directly to lists
predicted_labels = []

# Get predictions
for text in test_df['Review']:
    predicted_labels.append(predict(text))

# Convert lists to numpy arrays
gt = np.array(true_labels)
pred = np.array(predicted_labels)

# Calculate scores using get_score_modified function
aspect_scores = get_score_modified(gt, pred)

# Print the scores
for aspect, metrics in aspect_scores.items():
    print(f"\nAspect: {aspect}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Calculate the mean score across all aspects
mean_scores = {}
for metric in ['precision', 'recall', 'f1_score', 'r2_score', 'competition_score']:
    mean_scores[metric] = np.mean([metrics[metric] for _, metrics in aspect_scores.items()])

# Print mean scores
print("\nMean Scores Across Aspects:")
for metric, value in mean_scores.items():
    print(f"{metric}: {value:.4f}")