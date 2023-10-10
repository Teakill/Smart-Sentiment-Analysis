import torch
from models.PhoBertClassifierV1 import CustomPhoBERTModel
from transformers import AutoTokenizer
import json

# Load configurations
with open("config.json", "r") as file:
    config = json.load(file)

# Initialize model and load trained weights
model_path = config["models"]["weights_path"]
model = CustomPhoBERTModel()
model.load_state_dict(torch.load(model_path))
model.eval()

# Move model to device
device = torch.device(config["device"] if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize tokenizer
tokenizer_path = config["models"]["tokenizer_path"]
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Feature names for each classifier
feature_names = ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']

# Test on custom text
custom_text = "Được đặt trên một chiếc tàu nổi giữa đầm mang tới không gian thơ mộng..."  # (truncated for brevity)

encoded = tokenizer(custom_text, return_tensors='pt', padding='max_length', max_length=256, truncation=True)
input_ids = encoded['input_ids'].to(device)
attention_mask = encoded['attention_mask'].to(device)

# Get predictions
with torch.no_grad():
    predictions = model(input_ids, attention_mask)

# Convert predictions to probabilities (assuming Softmax was used)
probabilities = [pred.softmax(dim=1) for pred in predictions]

# Print the probabilities and the class with the highest probability
for i, probs in enumerate(probabilities):
    max_prob, max_idx = torch.max(probs, dim=1)
    print(f"Feature: {feature_names[i]}, Probabilities: {probs}, Highest Probability Class: {max_idx.item()} with probability {max_prob.item()}")
