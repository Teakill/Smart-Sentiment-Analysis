import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch
from transformers import AutoTokenizer
from utils.preprocessors import Preprocessing
from models.PhoBertClassifierV1 import CustomPhoBERTModel
from utils.scoring import get_score_modified
import json

class ModelTester:
    def __init__(self, config_path):
        # Load configurations
        with open(config_path, "r") as file:
            self.config = json.load(file)

        self.device = torch.device(self.config["device"] if torch.cuda.is_available() else 'cpu')
        self.preprocessor = Preprocessing(self.config['paths']['stopwords'], self.config['paths']['vncorenlp'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["tokenizer_path"])

        # Load model
        self.model = self.config['paths']['evaluation_model']
        self.model.load_state_dict(torch.load(self.config['paths']['model_weights'], map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Load test data
        self.test_df = pd.read_csv(self.config['paths']['test_data'])

    def predict(self, text):
        text = self.preprocessor.process(text)
        input_ids = self.tokenizer.encode(text, return_tensors="pt", padding='max_length', max_length=256, truncation=True)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).int()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

        predicted_probs = [F.softmax(output, dim=1) for output in outputs]
        predicted_classes = [torch.argmax(output, dim=1).item() for output in predicted_probs]
        return np.array(predicted_classes)

    def evaluate(self):
        true_labels = self.test_df.iloc[:, 1:].values.tolist()
        predicted_labels = []

        for text in self.test_df['Review']:
            predicted_labels.append(self.predict(text))

        gt = np.array(true_labels)
        pred = np.array(predicted_labels)

        aspect_scores = get_score_modified(gt, pred)

        for aspect, metrics in aspect_scores.items():
            print(f"\nAspect: {aspect}")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

        mean_scores = {}
        for metric in ['precision', 'recall', 'f1_score', 'r2_score', 'competition_score']:
            mean_scores[metric] = np.mean([metrics[metric] for _, metrics in aspect_scores.items()])

        print("\nMean Scores Across Aspects:")
        for metric, value in mean_scores.items():
            print(f"{metric}: {value:.4f}")



if __name__ == '__main__':
    tester = ModelTester(config_path="config.json")
    tester.evaluate()