import torch
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoTokenizer
from datasets.review_dataset import ReviewScoreDataset
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import scoring
from models.PhoBertClassifierV1 import CustomPhoBERTModel
from functools import partial
from utils.preprocessors import Preprocessing
import json

# Load the configurations
with open("config.json", "r") as file:
    config = json.load(file)

class ModelTrainer:

    def __init__(self, config):
        self.config = config

        self.device = torch.device(config["device"] if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])

        preprocessor = Preprocessing('data/external/vietnamese-stopwords-dash.txt', 'libs/vncorenlp/VnCoreNLP-1.2.jar')
        self.train_dataset = ReviewScoreDataset(config["paths"]["train_data"], self.tokenizer, preprocessor)
        self.val_dataset = ReviewScoreDataset(config["paths"]["val_data"], self.tokenizer, preprocessor)

        train_collate_fn = partial(self.train_dataset.collate_fn)
        val_collate_fn = partial(self.val_dataset.collate_fn)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=train_collate_fn)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=val_collate_fn)
        self.model = CustomPhoBERTModel().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["optimizer_lr"])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        for i, (input_ids_batch, attention_mask_batch, scores_batch) in enumerate(self.train_dataloader):
            input_ids_batch = input_ids_batch.to(self.device)
            attention_mask_batch = attention_mask_batch.to(self.device)
            scores_batch = scores_batch.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(input_ids_batch, attention_mask_batch)
            scores_batch = scores_batch.long()

            # Calculate loss
            total_loss = sum(self.criterion(output, scores_batch[:, i]) for i, output in enumerate(outputs))
            epoch_loss += total_loss.item()

            # Backward pass
            total_loss.backward()

            # Update weights
            self.optimizer.step()

            # Optionally print loss every n batches
            if i % 10 == 0:
                print(f'Batch: {i}, Loss: {total_loss.item()}')

        return epoch_loss / len(self.train_dataloader)

    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_true = []

        with torch.no_grad():
            for i, (input_ids_batch, attention_mask_batch, scores_batch) in enumerate(self.val_dataloader):
                input_ids_batch = input_ids_batch.to(self.device)
                attention_mask_batch = attention_mask_batch.to(self.device)
                scores_batch = scores_batch.to(self.device)

                outputs = self.model(input_ids_batch, attention_mask_batch)
                scores_batch = scores_batch.long()

                total_loss = sum(self.criterion(output, scores_batch[:, i]) for i, output in enumerate(outputs))
                val_loss += total_loss.item()

                batch_preds = [torch.argmax(output, dim=1).cpu().numpy() for output in outputs]
                all_preds.extend(list(zip(*batch_preds)))
                all_true.extend(scores_batch.cpu().numpy().tolist())

        avg_val_loss = val_loss / len(self.val_dataloader)

        gt = np.array(all_true)
        pred = np.array(all_preds)
        aspect_scores = scoring.get_score_modified(gt, pred)

        return avg_val_loss, aspect_scores

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self.train_epoch()

            # Evaluation phase
            val_loss, aspect_scores = self.evaluate()

            # Print training and validation results
            print(f'Epoch: {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
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

            # Update the learning rate based on validation loss
            self.scheduler.step(val_loss)

            # Save the model's state_dict (consider saving every few epochs or based on some criteria like best validation performance)
            model_save_path = self.config['paths']['model_save_path_template'].format(epoch + 1)
            torch.save(self.model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}\n')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == '__main__':
    trainer = ModelTrainer(config)
    trainer.train(num_epochs=30)
