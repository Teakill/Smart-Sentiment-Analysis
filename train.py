import torch
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoTokenizer
from datasets.review_dataset import ReviewScoreDataset
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import scoring
from models.PhoBertClassifierV1 import CustomPhoBERTModel, CustomPhoBERTModel_Mean_Max_Pooling
from functools import partial
from utils.preprocessors import Preprocessing
import json
import os

# Load the configurations
with open("config.json", "r") as file:
    config = json.load(file)



class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"] if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config["models"]["tokenizer_path"])

        preprocessor = Preprocessing(config['data']['stopwords'], config['data']['vncorenlp'])

        self.train_dataset = ReviewScoreDataset(config["data"]["train"], self.tokenizer, preprocessor)
        self.val_dataset = ReviewScoreDataset(config["data"]["val"], self.tokenizer, preprocessor)

        train_collate_fn = partial(self.train_dataset.collate_fn)
        val_collate_fn = partial(self.val_dataset.collate_fn)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=config["training"]["batch_size"], shuffle=True,
                                           collate_fn=train_collate_fn)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                                         collate_fn=val_collate_fn)
####
        if self.config["models"]["evaluation_model"] == "CustomPhoBERTModel":
            self.model = CustomPhoBERTModel()

        elif self.config["models"]["evaluation_model"] == "CustomPhoBERTModel_Mean_Max_Pooling":
            self.model = CustomPhoBERTModel_Mean_Max_Pooling()

        # Load weights if the file exists
        if os.path.exists(self.config['models']['weights_path']):
            self.model.load_state_dict(torch.load(self.config['models']['weights_path'], map_location=self.device))
        else:
            print(f"Weights file at {self.config['models']['weights_path']} does not exist!")
#####

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["training"]["optimizer_lr"])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        self.criterion = nn.CrossEntropyLoss()

        # Load checkpoint if it exists
        self.checkpoint_path = config['models']['checkpoint_path']
        if os.path.isfile(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            print(
                f"Loaded checkpoint from epoch {self.start_epoch} with best validation loss: {self.best_val_loss:.4f}")
        else:
            self.start_epoch = 0
            self.best_val_loss = float('inf')
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
                print(f'Epoch: {self.start_epoch + 1}, Batch: {i}, Loss: {total_loss.item()}')

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
        best_dev_score = float('-inf')  # Initialize best_dev_score

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
            for metric in ['precision', 'recall', 'f1_score', 'r2_score', 'final_score']:
                mean_scores[metric] = np.mean([metrics[metric] for _, metrics in aspect_scores.items()])

            # Print mean scores
            print("\nMean Scores Across Aspects:")
            for metric, value in mean_scores.items():
                print(f"{metric}: {value:.4f}")

            # Update the learning rate based on validation loss
            self.scheduler.step(val_loss)

            # Save the model's state_dict if dev_score improved
            if mean_scores['final_score'] > best_dev_score:
                best_dev_score = mean_scores['final_score']
                model_save_path = self.config['models']['weights_path']
                torch.save(self.model.state_dict(), model_save_path)
                print(f'New best dev score: {best_dev_score:.4f}. Model saved to {model_save_path}\n')
            else:
                print(f'Dev score did not improve. Current best is {best_dev_score:.4f}\n')
            self.start_epoch += 1

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == '__main__':
    trainer = ModelTrainer(config)
    trainer.train(config['training']['num_epochs'])
