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

preprocessor = Preprocessing('data/external/vietnamese-stopwords-dash.txt',
                                 'libs/vncorenlp/VnCoreNLP-1.2.jar')
###
# Initialize the tokenizer (and preprocessor if needed)
model = CustomPhoBERTModel()

######
device = torch.device(config["device"] if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])



train_dataset = ReviewScoreDataset(config["train_data_path"], tokenizer, preprocessor)
train_collate_fn = partial(train_dataset.collate_fn)  # Bind the method to the instance

train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=train_collate_fn)

val_dataset = ReviewScoreDataset(config["val_data_path"], tokenizer, preprocessor)
val_collate_fn = partial(val_dataset.collate_fn)  # Bind the method to the instance
val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=val_collate_fn)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["optimizer_lr"])

####
# 3. Adjust training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for i, (input_ids_batch, attention_mask_batch, scores_batch) in enumerate(train_dataloader):
        input_ids_batch = input_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)
        scores_batch = scores_batch.to(device)

        # Forward pass
        outputs = model(input_ids_batch, attention_mask_batch)

        scores_batch = scores_batch.long()
        criterion = nn.CrossEntropyLoss()
        total_loss = sum(criterion(output, scores_batch[:, i]) for i, output in enumerate(outputs))
        epoch_loss += total_loss.item()  # Accumulate the loss for this epoch

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass
        total_loss.backward()

        # Update weights
        optimizer.step()

        if i % 10 == 0:
            print(f'Epoch: {epoch}, Batch: {i}, Loss: {total_loss.item()}')
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for i, (input_ids_batch, attention_mask_batch, scores_batch) in enumerate(val_dataloader):
            input_ids_batch = input_ids_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)
            scores_batch = scores_batch.to(device)

            outputs = model(input_ids_batch, attention_mask_batch)
            scores_batch = scores_batch.long()

            total_loss = sum(criterion(output, scores_batch[:, i]) for i, output in enumerate(outputs))
            val_loss += total_loss.item()

            batch_preds = [torch.argmax(output, dim=1).cpu().numpy() for output in outputs]
            all_preds.extend(list(zip(*batch_preds)))
            all_true.extend(scores_batch.cpu().numpy().tolist())

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    avg_val_loss = val_loss / len(val_dataloader)

    # Compute the scores using the get_score_modified function
    gt = np.array(all_true)
    pred = np.array(all_preds)
    aspect_scores = scoring.get_score_modified(gt, pred)

    # Print the scores
    print(f'Epoch: {epoch}, Avg Training Loss: {avg_epoch_loss:.4f}, Avg Validation Loss: {avg_val_loss:.4f}')
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

    scheduler.step(avg_val_loss)
    torch.save(model.state_dict(), 'models/PhoBertClassifierV2 weights')
    print('saved')

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    scheduler.step(avg_epoch_loss)