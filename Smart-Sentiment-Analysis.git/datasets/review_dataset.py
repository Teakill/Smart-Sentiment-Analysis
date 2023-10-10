import torch
from torch.utils.data import Dataset
import pandas as pd
from utils.preprocessors import Preprocessing

class ReviewScoreDataset(Dataset):
    def __init__(self, filename, tokenizer, preprocessor):
        self.data = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review = self.data.iloc[idx, 0]
        scores_values = self.data.iloc[idx, 1:7].astype(float).values
        scores = torch.tensor(scores_values, dtype=torch.float)
        return review, scores

    def collate_fn(self, batch):
        reviews, scores = zip(*batch)

        # Preprocess each review using preprocessor.process
        preprocessed_reviews = [self.preprocessor.process(review) for review in reviews]

        inputs = self.tokenizer(preprocessed_reviews, padding=True, truncation=True, max_length=128,
                                return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        return input_ids, attention_mask, torch.stack(scores)