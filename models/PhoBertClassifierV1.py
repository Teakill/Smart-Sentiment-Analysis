import torch.nn as nn
from transformers import AutoModel
import torch

class CustomPhoBERTModel(nn.Module):
    def __init__(self):
        super(CustomPhoBERTModel, self).__init__()

        # Load PhoBERT model
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")

        # Ensure hidden states are output
        self.phobert.config.output_hidden_states = True

        # Freeze the first 4 transformer layers
        for i in range(4):
            for param in self.phobert.encoder.layer[i].parameters():
                param.requires_grad = False

        # The concatenated size of last 4 hidden states
        self.hidden_size = self.phobert.config.hidden_size * 4
        self.fc = nn.Linear(self.hidden_size, 512)
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Separate classifiers for each feature
        self.classifiers = nn.ModuleList([nn.Linear(512, 6) for _ in range(6)])

    def forward(self, input_ids, attention_mask):
        # Pass inputs through PhoBERT model
        outputs = self.phobert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        # Concatenate the last 4 hidden states
        concat_hidden = torch.cat(tuple(hidden_states[-4:]), dim=-1)
        x = torch.mean(concat_hidden, dim=1)

        x = self.dropout(x)
        x = self.relu(self.bn(self.fc(x)))

        logits = [classifier(x) for classifier in self.classifiers]

        return logits

class CustomPhoBERTModel_Mean_Max_Pooling(nn.Module):
    def __init__(self):
        super(CustomPhoBERTModel_Mean_Max_Pooling, self).__init__()

        # Load PhoBERT model
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")

        # Ensure hidden states are output
        self.phobert.config.output_hidden_states = True

        # Freeze the first 4 transformer layers
        for i in range(4):
            for param in self.phobert.encoder.layer[i].parameters():
                param.requires_grad = False

        # The concatenated size of last 4 hidden states (times 2 for concatenation of mean and max)
        self.hidden_size = self.phobert.config.hidden_size * 4 * 2
        self.fc = nn.Linear(self.hidden_size, 512)
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Separate classifiers for each feature
        self.classifiers = nn.ModuleList([nn.Linear(512, 6) for _ in range(6)])

    def forward(self, input_ids, attention_mask):
        # Pass inputs through PhoBERT model
        outputs = self.phobert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        # Concatenate the last 4 hidden states
        concat_hidden = torch.cat(tuple(hidden_states[-4:]), dim=-1)

        # Mean pooling
        mean_pooled = torch.mean(concat_hidden, dim=1)

        # Max pooling
        max_pooled, _ = torch.max(concat_hidden, dim=1)

        # Concatenate mean and max pooled results

        x = torch.cat((mean_pooled, max_pooled), dim=-1)

        x = self.dropout(x)
        x = self.relu(self.bn(self.fc(x)))

        logits = [classifier(x) for classifier in self.classifiers]

        return logits
