import torch.nn as nn
from transformers import AutoModel

# Complex classifier structure
def build_complex_classifier(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.BatchNorm1d(input_dim),
        nn.Linear(input_dim, 512),  # Increase dimensionality first
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),  # Reduce dimensionality
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.BatchNorm1d(256),
        nn.Linear(256, output_dim)
    )


class CustomPhoBERTModel(nn.Module):
    def __init__(self):
        super(CustomPhoBERTModel, self).__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")

        # Linear layer to reduce dimension
        self.fc1 = nn.Linear(768, 512)

        # BatchNorm Layer
        self.bn1 = nn.BatchNorm1d(512)

        # Activation
        self.relu = nn.ReLU()

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Separate, simpler classifiers for each feature
        self.classifiers = nn.ModuleList([nn.Linear(512, 6) for _ in range(6)])

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids, attention_mask=attention_mask)
        x = outputs.pooler_output
        x = self.dropout(x)

        # Simplified hidden layer
        x = self.relu(self.bn1(self.fc1(x)))

        # Separate Classifiers for each feature
        logits = [classifier(x) for classifier in self.classifiers]

        return logits