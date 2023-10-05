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

        # Dropout Layer
        self.dropout = nn.Dropout(0.3)

        # Increase the dimension first before reducing (expansion and contraction idea)
        self.fc_expand = nn.Linear(768, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 384)

        # BatchNorm Layers
        self.bn_expand = nn.BatchNorm1d(1024)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(384)

        # Activation
        self.relu = nn.ReLU()

        # Additional Dropout
        self.dropout2 = nn.Dropout(0.3)

        # Branching Classifiers
        self.classifiers = nn.ModuleList([build_complex_classifier(384, 6) for _ in range(6)])

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids, attention_mask=attention_mask)
        x = outputs.pooler_output
        x = self.dropout(x)

        # Expand and Contract layers
        x = self.relu(self.bn_expand(self.fc_expand(x)))
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))

        # Dropout
        x = self.dropout2(x)

        # Separate Classifiers for each feature
        logits = [classifier(x) for classifier in self.classifiers]

        return logits