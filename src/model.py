import torch.nn as nn
from embeddings import TokenAndPositionEmbedding
from transformer_block import TransformerBlock


class TransformerModel(nn.Module):
    def __init__(self, max_tokens, vocab_size, embed_dim, num_heads, ff_dim, vectorize_layer):
        super(TransformerModel, self).__init__()
        self.vectorize_layer = vectorize_layer
        self.embedding_layer = TokenAndPositionEmbedding(
            max_tokens,
            vocab_size,
            embed_dim
        )
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(embed_dim, 20)
        self.fc2 = nn.Linear(20, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.vectorize_layer(x)
        x = self.embedding_layer(x)
        x = self.transformer_block(x)
        x = self.global_avg_pooling(x.permute(0, 2, 1)).squeeze(2)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
