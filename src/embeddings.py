import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, max_len, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.token_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        self.pos_emb = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=embed_dim
        )

        self.positions = torch.arange(0, max_len, device=self.device)

    def forward(self, x):
        positions = self.pos_emb(self.positions)
        x = self.token_emb(x)
        return x + positions
