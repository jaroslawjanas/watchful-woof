import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, max_len, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()

        self.token_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        self.pos_emb = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=embed_dim
        )

    def forward(self, x):
        max_len = x.size(-1)
        positions = torch.arange(0, max_len, dtype=torch.int32, device=x.device)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
