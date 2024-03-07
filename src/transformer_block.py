import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Multihead self-attention
        attn_output, _ = self.attention(x, x, x)

        # Residual connection and normalization
        x = x + self.dropout(attn_output)
        x = self.norm_1(x)

        # Feed-forward network
        ffn_output = self.ffn(x)

        # Residual connection and normalization
        x = x + self.dropout(ffn_output)
        x = self.norm_2(x)

        return x
    