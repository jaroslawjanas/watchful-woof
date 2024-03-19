import torch.nn as nn
from embeddings import TokenAndPositionEmbedding
from transformer_block import TransformerBlock
import lightning as L


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


class LightningModelWrapper(L.LightningModule):
    def __init__(self, model, loss, optimizer, lr_scheduler):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def accuracy(outputs, labels, threshold=0.5):
        predictions = (outputs > threshold).float()
        accuracy = (predictions == labels).float().mean()
        return accuracy

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return [self.optimizer], [self.lr_scheduler]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = self.loss(y_hat, y)
        val_accuracy = self.accuracy(y_hat, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        test_loss = self.loss(y_hat, y)
        test_accuracy = self.accuracy(y_hat, y)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy", test_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)