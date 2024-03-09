import json
import random
from typing import Callable
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from custom_dataset import CustomDataset
from text_vectorization import TextVectorization
from embeddings import TokenAndPositionEmbedding
from transformer_block import TransformerBlock
from model import TransformerModel
from train import Trainer

from standardize import standardize_parallel
from utils import cuda_device_status, raw_data_stats, separate_data


def main():
    # Environment
    seed = 27
    random.seed(seed)

    cuda_device_status()

    # Load data
    file_path = "./data/ahk_dataset_v2.json"
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    # Raw data stats
    raw_data_stats(data)

    # Separate Data
    texts, labels = separate_data(data)

    print("Separated data:")
    print(f"\t{texts[7100]}")
    print(f"\t{labels[7100]}")
    print("\n")

    print("Standardizing text data...")
    start_time = time.time()
    texts = standardize_parallel(texts, workers=8)

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    print(f"Done! Took {elapsed_time} seconds.")

    print("\n")
    print("Sample:")
    print(f"\t{texts[7100]}")
    print("Standardized text:")
    print(f"\t{texts[7100]}")
    print("\n")

    # Build Dataset
    ahk_dataset = CustomDataset(texts, labels)

    # Random Split
    validation_split = 0.1
    test_split = 0.1

    train_split = 1 - validation_split - test_split
    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        ahk_dataset,
        [train_split, validation_split, test_split],
        generator=generator
    )

    print("Dataset proportions:")
    print(f"Training: \t\t{len(train_dataset)}")
    print(f"Validation: \t{len(val_dataset)}")
    print(f"Test: \t\t\t{len(test_dataset)}")
    print("\n")

    # Data Loaders
    batch_size = 400
    num_workers = 8
    prefetch = 40  # batches
    pin_memory = True  # https://pytorch.org/docs/stable/data.html#memory-pinning

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch,
        pin_memory=pin_memory
    )

    valid_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch,
        pin_memory=pin_memory
    )

    # Text Vectorization
    vectorize_layer = TextVectorization(
        max_vocabulary=60000,
        max_tokens=150
    )

    vectorize_layer.adapt(train_dataset)
    vectorize_layer.vocabulary_size()

    samples = [
        "trvger7 implements the russian style of trap metal perfectly, where he's enjoyable mnogoznaal's music is "
        "literally magical triplesixdelete is the best trap metal producer",
        "That's why I'm going 3090, 24 GB of vram, but the 4070 ti is equal or greater than by 1-5% depending on the "
        "title to the 3090 ti so some performance loss but fuck 12 GB of vram utter scam",
        "ya I always hear about that, have yet to have it happen then again I don't uaually do wild stuff fortnite, "
        "I kinda did sketchy stuff, everything else was just keybind stuff",
        "I went from a full desk as a mouse pad to a giant one, now to one that fits in my laptop bag But I am not a "
        "pro gamer so ya",
        "I've played cs2 yesterday for the first time, and cs overall after like an year or two.  I didn't follow too "
        "much info cs2 related, I can't say I went in blind but I only knew the most big changes (like smoke).  It's "
        "nice, but I wasn't very impressed. That said I don't know what would make me impressed. I had some trouble "
        "adjusting to the gameplay style. Idk, peeker's advantage feels way bigger than it was before. Also I used to "
        "play at 128 tickrate, and shots in the new system doesn't seem to be as accurate as they were. Could be my "
        "aim being crap after such a long period without playing, but I swear too many eagle shots missed when they "
        "were spot on in the model's head."
    ]

    for idx, sample in enumerate(samples):
        samples[idx] = standardizer(sample)

    vectorize_layer(samples)

    # Test Embeddings
    random_tokens = torch.randint(
        low=0,
        high=vectorize_layer.vocabulary_size(),
        size=(batch_size, 150),
        dtype=torch.int32
    )

    test_emb = TokenAndPositionEmbedding(150, vectorize_layer.vocabulary_size(), 16)
    test_emb_out = test_emb(random_tokens)

    # Test Transformer Block
    test_tblock = TransformerBlock(
        embed_dim=16,
        num_heads=2,
        ff_dim=32,
        dropout=0.1
    )

    test_tblock_out = test_tblock(test_emb_out)

    print("Transformer block test:")
    print(test_tblock_out.shape)
    print(test_tblock_out.dtype)
    print("\n")

    # Model params
    max_tokens = 150
    vocab_size = vectorize_layer.vocabulary_size()
    embed_dim = 16  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    # Model
    model = TransformerModel(
        max_tokens=max_tokens,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        vectorize_layer=vectorize_layer
    )

    # Model Test
    model.forward(
        [
            "This is a test",
            "trvger7 implement the russian style of trap metal perfect where he is enjoy mnogozna music is liter "
            "magic triplesixdelet is the best trap metal produc"
        ]
    )

    # Training params
    learning_rate = 0.00003
    weight_decay = 0.001
    patience = 8
    epochs = 50

    # Train
    pos_weights = torch.tensor([0, 50, 500])
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    # loss = nn.BCELoss()

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    model_trainer = Trainer(
        epochs=epochs,
        loss=loss,
        optimizer=optimizer,
        patience=patience
    )

    model_trainer.train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader
    )

    # Post training test
    sample = "Can someone help with an ahk script?"
    std_sample = standardizer(sample)
    print(std_sample)

    model([std_sample])


if __name__ == "__main__":
    main()
