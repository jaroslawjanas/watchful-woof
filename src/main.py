import json
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from custom_dataset import CustomDataset
from text_vectorization import TextVectorization
from model import TransformerModel, LightningModelWrapper
import lightning as L
from callbacks import TrainingStatus
from lightning.pytorch.callbacks import RichProgressBar

from standardize import standardize_parallel
from utils import raw_data_stats, separate_data


def main():
    # Environment
    seed = 27
    random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    texts = standardize_parallel(texts)

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
    persistent = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch,
        pin_memory=pin_memory,
        persistent_workers=persistent
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,

        num_workers=num_workers,
        prefetch_factor=prefetch,
        pin_memory=pin_memory,
        persistent_workers=persistent
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch,
        pin_memory=pin_memory,
        persistent_workers=persistent
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

    samples = standardize_parallel(samples)

    vectorize_layer(samples)

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
        vectorize_layer=vectorize_layer,
    ).to(device)

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
    min_epochs = 10
    max_epochs = 50
    precision = "16-mixed"

    # Train
    pos_weights = torch.tensor([0, 50, 500], device=device)
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    # loss = nn.BCELoss()

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    lr_scheduler = {
        "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2), # get_last_lr()
        "monitor": "val_loss",  # Name of the metric to monitor
        "interval": "epoch",
        "frequency": 1,
    }

    # model
    lightning_model = LightningModelWrapper(
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

    # train model
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        precision=precision,
        callbacks=[TrainingStatus(), RichProgressBar(leave=True)],
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    trainer.fit(lightning_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
