import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_data = self.inputs[index]
        label = self.labels[index]
        return input_data, label
