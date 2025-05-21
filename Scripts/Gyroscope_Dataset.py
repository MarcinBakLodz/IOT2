import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from enum import Enum

class SequenceDataset(Dataset):
    class DataType(Enum):
        ALL = 1,
        GOOD = 2,
        WRONG = 3
    
    def __init__(self, base_dir, dataType = DataType.ALL):
        self.samples = []
        self.labels = []
        self.dataType = dataType

        for root, _, files in os.walk(base_dir):
            for file in files:
                if dataType == SequenceDataset.DataType.ALL:
                    if file.endswith(".txt") and ("good" in file or "wrong" in file):
                        path = os.path.join(root, file)
                        label = 0 if "good" in file else 1
                        self.samples.append(path)
                        self.labels.append(label)
                if dataType == SequenceDataset.DataType.GOOD:
                    if file.endswith(".txt") and ("good" in file):
                        path = os.path.join(root, file)
                        label = 0 if "good" in file else 1
                        self.samples.append(path)
                        self.labels.append(label)
                if dataType == SequenceDataset.DataType.WRONG:
                    if file.endswith(".txt") and ("wrong" in file):
                        path = os.path.join(root, file)
                        label = 0 if "good" in file else 1
                        self.samples.append(path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        with open(file_path, "r") as f:
            lines = f.readlines()

        sequence = torch.tensor([float(line.strip()) for line in lines], dtype=torch.float32)
        return sequence, self.labels[idx]

    def visualize_sample(self, idx):
        """Wizualizuje jedną sekwencję z datasetu"""
        sequence, label = self[idx]
        label_name = "good" if label == 0 else "wrong"
        file_path = self.samples[idx]

        plt.figure(figsize=(10, 4))
        plt.plot(sequence.numpy(), marker='o', linestyle='-')
        plt.title(f"Sample: {os.path.basename(file_path)} | Label: {label_name}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if "__main__" == __name__:
    dataset = SequenceDataset("Dane\\Gyroscope", SequenceDataset.DataType.GOOD)
    for i in range(100):
        dataset.visualize_sample(i)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)