from comet_ml import Experiment
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
from Gyroscope_Dataset import SequenceDataset
from torch.utils.data import random_split, DataLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

load_dotenv()
api_key = os.getenv("COMET_API_KEY")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, latent_size=64):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_size//4, kernel_size=3, stride=2, padding=1),  # -> [B, 16, seq_len/2]
            nn.ReLU(),
            nn.Conv1d(hidden_size//4, hidden_size//2, kernel_size=3, stride=2, padding=1), # -> [B, 32, seq_len/4]
            nn.ReLU(),
            nn.Conv1d(hidden_size//2, hidden_size, kernel_size=3, stride=2, padding=1), # -> [B, 64, seq_len/8]
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_size, hidden_size//2, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [B, 32, seq_len/4]
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_size//2, hidden_size//4, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [B, 16, seq_len/2]
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_size//4, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> [B, 1, seq_len]
        )

    def forward(self, x):
        x = x.unsqueeze(1) 
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.squeeze(1)
        decoded = decoded[:,:201]
        return decoded


    def fit(self, number_of_epochs, train_loader, val_loader, experiment, patience=10, learning_rate=1e-3):
        criterion = nn.SmoothL1Loss()
        criterion2 = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.state_dict())
        epochs_no_improve = 0

        for epoch in range(number_of_epochs):
            print(f"\nEpoch {epoch + 1}/{number_of_epochs}")
            self.train()
            train_loss = self.train_phase(train_loader, criterion, criterion2, optimizer)

            self.eval()
            val_loss = self.val_phase(val_loader, criterion, criterion2)
            scheduler.step(val_loss)

            experiment.log_metric("train_loss", train_loss, step=epoch)
            experiment.log_metric("val_loss", val_loss, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 100 == 0:
                torch.save(self.state_dict(), f"Model\\Gyroscope\\4\\model_epoch_{epoch + 1}.pt")

            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

        self.load_state_dict(best_model_wts)
        torch.save(self.state_dict(), r"Model\Gyroscope\4\best_model.pt")

    def train_phase(self, loader, criterion, criterion2, optimizer):
        total_loss = 0.0
        self.train()

        for inputs, _ in loader:
            inputs = inputs.float()
            inputs = (inputs - inputs.mean()) / (inputs.std() + 1e-8)  # Standaryzacja

            optimizer.zero_grad()
            outputs = self(inputs)
            loss1 = criterion(outputs, inputs)
            loss2 = criterion2(outputs, inputs)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Train Loss: {total_loss:.4f}")
        return total_loss

    def val_phase(self, loader, criterion, criterion2):
        total_loss = 0.0
        self.eval()

        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.float()
                inputs = (inputs - inputs.mean()) / (inputs.std() + 1e-8)

                outputs = self(inputs)
                loss1 = criterion(outputs, inputs)
                loss2 = criterion2(outputs, inputs)
                total_loss += loss1.item() +loss2.item()

        print(f"Validation Loss: {total_loss:.4f}")
        return total_loss

    def test_and_visualize(self, loader, device):
        self.eval()
        self.to(device)

        with torch.no_grad():
            for i, (inputs, _) in enumerate(loader):
                inputs = inputs.to(device).float()
                norm_inputs = (inputs - inputs.mean()) / (inputs.std() + 1e-8)
                outputs = self(norm_inputs)

                norm_inputs = norm_inputs[0].cpu().numpy()
                output_seq = outputs[0].cpu().numpy()
                error_seq = norm_inputs - output_seq

                plt.figure(figsize=(12, 4))
                plt.plot(norm_inputs, label="Input", color="blue")
                plt.plot(output_seq, label="Reconstruction", color="red", linestyle="--")
                plt.plot(error_seq, label="Error", color="green", linestyle=":")
                plt.title(f"Sample {i+1}: Input vs Reconstruction")
                plt.xlabel("Time Step")
                plt.legend()
                plt.tight_layout()
                plt.grid(True)
                plt.show()

                if i >= 10:
                    break

                
if __name__ == "__main__":

    experiment = Experiment(
        api_key = os.getenv("COMET_API_KEY"),
        project_name="gyroscope"
    )

    # === Parametry ===
    base_dir = "Dane\\Gyroscope"
    data_type = SequenceDataset.DataType.GOOD
    train_ratio = 0.8
    batch_size = 64

    # === Dataset pełny ===
    full_dataset = SequenceDataset(base_dir, data_type)

    # === Podział na train/val ===
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(42)  # dla powtarzalności
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)


    # === Loadery ===
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = LSTMAutoencoder(hidden_size= 16, latent_size= 8)
    model.fit(200, train_loader, val_loader, experiment, patience=80, learning_rate=3e-4)
    model.test_and_visualize(val_loader, device)

    data_type = SequenceDataset.DataType.WRONG
    full_dataset = SequenceDataset(base_dir, data_type)
    test_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.test_and_visualize(test_dataloader, device)