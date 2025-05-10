from comet_ml import Experiment
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
from Dataloader_MB import CustomDataset, DataLoaderType, DataType
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



load_dotenv()
api_key = os.getenv("COMET_API_KEY")



class DrunknesClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=6, dropout=0.3, bidirectional=False):
        super(DrunknesClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,  # channels
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (hn, cn) = self.lstm(x) 
        last_hidden = hn[-1]  # shape: [batch, hidden]
        logits = self.fc(last_hidden)
        return logits
    
    def fit(self, number_of_epochs: int, train_loader, val_loader, experiment, patience: int = 10, learning_rate:float = 1e-3):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        experiment.log_parameters({
            "lr": 1e-3,
            "epochs": number_of_epochs,
            "optimizer": "Adam",
            "hidden_size": self.lstm.hidden_size,
            "num_layers": self.lstm.num_layers
        })

        best_val_acc = 0.0
        best_model_wts = copy.deepcopy(self.state_dict())
        epochs_no_improve = 0

        for epoch in range(number_of_epochs):
            print(f"\nEpoch {epoch + 1}/{number_of_epochs}")
            self.train()
            train_loss, train_acc = self.train_phase(train_loader, criterion, optimizer)

            self.eval()
            val_loss, val_acc = self.val_phase(val_loader, criterion)

            experiment.log_metric("train_loss", train_loss, step=epoch)
            experiment.log_metric("train_accuracy", train_acc, step=epoch)
            experiment.log_metric("val_loss", val_loss, step=epoch)
            experiment.log_metric("val_accuracy", val_acc, step=epoch)

            # Sprawdzenie poprawy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
                print(f"New best validation accuracy: {val_acc:.2f}%")
            else:
                epochs_no_improve += 1

            # Zapis co 10 epok
            if (epoch + 1) % 100 == 0:
                torch.save(self.state_dict(), f"Scripts\\Model\\DrunknessClassifier\\2\\model_epoch_{epoch + 1}.pt")
                print(f"Model checkpoint saved at epoch {epoch + 1}.")

            # Early stopping
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break

        # Zapisz najlepszy model na koniec
        self.load_state_dict(best_model_wts)
        torch.save(self.state_dict(), r"Scripts\Model\DrunknessClassifier\2\best_model.pt")
        print(f"Best model saved with accuracy: {best_val_acc:.2f}%")


    def train_phase(self, loader, criterion, optimizer):
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in loader:
            labels = labels.long()
            optimizer.zero_grad()

            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total * 100
        print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def val_phase(self, loader, criterion):
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                labels = labels.long()
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total * 100
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy
        
    # preds = torch.argmax(outputs, dim=1)  # shape [batch], dtype lon
    
    def test_and_visualize(self, loader, class_names=None):
        self.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in loader:
                outputs = self(inputs)
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.item())
                all_labels.append(labels.item())

        # Macierz pomyłek
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.show()

        # Wyświetl próbki
        print("\n--- Visualizing Individual Samples ---")
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(loader):
                outputs = self(inputs)
                preds = torch.argmax(outputs, dim=1)

                label = labels.item()
                pred = preds.item()

                print(f"\nSample {i+1}")
                print(f"True label: {class_names[label] if class_names else label}")
                print(f"Predicted:  {class_names[pred] if class_names else pred}")

                x_np = inputs.squeeze(0).cpu().numpy()  # [seq_len, channels]

                plt.figure(figsize=(12, 6))
                plt.title(f"Prediction: {pred} | Ground Truth: {label}")
                plt.imshow(x_np.T, aspect="auto", cmap="viridis", interpolation="nearest")
                plt.xlabel("Time Step")
                plt.ylabel("Channel")
                plt.colorbar(label="Signal Value")
                plt.show()

                if i >= 10:
                    break

if __name__ == "__main__":
    experiment = Experiment(
        api_key = os.getenv("COMET_API_KEY"),
        project_name="drunkness-classifier"
    )
    batch_size = 32
    seq_len = 200
    channels = 13

    x = torch.randn(batch_size, seq_len, channels)

    model = DrunknesClassifier(input_size=13)
    print(model(x).shape)  # shape: [batch, 6]
    
    dataset = CustomDataset(False, "Data\\opis_przejsc.csv", "C:\\Users\\Marcin\\Desktop\\Studia\\IoT\\Data", data_from_samples_ratio=3, data_lenght = 400 , random_state = 42, mode = DataLoaderType.POCKET, dataset_directory =r"C:\Users\Marcin\Desktop\Studia\IoT\Data\Tensory\POCKET20250427_172927", debug=False)
    dataset.set_datatype(DataType.TRAIN)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataset.set_datatype(DataType.VALIDATION)
    val_loader = DataLoader(dataset, batch_size=32)
    dataset.set_datatype(DataType.TEST)
    test_loader = DataLoader(dataset, batch_size=1)

    model.fit(800, train_loader, val_loader, experiment, patience=80, learning_rate=3e-5)
    model.test_and_visualize(test_loader, ["none", "green", "blue", "black", "red", "orange"])