from comet_ml import Experiment
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
from VoiceCommandsDataset import AudioDataset, ListDataset
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



load_dotenv()
api_key = os.getenv("COMET_API_KEY")



class VoiceCommandsClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=6, dropout=0.3, bidirectional=False):
        super(VoiceCommandsClassifier, self).__init__()

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
        if x.shape[1] == 1:
            x = x.permute(0, 2, 1)
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
                torch.save(self.state_dict(), f"Scripts\\Model\\VoiceCommands\\1\\model_epoch_{epoch + 1}.pt")
                print(f"Model checkpoint saved at epoch {epoch + 1}.")

            # Early stopping
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break

        # Zapisz najlepszy model na koniec
        self.load_state_dict(best_model_wts)
        torch.save(self.state_dict(), r"Scripts\Model\VoiceCommands\1\best_model.pt")
        print(f"Best model saved with accuracy: {best_val_acc:.2f}%")


    def train_phase(self, loader, criterion, optimizer):
        total_loss = 0.0
        correct = 0
        total = 0

        self.train()

        for inputs, speaker, content in loader:
            # content.shape = [batch_size]
            inputs = inputs.float()
            content = content.long()  # wymagane przez CrossEntropyLoss

            optimizer.zero_grad()

            outputs = self(inputs)  # outputs.shape = [batch_size, num_classes]
            loss = criterion(outputs, content)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == content).sum().item()
            total += content.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total * 100
        print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy

    def val_phase(self, loader, criterion):
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, speaker, content in loader:
                inputs = inputs.float()
                content = content.long()

                outputs = self(inputs)
                loss = criterion(outputs, content)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == content).sum().item()
                total += content.size(0)


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

                x_np = inputs.squeeze(0).cpu().numpy()

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
        project_name="voice-commands"
    )
    batch_size = 4
    seq_len = 84319
    channels = 1

    x = torch.randn(batch_size, seq_len, channels)

    model = VoiceCommandsClassifier(input_size=1, num_classes=42)
    print(model(x).shape)  # shape: [batch, 6]
    
    dataset_path = "Data\\VoiceCommands\\Gloskomendy"
    dataset = AudioDataset(root_dir=dataset_path)

    train_list, val_list, test_list = dataset.get_splits()
    
    train_dataset = ListDataset(train_list)
    val_dataset = ListDataset(val_list)
    test_dataset = ListDataset(test_list)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model.fit(800, train_loader, val_loader, experiment, patience=80, learning_rate=3e-4)
    model.test_and_visualize(test_loader, ["none", "green", "blue", "black", "red", "orange"])
from comet_ml import Experiment
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
from VoiceCommandsDataset import AudioDataset, ListDataset
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



load_dotenv()
api_key = os.getenv("COMET_API_KEY")



class VoiceCommandsClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=6, dropout=0.3, bidirectional=False):
        super(VoiceCommandsClassifier, self).__init__()

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
        if x.shape[1] == 1:
            x = x.permute(0, 2, 1)
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
                torch.save(self.state_dict(), f"Scripts\\Model\\VoiceCommands\\1\\model_epoch_{epoch + 1}.pt")
                print(f"Model checkpoint saved at epoch {epoch + 1}.")

            # Early stopping
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break

        # Zapisz najlepszy model na koniec
        self.load_state_dict(best_model_wts)
        torch.save(self.state_dict(), r"Scripts\Model\VoiceCommands\1\best_model.pt")
        print(f"Best model saved with accuracy: {best_val_acc:.2f}%")


    def train_phase(self, loader, criterion, optimizer):
        total_loss = 0.0
        correct = 0
        total = 0

        self.train()

        for inputs, speaker, content in loader:
            # content.shape = [batch_size]
            inputs = inputs.float()
            content = content.long()  # wymagane przez CrossEntropyLoss

            optimizer.zero_grad()

            outputs = self(inputs)  # outputs.shape = [batch_size, num_classes]
            loss = criterion(outputs, content)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == content).sum().item()
            total += content.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total * 100
        print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy

    def val_phase(self, loader, criterion):
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, speaker, content in loader:
                inputs = inputs.float()
                content = content.long()

                outputs = self(inputs)
                loss = criterion(outputs, content)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == content).sum().item()
                total += content.size(0)


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

                x_np = inputs.squeeze(0).cpu().numpy()

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
        project_name="voice-commands"
    )
    batch_size = 4
    seq_len = 84319
    channels = 1

    x = torch.randn(batch_size, seq_len, channels)

    model = VoiceCommandsClassifier(input_size=1, num_classes=42)
    print(model(x).shape)  # shape: [batch, 6]
    
    dataset_path = "Data\\VoiceCommands\\Gloskomendy"
    dataset = AudioDataset(root_dir=dataset_path)

    train_list, val_list, test_list = dataset.get_splits()
    
    train_dataset = ListDataset(train_list)
    val_dataset = ListDataset(val_list)
    test_dataset = ListDataset(test_list)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model.fit(800, train_loader, val_loader, experiment, patience=80, learning_rate=3e-4)
    model.test_and_visualize(test_loader, ["none", "green", "blue", "black", "red", "orange"])