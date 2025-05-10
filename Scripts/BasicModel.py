from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("COMET_API_KEY")

class TorchModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 device='cuda', patience=5, save_path='model_checkpoint.pt',
                 experiment_name='default-run', project_name='my-project', api_key='YOUR_API_KEY'):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.patience = patience
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        self.train_history = []
        self.val_history = []

        # Logger setup
        self.experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            auto_param_logging=True,
            auto_metric_logging=False
        )
        self.experiment.set_name(experiment_name)

        # Optional: log model architecture
        self.experiment.set_model_graph(model)

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        self.experiment.log_metric("train_loss", epoch_loss, step=epoch)
        self.experiment.log_metric("train_accuracy", epoch_acc, step=epoch)

        return epoch_loss, epoch_acc

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        self.experiment.log_metric("val_loss", epoch_loss, step=epoch)
        self.experiment.log_metric("val_accuracy", epoch_acc, step=epoch)

        return epoch_loss, epoch_acc

    def fit(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            self.train_history.append({'loss': train_loss, 'acc': train_acc})
            self.val_history.append({'loss': val_loss, 'acc': val_acc})

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.save_path)
                self.experiment.log_model("best_model", self.save_path)
                print("Checkpoint saved.")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print("Early stopping triggered.")
                    break

        self.save_training_history()

    def save_training_history(self, filename='training_history.json'):
        history = {
            'train': self.train_history,
            'val': self.val_history
        }
        with open(filename, 'w') as f:
            json.dump(history, f)

        self.experiment.log_asset(file_data=json.dumps(history), file_name=filename)

    def load_best_model(self):
        self.model.load_state_dict(torch.load(self.save_path))

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return correct / total
