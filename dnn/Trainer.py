import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR


def get_classes(logits):
    return torch.argmax(logits, dim=1)


def check_diff(arr, tol):
    arr = np.array(arr)
    return np.max(arr) - np.min(arr) < tol


def save_model(model):
    torch.save(model.state_dict(), "dnn/checkpoints/model_weights.pth")


def get_accuracy(classes, y):
    return (classes == y).sum().item() / y.shape[0]


class Trainer:
    def __init__(self, model: nn.Module, optimizer, ):
        self.model = model
        self.optimizer = optimizer

    def set_best_metric_model(self):
        self.model.load_state_dict(torch.load("dnn/checkpoints/model_weights.pth"))
        self.model.eval()

    def fit(self, epochs, train_loader, val_loader, validating=True, device=torch.device("cuda"),
            train_metric_accumulating=10, last_e_changes=50):

        loss_history = []
        val_epoch_metric_history = []
        train_metrics_history = []
        last_saved_best_metric = {"metric" : 0, "epoch": 0}
        loss_fn = nn.CrossEntropyLoss()
        scheduler = StepLR(self.optimizer, step_size=40, gamma=0.1)
        for e in range(epochs):
            self.model.train()

            epoch_loss = 0.0
            true_pred = 0
            total_samples = 0

            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)

                logits = self.model(X)
                self.optimizer.zero_grad()
                loss = loss_fn(logits, y)

                batch_size = X.shape[0]
                epoch_loss += loss.item() * batch_size
                total_samples += batch_size

                loss.backward()
                self.optimizer.step()

                classes = get_classes(logits)
                t_pred = (y == classes).sum().item()
                true_pred += t_pred

                if e % train_metric_accumulating == 0:
                    batch_acc = t_pred / batch_size
                    print('Epoch: {}, Loss: {}, Batch_Accuracy: {}'.format(e, loss.item(), batch_acc))

            avg_epoch_loss = epoch_loss / total_samples
            train_metrics_history.append(true_pred / total_samples)
            loss_history.append(avg_epoch_loss)

            if validating:
                acc = self.validate(val_loader, device)

                val_epoch_metric_history.append(acc)
                print(f"Val accuracy: {acc} Epoch {e}")

                if acc > last_saved_best_metric["metric"]:
                    last_saved_best_metric["metric"] = acc
                    last_saved_best_metric["epoch"] = e
                    save_model(self.model)
                    print(f"New best acc model saved Acc: {acc}")

            scheduler.step()

        return loss_history, train_metrics_history, val_epoch_metric_history, last_saved_best_metric

    def validate(self, val_loader, device):
        self.model.eval()
        t_pred = 0
        with torch.no_grad():
            for i, (X, y) in enumerate(val_loader):
                X, y = X.to(device), y.to(device)
                logits = self.model(X)
                classes = get_classes(logits)
                t_pred += (classes == y).sum().item()
        self.model.train()  # возвращаем модель в режим обучения
        return t_pred / len(val_loader.dataset)

    def predict(self, loader, device):
        self.model.eval()
        result = []
        with torch.no_grad():
            for i, X in enumerate(loader):
                X = X.to(device)
                logits = self.model(X)
                classes = torch.argmax(logits, dim=1)
                result.extend(classes.cpu().numpy())

        self.model.train()
        return result
