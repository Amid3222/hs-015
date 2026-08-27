import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR


def check_diff(arr, tol):
    arr = np.array(arr)
    return np.max(arr) - np.min(arr) < tol


def save_model(model):
    torch.save(model.state_dict(), "dnn/checkpoints/model_weights.pth")


class DNNRegressor:
    def __init__(self, model: nn.Module, optimizer, ):
        self.model = model
        self.optimizer = optimizer

    def set_best_metric_model(self):
        self.model.load_state_dict(torch.load("dnn/checkpoints/model_weights.pth"))
        self.model.eval()

    def fit(self, epochs, train_loader, val_loader, validating=True, device=torch.device("cuda"),
            train_metric_accumulating=10, last_e_changes=50):

        loss_history = []

        last_saved_best_metric = {"metric": 0, "epoch": 0}
        loss_fn = nn.MSELoss()
        scheduler = StepLR(self.optimizer, step_size=40, gamma=0.1)

        for e in range(epochs):
            self.model.train()

            epoch_loss = 0.0
            total_samples = 0

            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)

                preds = self.model(X)
                self.optimizer.zero_grad()
                loss = loss_fn(preds, y)

                batch_size = X.shape[0]
                epoch_loss += loss.item() * batch_size
                total_samples += batch_size

                loss.backward()
                self.optimizer.step()

                if e % train_metric_accumulating == 0:
                    print('Epoch: {}, MSE: {}'.format(e, loss.item()))

            avg_epoch_loss = epoch_loss / total_samples
            loss_history.append(avg_epoch_loss)

            if validating:

                mse = self.validate(val_loader, device)

                print(f"Val accuracy: {mse} Epoch {e}")

                if mse < last_saved_best_metric["metric"]:
                    last_saved_best_metric["metric"] = mse
                    last_saved_best_metric["epoch"] = e
                    save_model(self.model)
                    print(f"New best mse model saved Acc: {mse}")

            scheduler.step()

        return loss_history, last_saved_best_metric

    def validate(self, val_loader, device):
        self.model.eval()
        loss = 0

        with torch.no_grad():
            for i, (X, y) in enumerate(val_loader):
                X, y = X.to(device), y.to(device)
                preds = self.model(X)
                loss_fn = nn.MSELoss()
                loss += loss_fn(preds, y)

        self.model.train()
        return loss / len(val_loader.dataset)

    def predict(self, loader, device="cpu"):
        self.model.eval()
        result = []
        with torch.no_grad():
            for i, X in enumerate(loader):
                X = X.to(device)
                preds = self.model(X)
                result.extend(preds.cpu().numpy())

        self.model.train()
        return result
