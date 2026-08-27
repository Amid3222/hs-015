import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from sklearn.base import BaseEstimator, RegressorMixin

from dnn import DNNLayers


class DNNRegressorSKLearn(BaseEstimator, RegressorMixin):

    def __init__(
            self,
            model_cls=DNNLayers,
            model_kwargs=None,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs=None,
            epochs=100,
            batch_size=32,
            device="cpu",
            validating=False,
            train_metric_accumulating=10,
            scheduler_step_size=40,
            scheduler_gamma=0.1,
            random_state=None
    ):
        self.is_fitted_ = None
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.validating = validating
        self.train_metric_accumulating = train_metric_accumulating
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.random_state = random_state

    def _set_random_seed(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    def _build_model(self, input_size):

        model_kwargs = self.model_kwargs or {}

        model_kwargs["input_dim"] = input_size

        self.model_ = self.model_cls(**model_kwargs)
        self.model_ = self.model_.to(self.device)

    def _build_optimizer(self):
        optimizer_kwargs = self.optimizer_kwargs or {}

        self.optimizer_ = self.optimizer_cls(
            self.model_.parameters(),
            **optimizer_kwargs,
        )

    def _make_train_loader(self, X, y):
        X_tensor = torch.as_tensor(
            np.asarray(X),
            dtype=torch.float32,
        )

        y_tensor = torch.as_tensor(
            np.asarray(y),
            dtype=torch.float32,
        ).reshape(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def fit(self, X, y):

        self._set_random_seed()

        self._build_model(input_size=X.shape[1])
        self._build_optimizer()

        train_loader = self._make_train_loader(X, y)

        loss_fn = nn.MSELoss()

        scheduler = StepLR(
            self.optimizer_,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )

        self.loss_history_ = []

        self.model_.train()

        for epoch in range(self.epochs):
            self.model_.train()

            epoch_loss = 0.0
            total_samples = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer_.zero_grad()

                preds = self.model_(X_batch)

                loss = loss_fn(preds, y_batch)

                loss.backward()
                self.optimizer_.step()

                batch_size = X_batch.shape[0]

                epoch_loss += loss.item() * batch_size

                total_samples += batch_size

            avg_epoch_loss = epoch_loss / total_samples

            self.loss_history_.append(avg_epoch_loss)

            if epoch % self.train_metric_accumulating == 0:
                print(
                    f"Epoch: {epoch}, "
                    f"MSE: {avg_epoch_loss}"
                )

            scheduler.step()

        self.is_fitted_ = True

        return self

    def predict(self, X):

        X_tensor = torch.as_tensor(
            np.asarray(X),
            dtype=torch.float32,
        )

        dataset = TensorDataset(X_tensor)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model_.eval()

        predictions = []

        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)

                preds = self.model_(X_batch)

                predictions.append(
                    preds.detach().cpu().numpy()
                )

        predictions = np.concatenate(predictions, axis=0)

        return predictions
