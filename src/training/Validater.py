from copy import deepcopy

from sklearn.base import clone
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier, Pool
import numpy as np
from utils import utils
from config import omegaconfig as conf


class Validater:

    def k_fold(self, dataframe, model, kf=utils.get_skf()):
        X, y = utils.split_data_pd(dataframe, conf.get_global_conf().params.target_column_name)
        scores = []
        best_score = 0
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_val = y.iloc[val_idx].copy()

            X_val, X_train = utils.fillnas(X_train, X_val, columns=["Age"])
            X_val, X_train = utils.clip(X_train, X_val)

            f_model = clone(model)

            f_model.fit(X_train, y_train)

            score = f_model.score(X_val, y_val)  # accuracy по умолчанию
            scores.append(score)
            print(f"Fold {fold + 1}: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model = f_model

        print(f"\nСредний score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        return scores, best_model

    def k_fold_catboost(self, dataframe, model_params, model_class, cat_features=None, kf=utils.get_skf()):

        X, y = utils.split_data_pd(dataframe, conf.get_global_conf().params.target_column_name)
        scores = []
        best_score = 0
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_val = y.iloc[val_idx].copy()

            X_val, X_train = utils.fillnas(X_train, X_val, columns=["Age"])
            X_val, X_train = utils.clip(X_train, X_val)
            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)

            modelc = model_class
            model = modelc(**model_params)

            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

            score = model.score(X_val, y_val)  # accuracy по умолчанию
            scores.append(score)
            print(f"Fold {fold + 1}: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model = model

        print(f"\nСредний score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        return scores, best_model

    def test_train_split_val(self, val_data_x, val_data_y, model):

        y_preds = model.predict(val_data_x)
        return utils.accuracy(y_preds, val_data_y)


def create_model(self, model_class, model_params):
    modelc = model_class
    model = modelc(**model_params)
    return model
