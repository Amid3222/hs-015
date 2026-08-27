from copy import deepcopy

import pandas as pd
from sklearn.metrics import mean_squared_error, root_mean_squared_error

from sklearn.base import clone
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier, Pool
import numpy as np
from utils import utils
from config import omegaconfig as conf


class Validater:

    def k_fold(self, dataframe, model, kf=utils.get_kf_reg()):
        print("kfold process start...")
        X, y = utils.split_data_pd(dataframe, conf.get_global_conf().params.target_column_name)
        scores = []
        best_score = None
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_val = y.iloc[val_idx].copy()

            f_model = clone(model)

            f_model.fit(X_train, y_train)

            y_pred = f_model.predict(X_val)
            score = root_mean_squared_error(y_val, y_pred)

            scores.append(score)
            print(f"Fold {fold + 1}: {score:.4f}")

            if best_score is None or score < best_score:
                best_score = score
                best_model = f_model

        print(f"\nСредний score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        return scores, best_model

    def k_fold_catboost(self, dataframe, model_params, model_class, cat_features=None, kf=utils.get_kf_reg()):

        dataframe = dataframe.apply(lambda col: pd.to_numeric(col, errors='coerce'))
        X, y = utils.split_data_np(dataframe, conf.get_global_conf().params.target_column_name)
        scores = []
        best_score = 0
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train = X[train_idx].copy()
            X_val = X[val_idx].copy()
            y_train = y[train_idx].copy()
            y_val = y[val_idx].copy()




            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)

            modelc = model_class
            model = modelc(**model_params)

            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

            score = model.score(X_val, y_val)
            scores.append(score)
            print(f"Fold {fold + 1}: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model = model

        print(f"\nСредний score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        return scores, best_model

    def RMSE(self, y_t, y_p):
        return root_mean_squared_error(y_t, y_p)


def create_model(self, model_class, model_params):
    modelc = model_class
    model = modelc(**model_params)
    return model
