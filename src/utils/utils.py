from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from config import omegaconfig as conf
import pandas as pd


def validate(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=conf.get_global_conf().params.random_state)
    score = cross_val_score(model, X, y, cv=skf)

    return score


def save_results_csv(path_f=None, **kwargs):
    if path_f is None:
        path_f = conf.get_global_conf().params.save_result_to
    print(f"saving... {kwargs}")
    new_data = kwargs
    df_new = pd.DataFrame([new_data])
    df_new.to_csv(path_f, mode='a', header=False, index=False)


def fillnas(X_train: pd.DataFrame, X_val: pd.DataFrame, columns):
    fill_dict = {}

    for col in columns:
        fill_dict[col] = X_train[col].median(skipna=True)

    X_train.fillna(fill_dict, inplace=True)
    X_val.fillna(fill_dict, inplace=True)

    return X_val, X_train

def clip(X_train,X_val):
    fare_upper = X_train['Fare'].quantile(0.95)
    X_train['Fare'] = X_train['Fare'].clip(upper=fare_upper)
    X_val['Fare'] = X_val['Fare'].clip(upper=fare_upper)

    return  X_val , X_train

def test_train_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.1,
        random_state=conf.get_global_conf().params.random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_skf(splits=5, r_state=conf.get_global_conf().params.random_state):
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=r_state)


def accuracy(y_pred, y_true):
    return (y_pred == y_true).sum() / y_true.shape[0]


def split_data_pd(data, target: str):
    X_data = data.drop(columns=target)
    y_data = data[target]

    return X_data, y_data


def split_data_np(data, target: str):
    X_data = data.drop(columns=target).values
    y_data = data[target].values

    return X_data, y_data
