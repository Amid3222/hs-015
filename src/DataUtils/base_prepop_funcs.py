import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

merge_to_ter_or_bin = {"Alley": None,
                       "LandSlope": 'Gtl',
                       "MiscFeature": None,
                       "Condition2": "Norm",
                       "RoofMatl": "CompShg",
                       "GarageCond": 'TA',
                       "PavedDrive": "Y",
                       }

rare_dict = {
    "LotConfig": ("Other", ["FR3"]),
    "Condition1": ("Other", ["RRNe", "RRNn", "RRAn", "RRAe", "PosN", "PosA"]),
    "RoofStyle": ("Other", ["Shed"]),
    "MSZoning": ("Other", ["C", "FV", "RH", "RL"]),
    "Exterior1st": ("Other", ["AsphShn", "BrkComm", "CBlock", "ImStucc", "Stone"]),
    "Exterior2nd": ("Other", ["CBlock", "Stone", "AsphShn"]),
    "ExterCond": ("Other", ["Ex", "Po"]),
    "Foundation": ("Other", ["Stone", "Wood"]),
    "BsmtCond": ("Other", ["Po"]),
    "HeatingQC": ("Other", ["Po"]),
    "Electrical": ("Other", ["FuseP", "Mix"]),
    "Functional": ("Other", ["Maj2", "Sev"]),
    "GarageType": ("Other", ["2Types"]),
    "GarageQual": ("Other", ["Ex", "Po"]),
    "GarageCond": ("Other", ["Ex"]),
    "SaleType": ("Other", ["CWD", "Con", "ConLI", "ConLw", "Oth"]),
    "SaleCondition": ("Other", ["AdjLand"])
}

drop_col = ['PoolQC', 'Street', 'Utilities', "Id"]

num_features = [
    "LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF",
    "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath",
    "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
    "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF",
    "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",
    "MiscVal", "MoSold", "YrSold",
]

target = 'SalePrice'

cat_to_ordinal = ['LotShape', 'LandContour', 'LandSlope', 'HouseStyle', 'ExterQual',
                  'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC',
                  'Electrical', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond',
                  'PavedDrive', 'Functional', 'GarageFinish', 'Fence', ]

cat_to_onehot = ['MSZoning', 'CentralAir', 'Condition1', 'Condition2', 'BldgType', 'RoofStyle', 'RoofMatl',
                 'MasVnrType', 'Foundation', 'Heating', 'Alley', 'Neighborhood', 'GarageType',
                 'MiscFeature', 'SaleType', 'SaleCondition', "OverallQual", "OverallCond"]

# num_cat = ["OverallQual", "OverallCond"]

cat_to_targetenc = ["LotConfig", "Exterior1st", "Exterior2nd", "MSSubClass"]

missing_category = "MISSING"


def log_target(df: pd.DataFrame):
    try:
        print("Логарифмирование таргета")
        df[target] = np.log1p(df[target])
    except KeyError:
        print(f"No target coluumn {target}")

    return df


def drop_outliers_by_id(df: pd.DataFrame):
    try:
        print("Удаление аномалий")
        df = df[~df['Id'].isin([524, 1299])]
    except KeyError:
        print(f"No column id")

    return df


def col_drop(df: pd.DataFrame):
    df = df.drop(labels=drop_col, axis=1)
    print("Дропанье колонок")
    return df


def mutate_cats(df: pd.DataFrame):
    df = df.copy()

    print("Конвертация категорий")

    for k, v in merge_to_ter_or_bin.items():
        _collapse_classes(df, k, v)

    df = _merge_classes_multi(mapping=rare_dict, df=df)
    return df


def _collapse_classes(df: pd.DataFrame, label, cat):
    """
    Схлопывает все классы признака `label`, кроме `category`, в `other_label`.
    NaN выделяется в отдельную, третью категорию .
    """

    if cat is None:
        col = df[label]
        is_nan = col.isna()

        result = pd.Series("Other", index=col.index, dtype=object)

        result[col.notna()] = True
        result[is_nan] = False

        df[label] = result
        return df

    col = df[label]

    is_target = col == cat
    is_nan = col.isna()

    result = pd.Series("Other", index=col.index, dtype=object)
    result[is_target] = cat
    result[is_nan] = missing_category

    df[label] = result
    return df


def _merge_classes_multi(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Принимает словарь вида:
        {"LotConfig": ("Other", ["FR3", "FR2"]), ...}
    и мержит для каждого признака сразу.
    """
    target_df = df.copy()
    for label, (new_class, classes_to_merge) in mapping.items():
        _merge_classes(target_df, label, new_class, classes_to_merge)
    return target_df


def _merge_classes(df: pd.DataFrame, label: str, new_class: str, classes_to_merge) -> pd.DataFrame:
    """
    Сливает перечисленные классы признака `label` в один новый класс `new_class`.
    Остальные значения не трогаются.
    """
    target_df = df
    target_df[label] = target_df[label].replace({old: new_class for old in classes_to_merge})
    return target_df


pipline_funcs = [log_target, drop_outliers_by_id, col_drop, mutate_cats]
