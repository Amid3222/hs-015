from pathlib import Path

from sklearn.pipeline import Pipeline

from config import omegaconfig as conf
import pandas as pd
import re
import os
import numpy as np
from .base_prepop_funcs import pipline_funcs, cat_to_targetenc, num_features, cat_to_onehot, cat_to_ordinal
from .leak_safe_preprocess import build_encoding_pipeline


class DataManager:
    def __init__(self, path=conf.get_global_conf().params.path_to_data):
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / "data"
        self.indexes = None
        self.original_df = {"train": pd.read_csv(data_dir / "train.csv"),

                            "test": pd.read_csv(data_dir / "test.csv")}

        self.transformer = None

    def get_pipline(self):
        pass

    def get_base_preprocessed_train_data(self):
        train_data = self.original_df["train"]

        train_data = self.base_preprocess(train_data)

        return train_data

    def get_full_processed_test_data(self):
        test_data = self.original_df["test"]
        test_data = self.base_preprocess(test_data)

        pipeline = Pipeline(steps=[
            ("preprocess", self.transformer)])

        return pipeline.transform(test_data)

    def base_preprocess(self, data):
        df = data.copy()

        for func in pipline_funcs:
            df = func(df=df)

        return df

    def get_preprocess_pipline(self):
        tr = build_encoding_pipeline(num_features=num_features, target_enc_features=cat_to_targetenc,
                                     onehot_features=cat_to_onehot, ordinal_features=cat_to_ordinal)
        self.transformer = tr
        return tr

    def get_preprocess_pipline2(self):
        tr = build_encoding_pipeline(num_features=num_features, onehot_features=self.original_df["train"].select_dtypes(
            include='object').columns.tolist(), target_enc_features=[], ordinal_features=[])
        self.transformer = tr
        return tr

    def get_df_info(self):
        print(self.original_df.shape)
        print(self.original_df.info())
        print(self.original_df.describe())
        print(self.original_df.isnull().sum(axis=0))
