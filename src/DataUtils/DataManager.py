from pathlib import Path

from config import omegaconfig as conf
import pandas as pd
import re
import os
import numpy as np
from base_prepop_funcs import pipline_funcs
from leak_safe_preprocess import build_encoding_pipeline

class DataManager:
    def __init__(self, path=conf.get_global_conf().params.path_to_data):
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / "data"
        self.indexes = None
        self.original_df = {"train": pd.read_csv(data_dir / "train.csv"),

                            "test": pd.read_csv(data_dir / "test.csv")}

    def get_pipline(self):
        pass

    def get_base_preprocessed_data(self, mode_key="train"):
        data = self.original_df[mode_key]

        if mode_key == "test":
            self.indexes = data["PassengerId"]

        data = self.base_preprocess(data)
        return data

    def base_preprocess(self, data):
        data = data.copy()
        kwargs = {"df": data}

        for func in pipline_funcs:
            func(**kwargs)

        return data

    def leak_safe_preprocess_wrapper(self):
        pass


    def get_df_info(self):
        print(self.original_df.shape)
        print(self.original_df.info())
        print(self.original_df.describe())
        print(self.original_df.isnull().sum(axis=0))
