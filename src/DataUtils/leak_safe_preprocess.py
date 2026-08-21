import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder


def build_encoding_pipeline(target_enc_features, num_features, passthrough_features=None):

    transformers = [
        ("target_enc", TargetEncoder(target_type="continuous"), target_enc_features),
        ("num_impute", SimpleImputer(strategy="mean"), num_features),
    ]

    remainder = "drop"
    if passthrough_features:
        transformers.append(("passthrough", "passthrough", passthrough_features))

    return ColumnTransformer(transformers=transformers, remainder=remainder)
