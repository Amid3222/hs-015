from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import TargetEncoder


def build_encoding_pipeline(target_enc_features, num_features, onehot_features, ordinal_features, passthrough_features=None):


    num_pipeline = Pipeline([
        ("num_impute", SimpleImputer(strategy="mean")),
        ("normalizer", RobustScaler())
    ])

    onehot_pipeline = Pipeline([
        ("cat_impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    ordinal_pipeline = Pipeline([
        ("cat_impute", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    transformers = [
        ("target_enc", TargetEncoder(target_type="continuous"), target_enc_features),
        ("onehot", onehot_pipeline, onehot_features),
        ("ordinal", ordinal_pipeline, ordinal_features),
        ("num_features", num_pipeline, num_features),
    ]

    if passthrough_features:
        transformers.append(("passthrough", "passthrough", passthrough_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")