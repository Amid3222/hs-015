from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, \
    VotingClassifier, RandomForestRegressor, VotingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from config import omegaconfig as conf
from dnn.DNNSklearn import DNNRegressorSKLearn


def get_models():
    m = conf.get_param_conf().models

    return {
        "regression+baseline": Ridge(**m.regression.versions.baseline),
        "regression+l2": Lasso(**m.regression.versions.l2),
        "regression+ridge_l2": Ridge(**m.regression.versions.ridge_l2),  # Опционально
        "regression+elasticnet": ElasticNet(**m.regression.versions.elasticnet),  # Опционально

        "knn+default": KNeighborsRegressor(**m.knn.versions.default),

        "decision_tree+default": DecisionTreeRegressor(**m.decision_tree.versions.default),

        "random_forest+best": RandomForestRegressor(**m.random_forest.versions.best),

        "catboost+default": CatBoostRegressor(**m.catboost.versions.default),

        "dnn_regressor": DNNRegressorSKLearn(model_kwargs=m.dnn.model_kwargs, optimizer_kwargs=m.dnn.optimizer_kwargs),

        "xgboost+best": XGBRegressor(**m.xgboost.versions.best),

        "lightgbm+default": LGBMRegressor(**m.lightgbm.versions.default),

        "stacking+default": StackingRegressor(
            estimators=[
                ("knn", KNeighborsRegressor(**m.stacking.versions.default.base_models.knn)),
                ("tree", DecisionTreeRegressor(**m.stacking.versions.default.base_models.tree)),

            ],
            final_estimator=Ridge(alpha=1.0),
            cv=m.stacking.versions.default.cv,
            n_jobs=m.stacking.versions.default.n_jobs,
        ),

        "voting+default": VotingRegressor(
            estimators=[
                ("ridge", Ridge(alpha=1.0)),
                ("lasso", Lasso(alpha=0.1)),
                ("knn", KNeighborsRegressor(n_neighbors=5)),
                ("tree", DecisionTreeRegressor(max_depth=5, random_state=42))
            ],

        ),
    }
