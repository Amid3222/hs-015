import pandas as pd
from sklearn import clone
from sklearn.pipeline import Pipeline

from DataUtils.DataManager import DataManager
from config import omegaconfig as conf
from models import ModelsManager as mm
from training import Validater as v
from utils import utils
import numpy as np
import matplotlib.pyplot as plt
import joblib
from DataUtils import DataManager


class PipelineRunner:

    def __init__(self):
        self.data_manager = DataManager.DataManager()

    def start_pipline(self):
        print("pipline started")
        self._run()
        # self._evaluate_best()
        # self._create_submission()

    def _run(self):
        data_manager = self.data_manager
        data = data_manager.get_base_preprocessed_train_data()

        model_manager = mm.ModelsManager()
        validator = v.Validater()

        best_model_name = None
        best_rmse = 100
        best_model = None

        print("start k-fold process...")

        for model, model_name in model_manager.iterate_all_models():

            print(f"validating model {model_name}...")

            full_pipeline = self.create_pipeline(model, model_name)

            if str(model_name).startswith("catboost"):
                validator.k_fold_catboost(dataframe=data, model_params=model.get_params(deep=True),
                                          model_class=model.__class__)

            scores, best_fold_model = validator.k_fold(dataframe=data, model=full_pipeline)

            print(f"model {model_name} score {np.mean(scores)}")

            utils.save_results_csv(model=model_name, params=model.get_params(deep=True), cv_median=np.mean(scores),
                                   cv_std=np.std(scores))

            print(f"model {model_name} saved to csv")

            if np.mean(scores) < best_rmse:
                best_rmse = np.mean(scores)
                best_model_name = model_name
                best_model = model
                print(f"Best model updated {best_model_name}, mean RMSE: {best_rmse}")

        print(f"Best model: {best_model_name}, mean RMSE: {best_rmse}")
        joblib.dump(best_model, 'model.joblib')

    def create_pipeline(self, model, model_name) -> Pipeline:
        return Pipeline(steps=[
            ("preprocess", self.data_manager.get_preprocess_pipline()),
            (model_name, model),
        ])

    def _create_submission(self):
        print("creating test submission on best model")
        d = DataManager.DataManager()

        train_data = self.data_manager.get_base_preprocessed_train_data()

        target = conf.get_global_conf().params.target_column_name

        X, y = utils.split_data_pd(train_data, target)

        loaded_model = joblib.load('model.joblib')
        m = clone(loaded_model)

        full_pipeline = self.create_pipeline(m, "best_model")

        full_pipeline.fit(X, y)

        test = d.get_full_processed_test_data()

        ids_col = self.data_manager.original_df["test"]['Id']

        predictions = full_pipeline.predict(test)

        sub = pd.DataFrame({
            'Id': ids_col,
            'SalePrice': np.expm1(predictions)
        })
        sub.to_csv('submisson.csv', index=False)
        print(f"Submission saved as submisson.csv")

    def _evaluate_best(self):
        loaded_model = joblib.load('model.joblib')

        print(loaded_model)

        train_data = self.data_manager.get_base_preprocessed_train_data()

        target = conf.get_global_conf().params.target_column_name

        X_train, X_test, y_train, y_test = utils.test_train_split(*utils.split_data_pd(train_data, target))

        m = clone(loaded_model)

        full_pipeline = self.create_pipeline(m, "best_model")

        full_pipeline.fit(X_train, y_train)

        y_p = full_pipeline.predict(X_test)

        validator = v.Validater()
        rmse = validator.RMSE(y_p, y_test)
        print(f"Val RMSE on best model: {rmse}")

    def _show_model_stats(self):

        df = pd.read_csv(conf.get_global_conf().params.save_result_to)
        print("Models statistic")

        labels = df.iloc[:, 0].astype(str)
        scores = df.iloc[:, 2]

        sorted_idx = scores.argsort()[::-1]
        labels_sorted = labels.iloc[sorted_idx]
        scores_sorted = scores.iloc[sorted_idx]

        plt.figure(figsize=(16, 8))

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(scores_sorted)))

        bars = plt.bar(range(len(labels_sorted)), scores_sorted, color=colors, edgecolor='black', linewidth=0.5)

        for i, (bar, v) in enumerate(zip(bars, scores_sorted)):

            plt.text(bar.get_x() + bar.get_width() / 2,
                     v + 0.003,
                     f'{v:.3f}',
                     ha='center', va='bottom',
                     fontsize=7,
                     fontweight='bold')

            label_text = labels_sorted.iloc[i]
            if len(label_text) > 25:
                label_text = label_text[:22] + '...'

            plt.text(bar.get_x() + bar.get_width() / 2,
                     0.001,
                     label_text,
                     ha='center', va='bottom',
                     fontsize=7,
                     rotation=90,
                     color='black')

        plt.xlabel('Models', fontsize=12, fontweight='bold')
        plt.ylabel('CV Score', fontsize=12, fontweight='bold')
        plt.title('Model Performance Comparison (Sorted by CV Score)', fontsize=14, fontweight='bold')

        plt.xticks([])

        plt.grid(axis='y', alpha=0.3, linestyle='--')

        y_min = scores_sorted.min() - 0.02
        y_max = scores_sorted.max() + 0.02
        plt.ylim(y_min, y_max)

        mean_score = scores_sorted.mean()
        plt.axhline(y=mean_score, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                    label=f'Mean: {mean_score:.3f}')
        plt.legend(loc='lower right')

        plt.tight_layout()
        plt.show()

        print(f" BEST MODEL: {labels_sorted.iloc[0]} → {scores_sorted.iloc[0]:.4f}")
