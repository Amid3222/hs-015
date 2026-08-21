from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import omegaconfig as conf
from models.models import get_models
from omegaconf import OmegaConf


class ModelsManager:
    def __init__(self, config_path=None):
        self.model_reg = get_models()
        self.param_config = conf.get_param_conf() if config_path is None else OmegaConf.load(config_path)

    def iterate_all_models(self):

        if conf.get_global_conf().params.use_best_model:
            model_name = conf.get_global_conf().params.best_model
            model = self.model_reg[model_name]

            yield from [(model, model_name)]
        else:
            for model_name, model in self.model_reg.items():
                yield model, model_name

    def get_model_by_name(self, name: str):
        return self.model_reg[name]
