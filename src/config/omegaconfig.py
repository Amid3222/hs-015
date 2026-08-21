from omegaconf import OmegaConf
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # E:\d_temp\pp\src\config


def get_param_conf():
    return OmegaConf.load(os.path.join(CURRENT_DIR, "param_config.yaml"))


def get_global_conf():
    return OmegaConf.load(os.path.join(CURRENT_DIR, "global_config.yaml"))


global_params = get_global_conf()
model_params = get_param_conf()
