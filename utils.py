import os
from ray.tune.logger import UnifiedLogger

def custom_logger_creator(log_path):

    def logger_creator(config):
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        return UnifiedLogger(config, log_path, loggers=None)

    return logger_creator