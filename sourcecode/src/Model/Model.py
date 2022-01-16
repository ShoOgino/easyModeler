from src.Config.config import config
import optuna

class Model():
    def __init__(self):
        pass
    def loadHyperparameter():
        if(config.hyperparameter):
            return config.hyperparameter
        elif(config.pathDatabaseOptuna):
            # optunaデータベースから、最適なハイパーパラメータを読み出す
            study = optuna.create_study(study_name="optuna", storage='sqlite:///'+config.pathDirOutput + "/optuna.db", load_if_exists=True)
            return dict(study.best_params.items()^study.best_trial.user_attrs.items())
        else:
            #logger.error("Hyperparameter must be loaded.")
            raise Exception()
