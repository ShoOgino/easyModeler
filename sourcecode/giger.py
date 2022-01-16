from src.Config.config import config
import os
import datetime

if __name__ == '__main__':
    # config(タスクの設定)を更新
    config.pathConfigFile              = os.path.abspath(__file__)
    config.project                     = "egit"
    config.release                     = 2
    config.purpose                     = [
        config.Purpose.searchHyperParameter,
        config.Purpose.buildModel,
        config.Purpose.testModel
    ]
    config.typesInput                  = [
        config.TypeInput.codemetrics, config.TypeInput.processmetrics
    ]
    config.pathsDirSampleTrain         = [
        "dataset/{}/R{}_r_train".format(config.project, config.release)
    ]
    config.pathsDirSampleTest          = [
        "dataset/{}/R{}_r_test".format(config.project, config.release)
    ]
    config.algorithm = "RF"
    config.isCrossValidation           = False
    config.splitSize4CrossValidation   = 5
    config.epochs4EarlyStopping        = 10
    config.period4HyperParameterSearch = 60*1
    config.id                          = os.path.splitext(os.path.basename(config.pathConfigFile))[0] + "_" + config.project + "_" + str(config.release)
    config.pathDirOutput               = os.path.dirname(os.path.abspath(__file__)) + "/results/" + config.id + "_"+str(datetime.datetime.today().strftime("%Y%m%d_%H%M%S"))



    from src.Experiment.Experiment import Experiment
    from src.Dataset.DatasetGiger import DatasetGiger
    from src.Model.ModelRF import ModelRF
    from src.Node.RecordGiger import RecordGiger
    config.classDataset = DatasetGiger
    config.classRecord = RecordGiger
    config.classModel = ModelRF
    experiment = Experiment()
    experiment.run()