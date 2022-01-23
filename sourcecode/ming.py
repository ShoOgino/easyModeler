from src.Config.config import config
import os
import datetime

if __name__ == '__main__':
    # config(タスクの設定)を更新
    config.pathConfigFile              = os.path.abspath(__file__)
    config.project                     = "egit"
    config.release                     = 2
    config.purpose                     = [
        #config.Purpose.searchHyperParameter,
        #config.Purpose.buildModel,
        config.Purpose.testModel
    ]
    config.pathsDirSampleTrain         = [
        "dataset/{}_/R{}_r_train".format(config.project, config.release)
    ]
    config.pathsDirSampleTest          = [
        "dataset/{}_/R{}_r_test".format(config.project, config.release)
    ]
    config.isCrossValidation           = False
    config.splitSize4CrossValidation   = 5
    config.epochs4EarlyStopping        = 10
    config.period4HyperParameterSearch = 60*1
    config.id                          = os.path.splitext(os.path.basename(config.pathConfigFile))[0] + "_" + config.project + "_" + str(config.release)
    config.pathDirOutput               = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/result/" + config.id + "_"+str(datetime.datetime.today().strftime("%Y%m%d_%H%M%S"))
    config.device                      = "cuda:0"
    config.pathModel = r"C:\Users\login\data\workspace\easyModeler\sourcecode\results\ming_egit_2_20220122_053023\parameters"


    from src.Data.DatalotMingSequence import DatalotMingSequence
    config.classDatalot = DatalotMingSequence
    from src.Node.RecordMingSequence import RecordMingSequence
    config.classRecord = RecordMingSequence
    from src.Model.ModelDNNTensorflow import ModelDNNTensorflow
    config.classModel = ModelDNNTensorflow

    from src.Experiment.Experiment import Experiment
    experiment = Experiment()
    experiment.run()