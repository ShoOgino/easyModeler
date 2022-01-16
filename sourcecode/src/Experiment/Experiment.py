from src.Config.config import config
from src.Model.ModelDNNPytorch import ModelDNNPytorch
from src.Model.ModelDNNTensorflow import ModelDNNTensorflow
from src.Model.ModelRF import ModelRF
import shutil
import os

class Experiment():
    def __init__(self):
        self.dataset4Train = None
        self.dataset4Test = None
        self.model = None

    def run(self):
        #実験結果フォルダを作成
        os.makedirs(config.pathDirOutput, exist_ok=True)
        # 実行コードを保存
        shutil.copy(config.pathConfigFile, config.pathDirOutput)
        # ハイパーパラメータチューニングを再開する場合、前回のoptunaDBをコピー
        if(config.pathDatabaseOptuna!=None):
            shutil.copy(config.pathDatabaseOptuna, config.pathDirOutput)

        self.buildDataset()
        self.buildModel(config.splitSize4CrossValidation)
        self.testModel()


    def buildDataset(self):
        self.dataset4Train = config.classDataset(config.pathsDirSampleTrain)
        self.dataset4Test  = config.classDataset(config.pathsDirSampleTest)

    def buildModel(self, numOfFolds):
        self.model = config.classModel()
        self.model.build(
            listOfTrainValid =
                self.dataset4Train.splitToTwoGroups(
                    numOfFolds
                ),
            doHyperparametertuning = config.Purpose.searchHyperParameter in config.purpose
        )

    def testModel(self):
        self.model.test(
            self.dataset4Test.records
        )
