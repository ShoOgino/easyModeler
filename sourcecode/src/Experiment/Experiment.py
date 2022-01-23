from src.Config.config import config
from src.Model.ModelDNNPytorch import ModelDNNPytorch
from src.Model.ModelDNNTensorflow import ModelDNNTensorflow
from src.Model.ModelRF import ModelRF
import shutil
import os

class Experiment():
    def __init__(self):
        self.datalot4train = None
        self.datalot4test = None
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
        self.buildModel()
        self.testModel()


    def buildDataset(self):
        self.datalot4train = config.classDatalot(config.pathsDirSampleTrain, isTest=False)
        self.datalot4test  = config.classDatalot(config.pathsDirSampleTest, isTest=True)

    def buildModel(self):
        self.model = config.classModel()
        self.model.build(
            doTraining = config.Purpose.buildModel in config.purpose,
            datalot4Train = self.datalot4train,
            doHyperparametertuning = config.Purpose.searchHyperParameter in config.purpose,
            doCrossValidation = config.isCrossValidation
        )

    def testModel(self):
        self.model.test(
            datalot4test = self.datalot4test
        )
