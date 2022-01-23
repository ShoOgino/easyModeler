from src.Data.DatasetPytorch import DatasetPytorch
from src.Data.DatasetTensorflow import DatasetTensorflow
from src.Model.ModelDNNPytorch import ModelDNNPytorch
from src.Model.ModelDNNTensorflow import ModelDNNTensorflow
from src.Model.ModelRF import ModelRF
from src.Config.config import config
from src.Logger.wrapperLogger import wrapperLogger
logger = wrapperLogger.setup_logger(__name__, config.getPathFileLog())
from src.Node.VectorCommit import VectorCommit
from src.Node.VectorCommits import VectorCommits
from src.Data.Datalot import Datalot
import json
import os
import glob
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from src.Config.config import config
import random

class DatalotMingSequence(Datalot):
    def __init__(self, pathsRecord, isTest):
        super().__init__()
        self.pathsRecord = []
        self.setPathsRecord(pathsRecord)
        self.loadRecords()
        if(not isTest):
            self.identifyColumnsUseless()
        self.removeColumnsUseless()
        self.standardize(isTest)
        print()
    def setPathsRecord(self, pathsDir):
        for path in pathsDir:
            if(os.path.isdir(path)):
                pathsSearched = glob.glob(path+"/**/*.json", recursive=True)
                self.pathsRecord.extend(pathsSearched)
    def loadRecords(self):
        def checkLengthVectorCommit():
            with open(self.pathsRecord[0], encoding="utf-8") as fSample4Train:
                recordJson = json.load(fSample4Train)
                for vectorCommit in recordJson["commitsOnModuleInInterval"]["commitsOnModule"].values():
                     VectorCommit.numOfElements = len(
                         vectorCommit["vectorSemanticType"]+
                         vectorCommit["vectorAuthor"]+
                         vectorCommit["vectorInterval"]+
                         vectorCommit["vectorType"]+
                         vectorCommit["vectorCodeChurn"]+
                         vectorCommit["vectorCochange"]
                     )
                     break
            print(VectorCommit.numOfElements)
        checkLengthVectorCommit()
        def loadARecord(pathRecord):
            with open(pathRecord, encoding="utf-8") as fSample4Train:
                recordJson = json.load(fSample4Train)
                if(len(recordJson["commitsOnModuleInInterval"]["commitsOnModule"])==0):
                    return
                vectorsCommit = []
                for vectorCommit in recordJson["commitsOnModuleInInterval"]["commitsOnModule"].values():
                    vectorsCommit.append(
                         vectorCommit["vectorSemanticType"]+
                         vectorCommit["vectorAuthor"]+
                         vectorCommit["vectorInterval"]+
                         vectorCommit["vectorType"]+
                         vectorCommit["vectorCodeChurn"]+
                         vectorCommit["vectorCochange"]
                    )
                record = config.classRecord(
                    recordJson["path"],
                    recordJson["commitsOnModuleAll"]["isBuggy"],
                    vectorsCommit
                )
                self.records.append(record)
        for pathSample4Train in self.pathsRecord:
            loadARecord(pathSample4Train)
    def createDatasets(self, numOfFolds, indexOfFold, classModel, trial=None, hp=None):
        # 分けたい場合: 0 < numOfFolds
        # 分けたくない場合: numOfFolds=0
        #     すべてのレコードをtrain用レコードとしても、 valid用レコードとしても利用する。
        def splitRecords(numOfFolds):
            datasets_train_valid = []
            recordsBuggy    = []
            recordsNotBuggy = []
            for record in self.records:
                if(int(record.label)==1):
                    recordsBuggy.append(record)
                elif(int(record.label)==0):
                    recordsNotBuggy.append(record)
            random.seed(0)
            random.shuffle(recordsBuggy)
            random.shuffle(recordsNotBuggy)
            for i in range(numOfFolds):
                dataset_train_valid = []
                dataset4Train=[]
                dataset4Valid=[]
                validBuggy = recordsBuggy[(len(recordsBuggy)//numOfFolds)*i:(len(recordsBuggy)//numOfFolds)*(i+1)]
                validNotBuggy = recordsNotBuggy[(len(recordsNotBuggy)//numOfFolds)*i:(len(recordsNotBuggy)//numOfFolds)*(i+1)]
                validBuggy = random.choices(validBuggy, k=len(validNotBuggy))
                dataset4Valid.extend(validBuggy)
                dataset4Valid.extend(validNotBuggy)
                random.shuffle(dataset4Valid)#最初に1, 次に0ばっかり並んでしまっている。

                trainBuggy = recordsBuggy[:(len(recordsBuggy)//numOfFolds)*i]+recordsBuggy[(len(recordsBuggy)//numOfFolds)*(i+1):]
                trainNotBuggy = recordsNotBuggy[:(len(recordsNotBuggy)//numOfFolds)*i]+recordsNotBuggy[(len(recordsNotBuggy)//numOfFolds)*(i+1):]
                trainBuggy = random.choices(trainBuggy, k=len(trainNotBuggy))
                dataset4Train.extend(trainBuggy)
                dataset4Train.extend(trainNotBuggy)
                random.shuffle(dataset4Train)#最初に1, 次に0ばっかり並んでしまっている。
                dataset_train_valid.append(dataset4Train)
                dataset_train_valid.append(dataset4Valid)
                datasets_train_valid.append(dataset_train_valid)
            return datasets_train_valid
        if(0 < numOfFolds):
            records4train, records4valid = splitRecords(numOfFolds)[indexOfFold]
        elif(0==numOfFolds):
            records4train = self.records
            records4valid = self.records
        else:
            raise Exception()
        if(classModel == ModelRF):
            pass
        if(classModel == ModelDNNPytorch):
            def collate_pytorch(batchRecords):
                class2input = {}
                ids = []
                ys = []
                vectorCommitss = []
                for record in batchRecords:
                    vectorCommits=[]
                    for vectorCommit in record.children[0].children:
                        vectorCommits.append(vectorCommit.children)
                    vectorCommitss.append(vectorCommits)
                    ys.append(record.label)
                    ids.append(record.id)
            
                vectorCommitss = [torch.tensor(vectorCommits).float() for vectorCommits in vectorCommitss]
                vectorCommitssLength = torch.tensor([len(vectorCommits) for vectorCommits in vectorCommitss])
                vectorCemmitssPadded = pad_sequence(vectorCommitss, batch_first=True)
                vectorCommitss = pack_padded_sequence(vectorCemmitssPadded, vectorCommitssLength, batch_first=True, enforce_sorted=False)
                
                class2input[VectorCommits.__name__] = vectorCommitss.to(config.device)
                ys = torch.tensor(ys).float().to(config.device)
            
                return ids, ys, class2input
            dataset4train = DatasetPytorch(records4train, collate = collate_pytorch, trial=trial, hp=hp)
            dataset4valid = DatasetPytorch(records4valid, collate = collate_pytorch, trial=trial, hp=hp)
            return dataset4train, dataset4valid
        if(classModel == ModelDNNTensorflow):
            dataset4train = DatasetTensorflow(records4train, trial=trial, hp=hp)
            dataset4valid = DatasetTensorflow(records4valid, trial=trial, hp=hp)
            return dataset4train, dataset4valid