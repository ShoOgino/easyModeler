from distutils.command import config
from src.Config.config import config
import numpy
import random
class Dataset():
    def __init__(self):
        self.records = []
    def splitToTwoGroups(self, numOfFolds):
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
    def normalize(self):
        node = config.classRecord
        node.calcMax()
        for record in self.records:
            record.normalize()
    def standardize(self):
        node = config.classRecord
        node.calcMean()
        node.calcStandard()
        for record in self.records:
            record.standardize()
    def provideBatchTensorFlow(self):
        pass
    def provideBatchPytorch(self):
        pass
    #abstract
    def loadRecords():
        pass
