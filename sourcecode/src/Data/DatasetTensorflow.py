from src.Model.ModelDNNTensorflow import ModelDNNTensorflow
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy
import math
class DatasetTensorflow(Sequence):
    def __init__(self, records, collate=None, trial=None, hp = None):
        self.records = records
        if(trial!=None):
            self.sizeOfBatch = trial.suggest_int('sizeOfBatch', 100, 100)
        elif(hp!=None):
            self.sizeOfBatch = hp["sizeOfBatch"]
        else:
            self.sizeOfBatch = 100
        self.collate = collate
    def __len__(self):
        return math.ceil(len(self.records) / self.sizeOfBatch)
    def __getitem__(self, index):
        #todo x={nameInplt1: ..., nameInput2: ...}, y={nameOutput1: ...}の形式で渡す。ただし、x,yは辞書で良い。
        x={}
        y={}
        batchRecords = self.records[index*self.sizeOfBatch:(index+1)*self.sizeOfBatch]
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

        #-1でパディング
        vectorCommitss = pad_sequences(vectorCommitss, dtype='float32', padding='post')

        x["ids"] = ids
        x["input"] = vectorCommitss
        y["output"] = ys
        return (vectorCommitss, numpy.array(ys))
    def getIdsYX(self):
        ids = []
        y={}
        x={}

        ys = []
        vectorCommitss = []
        for record in self.records:
            vectorCommits=[]
            for vectorCommit in record.children[0].children:
                vectorCommits.append(vectorCommit.children)
            vectorCommitss.append(vectorCommits)
            ys.append(record.label)
            ids.append(record.id)

        #vectorCommitss = pad_sequences(vectorCommitss[0:2], dtype='float32', padding='post')

        x = vectorCommitss
        y = ys
        return ids, y, x
    def getIds(self):
        ids = []
        for record in self.records:
            ids.append(record.id)
        return ids
    def getX(self):
        x={}
        vectorCommitss=[]
        for record in self.records:
            vectorCommits=[]
            for vectorCommit in record.children[0].children:
                vectorCommits.append(vectorCommit.children)
            vectorCommitss.append(vectorCommits)
        x["VectorCommits"]=vectorCommitss
        return x
    def getY(self):
        ys = []
        for record in self.records:
            ys.append(record.label)
        return ys