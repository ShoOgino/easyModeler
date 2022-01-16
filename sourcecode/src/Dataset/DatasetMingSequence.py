import imp
from random import sample
from src.Node.VectorCommit import VectorCommit
from src.Dataset.Dataset import Dataset

import json
import os
import glob
import torch

from src.Config.config import config

class DatasetMingSequence(Dataset, torch.utils.data.Dataset):
    def __init__(self, pathsRecord):
        super().__init__()
        self.pathsRecord = []
        self.setPathsRecord(pathsRecord)
        self.loadRecords()
        self.standardize()
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
    def toVector(self):
        recordsVector = []
        for record in self.records:
            rec = {}
            rec["id"] = record.id
            rec["y"] = record.label
            rec["x"] = record.children
            recordsVector.append(rec)
        self.records = recordsVector
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        return self.records[index]
    def getNumOfNegatives(self):
        return len([record for record in self.records if record["y"]==0])
    def getNumOfPositives(self):
        return len([record for record in self.records if record["y"]==1])
    def collate_fn(self, batch):
        ids, asts, astseqs, codemetricss, commitgraphs, commitseqs, processmetricss, ys = list(zip(*batch))

        if(config.checkASTExists()):
            pass

        if(config.checkASTSeqExists()):
            astseqs = [torch.tensor(astseq).float() for astseq in astseqs]
            astseqsLength = torch.tensor([len(astseq) for astseq in astseqs])
            astseqsPadded = pad_sequence(astseqs, batch_first=True)
            astseqs = pack_padded_sequence(astseqsPadded, astseqsLength, batch_first=True, enforce_sorted=False)

        if(config.checkCodeMetricsExists()):
            codemetricss = torch.tensor(codemetricss).float()

        if(config.checkCommitGraphExists()):
            pass

        if(config.checkCommitSeqExists()):
            commitseqs = [torch.tensor(commitseq).float() for commitseq in commitseqs]
            commitseqsLength = torch.tensor([len(commitseq) for commitseq in commitseqs])
            commitseqsPadded = pad_sequence(commitseqs, batch_first=True)
            commitseqs = pack_padded_sequence(commitseqsPadded, commitseqsLength, batch_first=True, enforce_sorted=False)

        if(config.checkProcessMetricsExists()):
            processmetricss = torch.tensor(processmetricss).float()

        # yについて
        ys = torch.tensor(ys).float()
        return asts, astseqs, codemetricss, commitgraphs, commitseqs, processmetricss, ys