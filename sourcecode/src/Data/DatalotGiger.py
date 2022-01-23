from random import sample
from src.Dataset.Datalot import Datalot

import json
import os
import glob

from src.Config.config import config


class DatalotGiger(Datalot):
    def __init__(self, pathsRecord):
        super().__init__()
        self.pathsRecord = []
        self.setPathsRecord(pathsRecord)
        self.loadRecords()
        self.standardize()
        self.toVector()
        print("")

    def setPathsRecord(self, pathsDir):
        for path in pathsDir:
            if(os.path.isdir(path)):
                pathsSearched = glob.glob(path+"/**/*.json", recursive=True)
                self.pathsRecord.extend(pathsSearched)

    def loadRecords(self):
        def loadARecord(pathRecord):
            with open(pathRecord, encoding="utf-8") as fSample4Train:
                recordJson = json.load(fSample4Train)
                record = config.classRecord(
                    recordJson["path"],
                    recordJson["commitsOnModuleAll"]["isBuggy"],
                    [
                        recordJson["sourcecode"]["fanin"],
                        recordJson["sourcecode"]["fanout"],
                        recordJson["sourcecode"]["numOfParameters"],
                        recordJson["sourcecode"]["numOfVariablesLocal"],
                        recordJson["sourcecode"]["ratioOfLinesComment"],
                        recordJson["sourcecode"]["numOfPaths"],
                        recordJson["sourcecode"]["complexity"],
                        recordJson["sourcecode"]["numOfStatements"],
                        recordJson["sourcecode"]["maxOfNesting"],
                        recordJson["commitsOnModuleInInterval"]["numOfCommits"],
                        recordJson["commitsOnModuleInInterval"]["numOfCommittersUnique"],
                        recordJson["commitsOnModuleInInterval"]["sumOfAdditionsStatement"],
                        recordJson["commitsOnModuleInInterval"]["maxOfAdditionsStatement"],
                        recordJson["commitsOnModuleInInterval"]["avgOfAdditionsStatement"],
                        recordJson["commitsOnModuleInInterval"]["sumOfDeletionsStatement"],
                        recordJson["commitsOnModuleInInterval"]["maxOfDeletionsStatement"],
                        recordJson["commitsOnModuleInInterval"]["avgOfDeletionsStatement"],
                        recordJson["commitsOnModuleInInterval"]["sumOfChurnsStatement"],
                        recordJson["commitsOnModuleInInterval"]["maxOfChurnsStatement"],
                        recordJson["commitsOnModuleInInterval"]["avgOfChurnsStatement"],
                        recordJson["commitsOnModuleInInterval"]["sumOfChangesDeclarationItself"],
                        recordJson["commitsOnModuleInInterval"]["sumOfChangesCondition"],
                        recordJson["commitsOnModuleInInterval"]["sumOfAdditionStatementElse"],
                        recordJson["commitsOnModuleInInterval"]["sumOfDeletionStatementElse"]
                    ]
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
            rec["x"] = record.children[0].children
            recordsVector.append(rec)
        self.records = recordsVector