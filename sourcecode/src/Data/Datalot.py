from src.Config.config import config
from tensorflow.keras.utils import Sequence
import random
class Datalot():
    def __init__(self):
        self.records = []
    def normalize(self):
        node = config.classRecord
        node.calcMax()
        for record in self.records:
            record.normalize()
    def standardize(self, isTest):
        node = config.classRecord
        if(not isTest):
            node.calcMean()
            node.calcStandard()
        for i, record in enumerate(self.records):
            print(i)
            record.standardize()
    def identifyColumnsUseless(self):
        node = config.classRecord
        node.identifyColumnsUseless()
    def removeColumnsUseless(self):
        for record in self.records:
            record.removeColumnsUseless()
    #abstract
    def loadRecords():
        pass