import torch
from src.Model.ModelDNNPytorch import ModelDNNPytorch
class DatasetPytorch(torch.utils.data.Dataset):
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
        return len(self.records)
    def __getitem__(self, index):
        return self.records[index]