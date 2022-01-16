from src.Node.Node import Node
from src.Node.Record import Record
from src.Node.VectorCommits import VectorCommits

class RecordMingSequence(Record):
    isSequencial = False
    numOfElements = 1
    classesChildren = [VectorCommits]

    def __init__(self, id, label, children):
        Record.__init__(self, id, label, [children])