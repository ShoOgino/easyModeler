from src.Node.Node import Node
from src.Node.Record import Record
from src.Node.VectorGiger import VectorGiger

class RecordGiger(Record):
    isSequencial = False
    numOfElements = 1
    classesChildren = [VectorGiger]

    def __init__(self, id, label, children):
        Record.__init__(self, id, label, [children])