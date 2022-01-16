from src.Node.Node import Node
from src.Node.VectorCommit import VectorCommit

class VectorCommits(Node):
    isSequencial = True
    numOfElements = -1
    classesChildren = [VectorCommit]

    def __init__(self, children):
        Node.__init__(self, children)