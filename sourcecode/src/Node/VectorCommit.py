from src.Node.Node import Node

class VectorCommit(Node):
    isSequencial = False
    numOfElements = None
    classesChildren = [float]

    def __init__(self, children):
        Node.__init__(self, children)
        VectorCommit.all.append(children)
