from src.Node.Node import Node

class VectorGiger(Node):
    isSequencial = False
    numOfElements = 24
    classesChildren = [float]*24

    def __init__(self, children):
        Node.__init__(self, children)
        VectorGiger.all.append(children)
