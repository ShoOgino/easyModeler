from src.Node.Node import Node
from src.Model.Pass import Pass
class VectorCommit(Node):
    isSequential = False
    numOfElements = None
    classesChildren = [float]

    def __init__(self, children):
        Node.__init__(self, children)
        VectorCommit.all.append(children)

    @classmethod
    def calcComponentPytorch(cls, numOfInput, trial = None, hp = None):
        component = Pass()
        numOfOutput = numOfInput
        return component ,numOfOutput
