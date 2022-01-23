from src.Model.LinearLayered import LinearLayered
from src.Node.Node import Node
from src.Node.Record import Record
from src.Node.VectorCommits import VectorCommits
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras.layers.core import Activation

class RecordMingSequence(Record):
    isSequential = False
    numOfElements = 1
    classesChildren = [VectorCommits]

    def __init__(self, id, label, children):
        Record.__init__(self, id, label, [children])

    @classmethod
    def calcComponentPytorch(cls, numOfInput, trial=None, hp=None):
        numOfLayers = 1
        numOfOutput = 1
        component = LinearLayered(
            in_features  = numOfInput,
            out_features = numOfOutput,
            numOfLayers  = numOfLayers
        )
        return component ,numOfOutput

    @classmethod
    def calcComponentTensorflow(cls, x, inputs, outputs, trial=None, hp=None):
        if(type(x)==list):
            if(len(x)==1):
                x = x[0]
            else:
                x = concatenate(x)
        x = Dense(1, name=cls.__name__)(x)
        output = Activation("sigmoid")(x)
        outputs.append(output)
        return output