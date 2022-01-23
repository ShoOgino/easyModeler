from unicodedata import name
from src.Node.Node import Node
from src.Node.VectorCommit import VectorCommit
import torch.nn as nn
from keras.layers import LSTM, Input, Masking
from keras.layers.wrappers import Bidirectional
class VectorCommits(Node):
    isSequential = True
    numOfElements = 1
    classesChildren = [VectorCommit]

    def __init__(self, children):
        Node.__init__(self, children)

    @classmethod
    def calcComponentPytorch(cls, numOfInput, trial=None, hp = None):
        if(hp!=None):
            numOfLayers = hp['numOfLayers'+cls.__name__]
            hidden_size = hp['hiddenSize'+cls.__name__]
            rateDropout = hp['rateDropout'+cls.__name__]
        elif(trial!=None):
            numOfLayers = trial.suggest_int('numOfLayers'+cls.__name__, 1, 3)
            hidden_size = trial.suggest_int('hiddenSize'+cls.__name__, 16, 256)
            rateDropout = trial.suggest_uniform('rateDropout'+cls.__name__, 0.0, 0.0)
        else:
            raise Exception("no argument")
        component = nn.LSTM(
            input_size = numOfInput,
            hidden_size = hidden_size,
            num_layers = numOfLayers,
            batch_first = True,
            dropout = rateDropout,
            bidirectional = True
        )
        numOfOutput = hidden_size*2
        return component ,numOfOutput

    @classmethod
    def calcComponentTensorflow(cls, x, inputs, outputs,trial=None, hp = None):
        input = Input(shape = (None, VectorCommit.numOfElements))
        inputs.append(input)
        x = Masking(input_shape=(None, VectorCommit.numOfElements),mask_value=-1.0, name=cls.__name__)(input)
        if(hp!=None):
            numOfLayers = hp['numOfLayers'+cls.__name__]
            hidden_size = hp['hiddenSize'+cls.__name__]
            rateDropout = hp['rateDropout'+cls.__name__]
        elif(trial!=None):
            numOfLayers = trial.suggest_int('numOfLayers'+cls.__name__, 1, 3)
            hidden_size = trial.suggest_int('hiddenSize'+cls.__name__, 16, 256)
            rateDropout = trial.suggest_uniform('rateDropout'+cls.__name__, 0.0, 0.0)
        else:
            raise Exception("no argument")
        for i in range(numOfLayers):
            if( i != numOfLayers-1 ):
                x = Bidirectional(
                    LSTM(
                        hidden_size,
                        activation = "relu",
                        return_sequences=True
                    )
                )(x)
            else:
                x = Bidirectional(
                    LSTM(
                        hidden_size,
                        activation = "relu",
                        return_sequences=False
                    )
                )(x)
        return x