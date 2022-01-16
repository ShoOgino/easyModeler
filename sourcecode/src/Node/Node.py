from src.Model.LinearLayered import LinearLayered
import numpy
import torch.nn as nn

class Node():
    isSequencial = None
    isPictual = None
    numOfElements = None
    classesChildren = []
    columnUseless = []
    all = []
    max = []
    sum = []
    mean = []
    standard = []

    def __init__(self, children):
        self.children = []
        if(len(self.classesChildren) == 1):
            for child in children:
                self.children.append(self.classesChildren[0](child))
        else:
            for index,child in enumerate(children):
                self.children.append(self.classesChildren[index](child))
    #valuesがint or floatなら計算。
    #違うなら委譲
    @classmethod
    def identifyColumnUseless(cls):
        cls.calcMax()
        cls.calcSum()
        for i in range(cls.numOfElements):
            if(cls.max[i]==0 and cls.sum[i]==0):
                cls.columnUseless.append(i)
    @classmethod
    def calcMax(cls):
        flag = False
        for classChild in cls.classesChildren:
            if(not classChild in [int, float]):
                flag=True
        if(flag):
            for classChild in cls.classesChildren:
                classChild.calcMax()
        else:
            cls.max =  numpy.max(cls.all, axis=0)
    @classmethod
    def calcSum(cls):
        flag = False
        for classChild in cls.classesChildren:
            if(not classChild in [int, float]):
                flag=True
        if(flag):
            for classChild in cls.classesChildren:
                classChild.calcMax()
        else:
            cls.sum =  numpy.sum(cls.all, axis=0)
    @classmethod
    def calcMean(cls):
        flag = False
        for classChild in cls.classesChildren:
            if(not classChild in [int, float]):
                flag=True
        if(flag):
            for classChild in cls.classesChildren:
                classChild.calcMean()
        else:
            cls.mean =  numpy.mean(cls.all, axis=0)
    #valuesがint or floatなら計算。
    #違うなら委譲
    @classmethod
    def calcStandard(cls):
        flag = False
        for classChild in cls.classesChildren:
            if(not classChild in [int, float]):
                flag=True
        if(flag):
            for classChild in cls.classesChildren:
                classChild.calcStandard()
        else:
            cls.standard = numpy.std(cls.all, axis=0)
    def standardize(self):
        flag = False
        for classChild in self.classesChildren:
            if(not classChild in [int, float]):
                flag=True
        if(flag):
            for child in self.children:
                child.standardize()
        else:
            for index, child in enumerate(self.children):
                self.children[index] = (child-self.mean[index])/self.standard[index]
    def normelize(self):
        flag = False
        for classChild in self.classesChildren:
            if(not classChild in [int, float]):
                flag=True
        if(flag):
            for child in self.children:
                child.normalize()
        else:
            for index, child in enumerate(self.children):
                self.children[index] = child/self.max[index]
    def toVector(self):
        if(self.numOfElements==1):
            self.children[0].toVector()
        else:
            vector = []
            for child in self.children:
                if(type(child) in [int, float, numpy.float64]):
                    vector.append(child)
                else:
                    vector.append(child.toVector())
            return vector
    def calcComponent(self, numOfInput):
        numOfOutput = 0
        if(self.isSequencial):
            #LSTM
            numOfLayers = 1
            hidden_size = numOfInput//10,
            component = nn.LSTM(
                input_size = numOfInput,
                hidden_size = hidden_size,
                num_layers = numOfLayers,
                batch_first = True,
                dropout = 0,
                bidirectional = True
            )
            numOfOutput += hidden_size*2
        elif(self.isPictual):
            pass
        else:
            #todo MyLinearクラスを作る。LSTMみたいに一つのコンポーネントで複数レイヤーを繋げられるようにする。
            numOfLayers = 1
            numOfOutput = numOfInput//10
            component = LinearLayered(
                in_features  = numOfInput,
                out_features = numOfOutput,
                numOfLayers  = numOfLayers
            )
        return component ,numOfOutput