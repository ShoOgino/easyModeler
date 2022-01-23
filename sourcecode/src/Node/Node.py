import numpy

class Node():
    isRoot = None
    isSequencial = None
    isPictual = None
    numOfElements = None
    classesChildren = []
    columnsUseless = []
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
    def removeColumnsUseless(self):
        flag = False
        for clsChild in self.classesChildren:
            if(not clsChild in [int, float, numpy.float64]):
                flag=True
        if(flag):
            for child in self.children:
                child.removeColumnsUseless()
        else:
            for column in sorted(self.columnsUseless, reverse=True):
                self.children.pop(column)
            self.__class__.numOfElements = len(self.children)
    @classmethod
    def identifyColumnsUseless(cls):
        flag = False
        for clsChild in cls.classesChildren:
            if(not clsChild in [int, float, numpy.float64]):
                flag=True
        if(flag):
            for clsChild in cls.classesChildren:
                clsChild.identifyColumnsUseless()
        else:
            cls.calcMax()
            cls.calcSum()
            for i in range(cls.numOfElements):
                if(cls.max[i]==0 and cls.sum[i]==0):
                    cls.columnsUseless.append(i)
            for column in sorted(cls.columnsUseless, reverse=True):
                for indexAll in range(len(cls.all)):
                    cls.all[indexAll].pop(column)

    @classmethod
    def calcMax(cls):
        flag = False
        for classChild in cls.classesChildren:
            if(not classChild in [int, float, numpy.float64]):
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
            if(not classChild in [int, float, numpy.float64]):
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
            if(not classChild in [int, float, numpy.float64]):
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
            if(not classChild in [int, float, numpy.float64]):
                flag=True
        if(flag):
            for classChild in cls.classesChildren:
                classChild.calcStandard()
        else:
            cls.standard = numpy.std(cls.all, axis=0)
    def standardize(self):
        flag = False
        for classChild in self.classesChildren:
            if(not classChild in [int, float, numpy.float64]):
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
            if(not classChild in [int, float, numpy.float64]):
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