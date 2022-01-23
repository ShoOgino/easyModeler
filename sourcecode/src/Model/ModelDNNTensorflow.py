from src.Node.Node import Node
from src.Config.config import config
from src.Logger.wrapperLogger import wrapperLogger
logger = wrapperLogger.setup_logger(__name__, config.getPathFileLog())
import sys
import optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
class ModelDNNTensorflow():
    def __init__(self):
        super().__init__()
        self.period4HyperParameterSearch = config.period4HyperParameterSearch
        self.epochsEarlyStopping = config.epochs4EarlyStopping
        self.forward = None
        self.device = config.device or "cuda:0"
    def defineArchitecture(self, trial=None, hp=None):
        inputs = []
        outputs = []
        def connect(clsNode):
            input = []
            for index in range(clsNode.numOfElements):
                clsChild = clsNode.classesChildren[max(index, len(clsNode.classesChildren)-1)]
                if(hasattr(clsChild, "calcComponentTensorflow")):
                    outputFromLayerPrevious = connect(clsChild)
                    if(clsNode.isSequential):
                        input.extend(outputFromLayerPrevious)
                    else:
                        input.append(outputFromLayerPrevious)
            if(hasattr(clsNode, "calcComponentTensorflow")):
                output = clsNode.calcComponentTensorflow(input, inputs, outputs, trial=trial, hp=hp)
            return output
        connect(config.classRecord)
        model = Model(inputs = inputs, outputs = outputs)
        return model
    def defineOptimizer(self, trial=None, hp=None):
        if(hp!=None):
            nameOptimizer = hp["optimizer"]
            if nameOptimizer == 'adam':
                lrAdam      = hp["lrAdam"]
                beta1Adam  = hp["beta1Adam"]
                beta2Adam  = hp["beta2Adam"]
                epsilonAdam = None
            optimizer = optimizers.Adam(learning_rate=lrAdam, beta_1=beta1Adam, beta_2=beta2Adam, epsilon=epsilonAdam)
        elif(trial!=None):
            nameOptimizer = trial.suggest_categorical('optimizer', ['adam'])
            if nameOptimizer == 'adam':
                lrAdam = trial.suggest_loguniform('lrAdam', 1e-6, 1e-6)
                beta1Adam = trial.suggest_uniform('beta1Adam', 0.9, 0.9)
                beta2Adam = trial.suggest_uniform('beta2Adam', 0.999, 0.999)
                epsilonAdam = None
            optimizer = optimizers.Adam(learning_rate=lrAdam, beta_1=beta1Adam, beta_2=beta2Adam, epsilon=epsilonAdam)
        else:
            raise Exception("no arguments")
        return optimizer
    def loadHyperparameter(self):
        if(config.hyperparameter):
            return config.hyperparameter
        elif(config.pathDatabaseOptuna):
            # optunaデータベースから、最適なハイパーパラメータを読み出す
            study = optuna.create_study(study_name="optuna", storage='sqlite:///'+config.pathDirOutput + "/optuna.db", load_if_exists=True)
            return dict(study.best_params.items()^study.best_trial.user_attrs.items())
        else:
            logger.error("Hyperparameter can't be loaded. config.hyperparameter or config.pathDatabaseOptuna are not defined")
            raise Exception()
    def plotGraphTraining(self, lossesTrain, lossesValid, accTrain, accValid, title):
        # todo: lossが最小値のところに縦線・横線を引く
        epochs = range(len(lossesTrain))

        fig = plt.figure()
        plt.ylim(0, 2)
        plt.plot(epochs, lossesTrain, linestyle="-", color='b', label = 'lossTrain')
        plt.plot(epochs, accTrain, linestyle="-", color='r', label = 'accTrain')
        plt.plot(epochs, lossesValid, linestyle=":", color='b' , label= 'lossVal')
        plt.plot(epochs, accValid, linestyle=":", color='r' , label= 'accVal')
        plt.title(title)
        plt.legend()

        pathGraph = os.path.join(config.pathDirOutput, title + '.png')
        fig.savefig(pathGraph)
        plt.clf()
        plt.close()
    def plotGraphHyperParameterSearch(self, trials):
        # todo: lossが最小値のところに縦線・横線を引く
        numOfTrials = range(len(trials))

        fig = plt.figure()
        plt.title("HyperParameterSearch")
        plt.ylim(0, 1)
        plt.plot(numOfTrials, trials, linestyle="-", color='b', label = 'lossTrain')
        plt.legend()

        pathGraph = os.path.join(config.pathDirOutput, "hyperParameterSearch" + '.png')
        fig.savefig(pathGraph)
        plt.clf()
        plt.close()
    def build(self, doTraining, datalot4Train, doHyperparametertuning, doCrossValidation):
        if(doTraining):
            def searchHyperParameter(dataset4Train):
                def objectiveFunction(trial):
                    logger.info("trial " + str(trial.number) + "started")
                    listLossesValid=[]
                    listEpochs=[]
                    model = self.defineArchitecture(trial=trial)
                    lossFunction = "binary_crossentropy"
                    optimizer = self.defineOptimizer(trial = trial)
                    model.compile(optimizer=optimizer, loss=lossFunction, metrics=['acc'])
                    model.summary()
                    for index4CrossValidation in range(config.splitSize4CrossValidation):
                        logger.info("cross validation " + str(index4CrossValidation+1) + "/" + str(config.splitSize4CrossValidation))
                        dataset4train, dataset4valid= datalot4Train.createDatasets(config.splitSize4CrossValidation, index4CrossValidation, self.__class__, trial=trial, hp=None)
                        history = model.fit(
                            x = dataset4train,
                            batch_size=100,
                            epochs = 10000,
                            verbose = 1,
                            validation_data = dataset4valid,
                            callbacks=[
                                EarlyStopping(monitor='val_loss', patience = config.epochs4EarlyStopping, mode='auto')
                            ]
                        )
                        lossesTrain = history.history["loss"]
                        lossesValid = history.history["val_loss"]
                        accsTrain = history.history["acc"]
                        accsValid = history.history["val_acc"]
                        epochBestValid = lossesValid.index(min(lossesValid))
                        listLossesValid.append(lossesValid)
                        listEpochs.append(epochBestValid)
                        self.plotGraphTraining(lossesTrain, lossesValid, accsTrain, accsValid, "graphTrainToValid_" + str(trial.number) + "_" + str(index4CrossValidation))
                        if(not doCrossValidation): break
                    numEpochsBest=sum(listEpochs)//len(listEpochs)
                    trial.set_user_attr("numEpochs", numEpochsBest)
                    sumOfSquare=0
                    for i,numEpochs in enumerate(listEpochs):
                        temp = sum(listLossesValid[i][numEpochs:numEpochs+5])/5
                        sumOfSquare += temp * temp
                    logger.info(
                        "trial " + str(trial.number) + " end" + "\n" +
                        "value: " + str(sumOfSquare) + "\n"
                    )
                    return sumOfSquare
                logger.info("hyperparameter search started")
                config.pathDatabaseOptuna = config.pathDatabaseOptuna or config.pathDirOutput + "/optuna.db"
                study = optuna.create_study(study_name="optuna", storage='sqlite:///'+config.pathDatabaseOptuna, load_if_exists=True)
                study.optimize(objectiveFunction, timeout=config.period4HyperParameterSearch)
                #save the hyperparameter that seems to be the best.
                self.plotGraphHyperParameterSearch([v.value for v in study.trials])
            if(doHyperparametertuning):
                searchHyperParameter(datalot4Train)
            logger.info("build Model")
            hp = self.loadHyperparameter()
            model = self.defineArchitecture(hp=hp)
            lossFunction = "binary_crossentropy"
            optimizer = self.defineOptimizer(hp=hp)
            model.compile(optimizer=optimizer, loss=lossFunction, metrics=['acc'])
            model.summary()
            dataset4train, dataset4valid = datalot4Train.createDatasets(0, 0, self.__class__, trial=None, hp=hp)
            history = model.fit(
                x = dataset4train,
                batch_size=100,
                epochs = hp["numEpochs"],
                verbose = 1,
                validation_data = dataset4valid,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience = config.epochs4EarlyStopping, mode='auto')
                ]
            )
            lossesTrain = history.history["loss"]
            lossesValid = history.history["val_loss"]
            accsTrain = history.history["acc"]
            accsValid = history.history["val_acc"]
            self.plotGraphTraining(lossesTrain, lossesValid, accsTrain, accsValid, "graphTrainToTest")
            config.pathModel = os.path.join(config.pathDirOutput, "parameters")
            model.save(config.pathModel)
        else:
            pass
    def test(self, datalot4test):
        logger.info("test model")
        dataset4test, dataset4test= datalot4test.createDatasets(0, 0, self.__class__)
        ids, ys, xs = dataset4test.getIdsYX()

        model = load_model(config.pathModel)
        model.summary()
        ysPredicted = []
        for x in xs:
            xsTemp = numpy.array([x])
            ysPredicted.append(model.predict(xsTemp, batch_size=1).flatten().tolist()[0])

        resultTest = np.stack((ids, ys, ysPredicted), axis=1)
        with open(config.pathDirOutput+"/prediction.csv", 'w', newline="") as file:
            csv.writer(file).writerows(resultTest)

        # output recall, praazdlfecision, f-measure, AUC
        ysPredicted = np.round(ysPredicted, 0)
        report = classification_report(ys, ysPredicted, output_dict=True)
        report["AUC"] = roc_auc_score(ys, ysPredicted)
        with open(config.pathDirOutput+"/report.json", 'w') as file:
            json.dump(report, file, indent=4)

        # output confusion matrics
        cm = confusion_matrix(ys, ysPredicted)
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.savefig(config.pathDirOutput+"/ConfusionMatrix.png")