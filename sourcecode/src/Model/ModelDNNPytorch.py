from src.Node.Node import Node
from src.Config.config import config
from src.Logger.wrapperLogger import wrapperLogger
logger = wrapperLogger.setup_logger(__name__, config.getPathFileLog())
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import json
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy

class ModelDNNPytorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.period4HyperParameterSearch = config.period4HyperParameterSearch
        self.epochsEarlyStopping = config.epochs4EarlyStopping
        torch.backends.cudnn.enabled = True
        self.forward = None
        self.device = config.device or "cuda:0"
    def defineArchitecture(self, trial=None, hp=None):
        self.components = nn.ModuleDict()
        def createComponents(clsNode):
            numOfInput = 0
            for index in range(clsNode.numOfElements):
                clsChild = clsNode.classesChildren[max(index, len(clsNode.classesChildren)-1)]
                if(issubclass(clsChild, Node)):
                    numOfInput += createComponents(clsChild)
                elif(clsChild in [int, float, numpy.float64]):#todo Nodeの子クラスとfloat値を共存させる
                    numOfInput = clsNode.numOfElements
                    break
                else:
                    raise Exception("what type is the clschild?")
            component, numOfOutput = clsNode.calcComponentPytorch(numOfInput, trial, hp)
            self.components.add_module(clsNode.__name__, component)
            return numOfOutput
        createComponents(config.classRecord)
        def forward(class2input):
            #バッチに相当するtensorのlistが入った、辞書を引き数とする。
            #で、そのバッチに相当するtensorのlistoを書くコンポーネントに入力する。
            def forwardInput(clsNode):
                if(not clsNode.__name__ in class2input):
                    input = []
                    for index in range(clsNode.numOfElements):
                        clsChild = clsNode.classesChildren[max(index, len(clsNode.classesChildren)-1)]
                        outputFromLayerPrevious = forwardInput(clsChild)
                        if(clsNode.isSequential):
                            input.extend(outputFromLayerPrevious)
                        else:
                            input.append(outputFromLayerPrevious)
                    class2input[clsNode.__name__] = input[0]#todo
                output = self.components[clsNode.__name__](class2input[clsNode.__name__])
                if(clsNode.isSequential):
                    _, (parametersBiLSTM, __) = output
                    parametersBiLSTM = torch.cat(torch.split(parametersBiLSTM[(self.components[clsNode.__name__].num_layers-1)*2:], 1), dim=2)
                    output = parametersBiLSTM.squeeze()
                return output
            return forwardInput(config.classRecord)
        self.forward = forward
        model = self.to(self.device)
        return model.to(self.device)
    def defineOptimizer(self, model, trial=None, hp=None):
        if(hp!=None):
            nameOptimizer = hp["optimizer"]
            if nameOptimizer == 'adam':
                lrAdam      = hp["lrAdam"]
                beta1Adam  = hp["beta1Adam"]
                beta2Adam  = hp["beta2Adam"]
                epsilonAdam = hp["epsilonAdam"]
            optimizer = torch.optim.Adam(model.parameters(), lr=lrAdam, betas=(beta1Adam,beta2Adam), eps=epsilonAdam)
        elif(trial!=None):
            nameOptimizer = trial.suggest_categorical('optimizer', ['adam'])
            if nameOptimizer == 'adam':
                lrAdam = trial.suggest_loguniform('lrAdam', 1e-6, 1e-4)
                beta1Adam = trial.suggest_uniform('beta1Adam', 0.9, 0.9)
                beta2Adam = trial.suggest_uniform('beta2Adam', 0.999, 0.999)
                epsilonAdam = trial.suggest_loguniform('epsilonAdam', 1e-8, 1e-8)
            optimizer = torch.optim.Adam(model.parameters(), lr=lrAdam, betas=(beta1Adam,beta2Adam), eps=epsilonAdam)
        else:
            raise Exception("no arguments")
        return optimizer
    def searchParameter(self, *, dataLoader, model, lossFunction, optimizer, numEpochs, isEarlyStopping):
        lossesTrain = []
        lossesValid = []
        accsTrain = []
        accsValid = []
        lossValidBest = sys.float_info.max
        epochBestValid = 0
        for epoch in range(numEpochs):
            for phase in ["train","valid"]:
                if phase=="train":
                    model.train()
                elif phase=="valid":
                    model.eval()
                loss_sum=0
                corrects=0
                total=0
                with tqdm(total=len(dataLoader[phase]),unit="batch") as pbar:
                    pbar.set_description(f"Epoch[{epoch}/{numEpochs}]({phase})")
                    for ids, ys, class2input in  dataLoader[phase]:
                        ysPredicted = model(class2input)
                        ysPredicted = ysPredicted.squeeze()#もしysが1つしかなかったら、ベクトルじゃなくてスカラーに鳴ってしまう
                        ys = ys.squeeze()
                        loss=lossFunction(ysPredicted, ys)
                        if phase=="train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        sig = nn.Sigmoid()
                        ysPredicted =  torch.round(sig(ysPredicted))
                        corrects+=int((ysPredicted==ys).sum())
                        total+=ys.size(0)
                        accuracy = corrects/total
                        #loss関数で通してでてきたlossはCrossEntropyLossのreduction="mean"なので平均
                        #batch sizeをかけることで、batch全体での合計を今までのloss_sumに足し合わせる
                        loss_sum += float(loss) * ys.size(0)
                        running_loss = loss_sum/total
                        pbar.set_postfix({"loss":running_loss,"accuracy":accuracy })
                        pbar.update(1)
                if(phase == "train"):
                    lossesTrain.append(loss_sum/total)
                    accsTrain.append(corrects/total)
                if(phase == "valid"):
                    lossesValid.append(loss_sum/total)
                    accsValid.append(corrects/total)
                    if(loss_sum < lossValidBest):
                        print("loss gets lower")
                        lossValidBest = loss_sum
                        epochBestValid = epoch
            if(isEarlyStopping and self.epochsEarlyStopping<epoch-epochBestValid):
                break
        return epochBestValid, lossesTrain, lossesValid, accsTrain, accsValid
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
            def searchHyperParameter(datalot4Train):
                def objectiveFunction(trial):
                    logger.info("trial " + str(trial.number) + "started")
                    listLossesValid=[]
                    listEpochs=[]
                    model = self.defineArchitecture(trial=trial)
                    lossFunction = nn.BCEWithLogitsLoss()
                    optimizer = self.defineOptimizer(model, trial = trial)
                    for index4CrossValidation in range(config.splitSize4CrossValidation):
                        logger.info("cross validation " + str(index4CrossValidation+1) + "/" + str(config.splitSize4CrossValidation))
                        dataset4train, dataset4valid = datalot4Train.createDatasets(config.splitSize4CrossValidation, index4CrossValidation, self.__class__, trial=trial, hp=None)
                        dataloader={
                            "train": DataLoader(
                                dataset4train,
                                batch_size = 100,
                                collate_fn = dataset4train.collate
                            ),
                            "valid": DataLoader(
                                dataset4valid,
                                batch_size = 100,
                                collate_fn= dataset4valid.collate
                            )
                        }
                        epochBestValid, lossesTrain, lossesValid, accsTrain, accsValid = self.searchParameter(
                            dataLoader = dataloader,
                            model = model,
                            lossFunction = lossFunction,
                            optimizer = optimizer,
                            numEpochs = 10000,
                            isEarlyStopping=config.epochs4EarlyStopping
                        )
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
            dataset4train, dataset4valid = datalot4Train.createDatasets(0, 0, self.__class__, hp=hp)
            dataloader={
                "train": DataLoader(
                    dataset4train,
                    batch_size = 100,
                    collate_fn = dataset4train.collate
                ),
                "valid": DataLoader(
                    dataset4valid,
                    batch_size = 100,
                    collate_fn = dataset4valid.collate
                )
            }
            model = self.defineArchitecture(hp = hp)
            lossFunction = nn.BCEWithLogitsLoss()
            optimizer = self.defineOptimizer(model, hp = hp)
            _, lossesTrain, lossesValid, accsTrain, accsValid = self.searchParameter(
                dataLoader = dataloader,
                model = model,
                lossFunction = lossFunction,
                optimizer = optimizer,
                numEpochs = hp["numEpochs"],
                isEarlyStopping=False
            )
            self.plotGraphTraining(lossesTrain, lossesValid, accsTrain, accsValid, "graphTrainToTest")

            config.pathModel = os.path.join(config.pathDirOutput, "parameters")
            torch.save(model.state_dict(), config.pathModel)
        else:
            pass

    def test(self, datalot4Test):
        logger.info("test model")
        hp = self.loadHyperparameter()
        dataset4test, dataset4test = datalot4Test.createDatasets(0, 0, self.__class__, hp=hp)
        dataloader={
            "test": DataLoader(
                dataset4test,
                batch_size = 100,
                collate_fn = dataset4test.collate
            )
        }
        model = self.defineArchitecture(hp=hp)
        model.load_state_dict(torch.load(config.pathModel))
        model.eval()

        ids = []
        ysPredicted = []
        ys = []
        for idsTemp, ysTemp, class2input in dataloader['test']:
            ids += idsTemp
            ys += [l for l in ysTemp.to("cpu").squeeze().tolist()]
            with torch.no_grad():
                ysPredictedTemp = model(class2input)
                sig = nn.Sigmoid()
                ysPredictedTemp = sig(ysPredictedTemp)
                ysPredicted += [l for l in ysPredictedTemp.to("cpu").squeeze().tolist()]


        # output prediction result
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