import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

dfTrain = pd.read_csv("Data/ProcessedTrain.csv")
dfTest = pd.read_csv("Data/ProcessedTest.csv")

xToTest = dfTest.drop(['Id'], axis=1)
xTestId = dfTest['Id']
xToTrain = dfTrain.drop(['Id', 'SalePrice'], axis=1)
yTraining = dfTrain['SalePrice']

xTrain, xValid, yTrain, yValid = train_test_split(xToTrain, yTraining, test_size=0.20)

Algos = [GradientBoostingRegressor(), AdaBoostRegressor(), BaggingRegressor(), ExtraTreesRegressor(),
         RandomForestRegressor()]

Predictions = pd.DataFrame()

for algo in Algos:
    algo.fit(xTrain, yTrain)
    trainPredict = algo.score(xTrain, yTrain)
    validPredict = algo.score(xValid, yValid)
    print(algo.__class__.__name__, 'trainPredict: ', trainPredict, ' validPredict: ', validPredict)

for algo in Algos:
    algo.fit(xToTrain, yTraining)
    testPredict = algo.predict(xToTest)
    Predictions = pd.concat([Predictions, pd.DataFrame(testPredict)], axis=1, copy=False)

finalPredictions = Predictions.mean(axis=1)

GBR = GradientBoostingRegressor()
ABR = AdaBoostRegressor()
GBR.fit(xToTrain, yTraining)
ABR.fit(xToTrain, yTraining)
finalPredictionsABR = ABR.predict(xToTest)
finalPredictionsGBR = GBR.predict(xToTest)

predsId = dfTest['Id'
submission = pd.DataFrame({'Id':predsId,'SalePrice':finalPredictions})
submission.to_csv("submissionHousesPricesKaggle2.csv",index=False)
submission = pd.DataFrame({'Id':predsId,'SalePrice':finalPredictionsGBR})
submission.to_csv("submissionHousesPricesKaggle3.csv",index=False)
submission = pd.DataFrame({'Id':predsId,'SalePrice':finalPredictionsABR})
submission.to_csv("submissionHousesPricesKaggle4.csv",index=False)


