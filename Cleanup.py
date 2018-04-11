import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

dfTrain = pd.read_csv("Data/train.csv")
dfTest = pd.read_csv("Data/test.csv")

#This is a brutal method, it's far from being the best, but this give me a starting point to start working on

def Preprocess(DataF):
    NeighPrice = dfTrain[['SalePrice','Neighborhood']].groupby(['Neighborhood'], as_index=False).median().sort_values(by='SalePrice')
    NeighPrice['Id'] = list(range(len(NeighPrice)))
    DataF['NeighborhoodID'] = DataF['Neighborhood'].map(dict(zip(list(NeighPrice['Neighborhood']), list(NeighPrice['Id']))))
    for col in DataF.columns:
        if DataF[col].dtype == object:
            DataF[col] = DataF[col].fillna(value='Nan')
            DataF[col] = LabelEncoder().fit_transform(DataF[col])
        DataF[col] = DataF[col].fillna(value=(-1))
    return DataF

CleanedTrain = Preprocess(dfTrain)
Gotcha = IsolationForest()
Gotcha.fit(CleanedTrain)
Outliers = Gotcha.predict(CleanedTrain)
Outliers = Outliers == -1
CleanedTrain.drop(CleanedTrain[Outliers].index.tolist(), inplace=True)
CleanedTrain.to_csv("Data/ProcessedTrain.csv")
Preprocess(dfTest).to_csv("Data/ProcessedTest.csv")


#Better Cleanup Could Be:
"""#Creating a list of Categorical Data Columns for One Hot Encoding
CatData = ['MSSubClass','MSZoning','Street','Alley','LotShape','LotConfig','Neighborhood',
           'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
           'Exterior1st','Exterior2nd','MasVnrType','ExterQual','']

#Creating a list of Ordinal Data for good ordering
OrdData = ['LandContour','Utilities','LotShape']

#Creating a list of Numerical Data for rescaling
NumData = ['LotFrontage','LotArea','']

#Creating a list of "good enough" Data
GoodData = ['OverallQual','OverallCond','']

#Data Engineering Needed (intuition)
Changing = ['YearBuilt', 'YearRemodAdd',]

ourData = {
    'MSSubClass' :
}"""
        



