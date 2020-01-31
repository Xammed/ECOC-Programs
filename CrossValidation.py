import numpy as np
import sklearn as sk
import random
import pandas as pd
from numpy import genfromtxt
from sklearn.impute import SimpleImputer

#This class also needs to manage the data.


class ECOCValidator():

    def __init__(self, codebook):
        self.codebook = codebook

    def getData(self, dataset, labelsColumn, dataBeginIndex, dataEndIndex):
        importedDataset = pd.read_csv(dataset, header=None)
        numColumns = len(importedDataset.columns)
        dataValues = genfromtxt(dataset, delimiter = ',', usecols = range(dataBeginIndex, dataEndIndex)).tolist()

        #1 == labels are in the first column. -1 == labels are in the last column
        if(labelsColumn == 1):
            labels = importedDataset.ix[:, 0].tolist()
        elif(labelsColumn == -1):
            labels = importedDataset.ix[:, (numColumns - 1)].tolist()

        return dataValues, labels

        # Preprocesses the data.
    def preprocessData(self, data):
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(data)
            imputedData = imputer.transform(data)  # nan values will take on mean
            scaledData = preprocessing.scale(imputedData).tolist()

            return scaledData

