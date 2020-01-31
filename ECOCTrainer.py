from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import MatrixGeneration
import math
import random

class Trainer():

    def __init__(self):
        pass

   # Return models so that predictions can be done later.
    def trainClassifiers(self, knownData, knownLabels, model):
        trainedModels = []

        for label in knownLabels:
            if model == 1:
                classifier = svm.SVC(gamma='auto')
            elif model == 2:
                classifier = DecisionTreeClassifier(random_state=0)
            elif model == 3:
                classifier = LinearDiscriminantAnalysis()
            elif model == 4:
                classifier = KNeighborsClassifier(n_neighbors=2)
            else:
                print("Specify Classifier")
            classifier = classifier.fit(knownData, label)
            trainedModels.append(classifier)

        return trainedModels

    def pickSingleClassifier(self, n):
        singleClassifierSets = []
        for i in range(n):
            singleClassifierSets.append([])
            singleClassifierSets[i].append(i)
        return singleClassifierSets

    # Sort the answer key
    # Go one to one between what it has
    # and what the classifier says.
    def testClassifiers(self, trainedModels, unseenData, answerKey):
        accuracies = []#np.array([])
        guesses = []
        oneToOneComparison = np.transpose(answerKey)
        for classifier in trainedModels:
            guesses.append(classifier.predict(unseenData))
        for guess, actual in zip(guesses, oneToOneComparison):
            accuracies.append(1-(MatrixGeneration.hammingDistance(guess, actual)/len(guess)))
        accuraciesNP = np.array(accuracies)
        return accuraciesNP

    def getPoorIndiciesNP(self, rawaccs, filterfactor):
        poorIndicies = []
        for j in range(filterfactor):
            #poorIndicies.append([])
            poorIndicies.append(rawaccs.argsort()[:j])
        return poorIndicies

    def getPoorIndicies(self, rawaccs, filterfactor):
        poorIndicies = []
        for j in range(filterfactor):
            poorIndicies.append((rawaccs.argsort()[:j]).tolist())
        return poorIndicies

    def makeFilteredSets(self, trainedModels, codebook, poorIndicies):
        modelSets = []
        codeSets = []
        trainedModelsNP = np.array(trainedModels)
        codebookT = MatrixGeneration.transpose(codebook)
        poorAccuracies = []
        print(poorIndicies)
        for poorAcc in poorIndicies:
            for a in poorAcc:
                poorAccuracies.append(a.tolist())
        print(poorAccuracies)
        for i in range(len(poorAccuracies)):
            codeSets.append([])
            for k in range(len(codebookT)):
                if k not in poorAccuracies[i]:
                    codeSets[i].append(codebookT[k])
        # for a in range(len(poorIndicies)):
        #    modelSets.append([])
        #    for b in range(len(trainedModels)):
        #        if b not in poorIndicies[a]:
        #            modelSets[a].append(trainedModels[b])
        for poorAccs in poorIndicies:
            modelSets.append(np.delete(trainedModelsNP, poorAccs))
        codeSetsNT = []
        for codebook in codeSets:
            codeSetsNT.append(MatrixGeneration.transpose(codebook))

        return modelSets, codeSets




    # Converts list containing multiple numpy arrays to list of lists containing codewords.
    def toCodeword(self, list):
        codeWordList = []
        tempList = []
        counter = 0

        while counter < len(list[0]):
            for prediction in list:
                tempList.append(prediction[counter])
            codeWordList.append(tempList)
            tempList = []
            counter += 1

        return codeWordList

    # Used trained classifiers to get predictions. Predictions will construct codewords.
    def getPredictions(self, validationData, trainedClassifiers):
        predictionList = []

        for classifier in trainedClassifiers:
            predictions = classifier.predict(validationData)
            predictionList.append(predictions)

        #print(len(predictionList))
        predictionList = self.toCodeword(predictionList)
        #print(len(predictionList))

        return predictionList

    # Takes codewords (usually predicted codewords) and "updates" them to whatever codeword they are
    # closest to (with respect to hamming distance) in a given codebook. Will also return a list that
    # shows what the minimum hamming distances were when deciding which codeword to updated the predicted
    # codeword with.
    def hammingDistanceUpdater(self, codebook, codewords):
        minHamWord = []
        # List containing actual CW based off of shortest HD
        UpdatedList = []
        minHamList = []
        for predictedCode in codewords:
            minHam = len(predictedCode)
            for actualCode in codebook:
                hammingDistance = 0
                for counter in range(0, len(predictedCode)):
                    #print(actualCode, predictedCode)
                    if actualCode[counter] != predictedCode[counter]:
                        hammingDistance += 1
                if hammingDistance < minHam:
                    minHam = hammingDistance
                    minHamWord = actualCode

            UpdatedList.append(minHamWord)
            minHamList.append(minHam)

        return UpdatedList, minHamList


    def distanceFunction(self, predicted, actual, accuracies, dF):
        maxAccuracy = max(accuracies)
        continuousDistance = 0
        if dF == 1:
            for counter in range(len(accuracies)):
                if predicted[counter] != actual[counter]:
                    continuousDistance += ((accuracies[counter])/maxAccuracy)
                else:
                    continuousDistance += (1- ((accuracies[counter])/maxAccuracy))
        elif dF == 2:
            for counter in range(len(accuracies)):
                if predicted[counter] != actual[counter]:
                    continuousDistance += ((accuracies[counter])/maxAccuracy)
                else:
                    continuousDistance -= ((accuracies[counter])/maxAccuracy)
        elif dF == 3:
            for counter in range(len(accuracies)):
                if predicted[counter] != actual[counter]:
                    continuousDistance -= math.log(((accuracies[counter])/maxAccuracy), 3)
                else:
                    continuousDistance += math.log(((accuracies[counter])/maxAccuracy), 3)
        return continuousDistance


    def hammingDistanceUpdaterDefinedFunction(self, codebook, codewords, classifieraccuracies, dF):
        minHamWord = []
        # List containing actual CW based off of shortest HD
        UpdatedList = []
        minHamList = []
        for predictedCode in codewords:
            minHam = len(predictedCode)
            for actualCode in codebook:
                hammingDistance = self.distanceFunction(predictedCode, actualCode, classifieraccuracies, dF)
                if hammingDistance < minHam:
                    minHam = hammingDistance
                    minHamWord = actualCode

            UpdatedList.append(minHamWord)
            minHamList.append(minHam)


        return UpdatedList, minHamList

    def hammingDistanceUpdaterFiltered(self, codebook, codewords, pooraccs):
        minHamWord = []
        # List containing actual CW based off of shortest HD
        UpdatedList = []
        minHamList = []
        predictionsSET = []
        for i in range(len(pooraccs)):
            for predictedCode in codewords:
                minHam = len(predictedCode)
                for actualCode in codebook:
                    hammingDistance = 0
                    for counter in range(0, len(predictedCode)):
                        if (actualCode[counter] != predictedCode[counter]) and counter not in pooraccs[i]:
                            hammingDistance += 1
                    if hammingDistance < minHam:
                        minHam = hammingDistance
                        minHamWord = actualCode

                UpdatedList.append(minHamWord)
                minHamList.append(minHam)

            predictionsSET.append(UpdatedList)
            UpdatedList = []

        return predictionsSET, minHamList


    # Gets accuracy of predicted codewords when compared to
    # actual (i.e. validation) codewords
    def compareIHD(self, predictions, actual):
        total = len(predictions)
        right = 0
        classes = []
        counter = 0
        for (x, y) in zip(predictions, actual):
            if x == y:
                right += 1
            else:
                classes.append(counter)

            counter+=1

        percentRight = right * 1.0 / total

        return percentRight, classes




    def compareFiltered(self, predictions, actual, setofIgnorances):
        total = len(predictions)
        right = 0
        accuracies = []
        for invalids in setofIgnorances:
            # for (x, y) in zip(predictions, actual):
            #    if x == y: #and index of x and y not in invalids
            #        right += 1
            print(invalids)
            for c in range(len(actual)):
                print(predictions[c], actual[c])
                if predictions[c] == actual[c] and c not in invalids:
                    right += 1
            accuracies.append((right * 1.0) / (total-len(invalids)))

        #    percentRight = right * 1.0 / total

        return accuracies  # percentRight