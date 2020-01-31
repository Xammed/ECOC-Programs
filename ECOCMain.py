from DataManager import DataManager
from ECOCTrainer import Trainer
from sklearn.model_selection import train_test_split
from Graphing import Visuals
import MatrixGeneration
import numpy as np
import array as arr
import matplotlib.pyplot as mpl
import matplotlib.pyplot as mpl2
import matplotlib.pyplot as mpl3


def getECOCBaselineAccuracies(dataset, listOfCBs, labelCol, beginData, endData):
    dm = DataManager()
    trainer = Trainer()
    codebookNum = 1
    grapher = Visuals()
    models = ["SVM", "DT", "LDA", "KNN"] # Used for printing
    endResultAccuracies = []  # Holds lists of all accuracies for each codebook

    for codebook in listOfCBs:
        # Lists to hold each iteration's accuracy for each model
        mDh = MatrixGeneration.minHammingDistance(codebook)
        svmAccuracies = []
        dtAccuracies = []
        ldaAccuracies = []
        knnAccuracies = []
        modelAccuracies = [svmAccuracies, dtAccuracies, ldaAccuracies, knnAccuracies]


        functionOne = [[], [], [], []]
        functionTwo = [[], [], [], []]
        functionThree = [[], [], [], []]


        print("Codebook Number:", codebookNum)
        for iteration in range(1):
            print("\tIteration:", (iteration + 1))

            for i in range(2,5): # range(1,5) because models are chosen based off of number ( 1 = SVM, 2 = DT, etc.)

                # Get and preprocess the data
                data, labels = dm.getData(dataset, labelCol, beginData, endData)



                indicesToRemove, dataToRemove, labelsToRemove = dm.getSmallClasses(data, labels) #Letters.
                data, labels = dm.removeSmallClasses(data, labels, indicesToRemove)

                data = dm.preprocessData(data)

                # You need the dictionary to get accuracies later!
                # Give each label a codeword then split the data
                updatedLabels, labelDictionary = dm.assignCodeword(labels, codebook)




                x_train, x_test, y_train, y_test = train_test_split(data, updatedLabels, test_size=.20, random_state=12)



                # Create lists of what each class's new label should be based off of the
                # columns of the codewords
                binaryClassifiers = dm.binarizeLabels(labelDictionary)



                # Since splitting happens after the assigning/updating of codwords, we need to get the
                # original labels back so that makeTrainingLabels works properly (This can be improved upon)
                originalTrainLabels = dm.originalY(y_train, labelDictionary)



                # Train the models
                trainingLabels = dm.makeTrainingLabels(binaryClassifiers, originalTrainLabels)
                trainedModels = trainer.trainClassifiers(x_train, trainingLabels, i)

                #Test the classifiers

                #A set of accuracies for each classifer
                classifierAccuracies = trainer.testClassifiers(trainedModels, x_test, y_test)

                #The maximum number of classifiers to remove, leaving log2 of them

                filterfactor = int(len(MatrixGeneration.transpose(codebook)) - np.log2(len(MatrixGeneration.transpose(codebook))))



                #A set of indicies to get which classifiers you can remove
                #classifierstoFilter = trainer.getPoorIndicies(classifierAccuracies, filterfactor)
                classifierstoFilter = trainer.pickSingleClassifier(len(MatrixGeneration.transpose(codebook)))



                indivDomain =  range(len(classifierAccuracies))


                mpl.plot(indivDomain, classifierAccuracies)
                mpl.title("Individual Classifiers" + " " + str(models[i - 1]))
                mpl.xlabel("Classifier Index")
                mpl.ylabel("Accuracy")

                mpl.show()
                mpl.clf()


                #allAccuracies = []
                #predictions = trainer.getPredictions(x_test, trainedModels)
                #updatedPredictionsSet, minHams = trainer.hammingDistanceUpdaterFiltered(codebook, predictions, classifierstoFilter)
                #for updatedPredictions in updatedPredictionsSet:
                #    allAccuracies.append(trainer.compare(updatedPredictions, y_test))

                #xVals = range(len(allAccuracies))

                #mpl2.plot(xVals, allAccuracies)
                #mpl2.title("Classifier " + str(models[i-1]) + " Codebook " + str(codebookNum) + " " + "(mHd: " + str(mDh) + ") " + str(models[i-1]))
                #mpl2.xlabel("Number of Classifiers Removed")
                #mpl2.xlabel("Index of Classifier Removed")
                #mpl2.ylabel("ECOC Accuracy")
                #mpl2.show()
                #mpl.savefig(str(codebookNum) + "_" + str(iteration) + str(models[i-1]) + ".png")
                #mpl2.clf()


                predictions = trainer.getPredictions(x_test, trainedModels)
                updatedPredictionsA, minHamsA = trainer.hammingDistanceUpdater(codebook, predictions)
                #updatedPredictionsB, minHamsB = trainer.hammingDistanceUpdaterDefinedFunction(codebook, predictions,
                #                                                                              classifierAccuracies, 1)
                #updatedPredictionsC, minHamsC = trainer.hammingDistanceUpdaterDefinedFunction(codebook, predictions,
                #                                                                              classifierAccuracies, 2)
                #updatedPredictionsD, minHamsD = trainer.hammingDistanceUpdaterDefinedFunction(codebook, predictions,
                #                                                                             classifierAccuracies, 3)

                accuracyA = trainer.compare(updatedPredictionsA, y_test)
                #accuracyB = trainer.compare(updatedPredictionsB, y_test)
                #accuracyC = trainer.compare(updatedPredictionsC, y_test)
                #accuracyD = trainer.compare(updatedPredictionsD, y_test)

                #Confusion Matrices:
                #nestedPredA = []
                #nestedPredB = []
                #nestedPredC = []
                #nestedPredD = []
                #nestedActual = []
                #nestedPredA.append(updatedPredictionsA)
                #nestedPredB.append(updatedPredictionsB)
                #nestedPredC.append(updatedPredictionsC)
                #nestedPredD.append(updatedPredictionsD)
                #nestedActual.append(y_test)
                #grapher.generateConfusionMatrix(nestedPredA, nestedActual, codebook,
                #"C:\\Users\M Sarosh Khan\PycharmProjects\\CodebookTesting\\LeafBase", i,
                #1, 0.2)
                #grapher.generateConfusionMatrix(nestedPredB, nestedActual, codebook,
                #                               "C:\\Users\M Sarosh Khan\PycharmProjects\\CodebookTesting\\LeafF1",
                #                                i,
                #                                1, 0.2)
                #grapher.generateConfusionMatrix(nestedPredC, nestedActual, codebook,
                #                                "C:\\Users\M Sarosh Khan\PycharmProjects\\CodebookTesting\\LeafF2",
                #                                i,
                #                                1, 0.2)
                #grapher.generateConfusionMatrix(nestedPredD, nestedActual, codebook,
                #                                "C:\\Users\M Sarosh Khan\PycharmProjects\\CodebookTesting\\LeafF3",
                #                                i,
                #                                1, 0.2)

                modelAccuracies[i - 1].append(accuracyA)
                #functionOne[i-1].append(accuracyB)
                #functionTwo[i-1].append(accuracyC)
                #functionThree[i-1].append(accuracyD)






                #print("Baseline: " + str(modelAccuracies))
                #print("Not counting same: " + str(functionOne))
                #print("Counting Same: " + str(functionTwo))
                #print("Cubing: " + str(functionThree))


        averagedListA = [] # A temp list to hold all the averaged accuracies for a codebook
        averagedListB = []
        averagedListC = []
        averagedListD = []



        for o in range(len(modelAccuracies)):
            averagedListA.append(np.average(modelAccuracies[o]))
            #averagedListB.append(np.average(functionOne[o]))
            #averagedListC.append(np.average(functionTwo[o]))
            #averagedListD.append(np.average(functionThree[o]))



        endResultAccuracies.append(averagedListA)
        #endResultAccuracies.append(averagedListB)
        #endResultAccuracies.append(averagedListC)
        #endResultAccuracies.append(averagedListD)

        print(endResultAccuracies)

        codebookNum += 1
    l=0
    print("##-- Baselines --##")
    for average in endResultAccuracies[0]:
        print("\t" + models[l] + ":", np.round(average, 2))
        l += 1

    l=0
    print("##-- Function One --##")
    for average in endResultAccuracies[1]:
        print("\t" + models[l] + ":", np.round(average, 2))
        l += 1

    l=0
    print("##-- Function Two --##")
    for average in endResultAccuracies[2]:
        print("\t" + models[l] + ":", np.round(average, 2))
        l += 1

    l=0
    print("##-- Function Three --##")
    for average in endResultAccuracies[3]:
        print("\t" + models[l] + ":", np.round(average, 2))
        l += 1


    return endResultAccuracies






cb = MatrixGeneration.makeCodes(31, 31, 5, 5, 1, 31)
cb2 = MatrixGeneration.makeCodes(31, 31, 10, 10, 1, 31) #Write to accept range of H
cb3 = MatrixGeneration.GenerateMatrix(31,31)


dataset= "C:\\Users\\M Sarosh Khan\\PycharmProjects\\CodebookTesting\\datasets\\Leaf.csv"
listOfCBs = [cb3]#cb, cb2, cb3]

print(getECOCBaselineAccuracies(dataset, listOfCBs, -1, 0, 15))