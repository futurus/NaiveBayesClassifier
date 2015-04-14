__author__ = 'vunguyen'

from numpy import *
import time


def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"


def prepare(fdata, flabel):
    docClassLabels = [int(label)-1 for label in open(flabel)]
    lines = array([map(int, line.split(' ')) for line in open(fdata)])

    dictionary = list(set(lines[:, 1]))

    featureMat = zeros((len(docClassLabels), max(dictionary)))

    for i in range(len(lines)):
        featureMat[lines[i][0]-1, lines[i][1]-1] = lines[i][2]

    return dictionary, docClassLabels, featureMat  # return matrix and labels


def translateDoc(finput, finputLabel, dictionary):
    docClassLabels = [int(label)-1 for label in open(finputLabel)]
    lines = array([map(int, line.split(' ')) for line in open(finput)])

    featureMat = zeros((len(docClassLabels), max(dictionary)))

    for i in range(len(lines)):
        if lines[i][1]-1 < max(dictionary):
            featureMat[lines[i][0]-1, lines[i][1]-1] = lines[i][2]

    return docClassLabels, featureMat


def getPriorDists(labels):
    priorDists = {}

    for i in range(len(set(labels))):
        priorDists[i] = 0

    for i in range(len(labels)):
        priorDists[labels[i]] += 1

    for key in priorDists.keys():
        priorDists[key] /= float(len(labels));

    return priorDists


def confusionMatrix(actual, prediction):
    mat = zeros((len(set(actual)), len(set(actual))))

    for i in range(len(actual)):
        mat[actual[i], prediction[i]] += 1

    return mat


def overallAccuracy(actual, prediction):
    correct = 0

    for i in range(len(actual)):
        if actual[i] == prediction[i]:
            correct += 1

    return correct*1.0/len(actual)


def analyze(actual, prediction):
    confMat = confusionMatrix(actual, prediction)

    print confMat

    for i in range(len(confMat)):
        print 'class', i, 'accuracy:', confMat[i, i] * 1.0 / sum(confMat[i, :])

    print 'Overall accuracy:', overallAccuracy(actual, prediction)
    return None


def trainNB(trainMatrix, trainLabels):
    numOfTrainDocs = len(trainLabels)
    numOfWords = len(trainMatrix[0])
    numOfClasses = len(set(trainLabels))

    countTable = ones((numOfClasses, numOfWords))
    denoms = ones((numOfClasses)) * numOfWords * 1.0

    for i in range(numOfTrainDocs):
        countTable[trainLabels[i], :] += trainMatrix[i]
        denoms[trainLabels[i]] += sum(trainMatrix[i])

    for c in range(numOfClasses):
        countTable[c] = countTable[c]/denoms[c]

    return countTable


def classify():
    dictionary, trainingLabels, trainingMatrix = prepare("p2ec/train.data.txt", "p2ec/train.label.txt")
    conditionalProbTable = trainNB(trainingMatrix, trainingLabels)
    testLabels, testMatrix = translateDoc("p2ec/test.data.txt", "p2ec/test.label.txt", dictionary)

    numOfClasses = len(set(trainingLabels))
    priorProbs = getPriorDists(trainingLabels)

    predictions = []

    for i in range(len(testLabels)):
        posteriorProbs = []

        for c in range(numOfClasses):
            posteriorProbs.append(sum(testMatrix[i] * log(conditionalProbTable[c])) + log(priorProbs[c]))

        predictions.append(posteriorProbs.index(max(posteriorProbs)))


    analyze(testLabels, predictions)

    return None


tic()

classify()

toc()