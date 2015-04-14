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


def breakWFpair(pair):
    pair = pair.split(':')
    return tuple([pair[0], int(pair[1])])


def formatLine(line):
    components = line.split(' ')

    docClass = int(components[0])
    pairs = map(breakWFpair, components[1:])

    return [docClass, dict(pairs)]


def prepare(finput):
    import operator

    lines = map(formatLine, [line for line in open(finput)])
    docClassLabels = [line[0] for line in lines]
    docs = [line[1] for line in lines]  # each doc is a dictionary of word:frequency
    dictionary = list(set(reduce(operator.add, [line[1].keys() for line in lines])))

    featureMat = zeros((len(docClassLabels), len(dictionary)))

    for i in range(len(lines)):
        for word in docs[i].keys():
            featureMat[i, dictionary.index(word)] = docs[i][word]

    return dictionary, docClassLabels, featureMat  # return matrix and labels


def translateDoc(finput, dictionary):
    lines = map(formatLine, [line for line in open(finput)])
    docClassLabels = [line[0] for line in lines]
    docs = [line[1] for line in lines]  # each doc is a dictionary of word:frequency

    featureMat = zeros((len(docClassLabels), len(dictionary)))

    for i in range(len(lines)):
        for word in docs[i].keys():
            if word in dictionary:
                featureMat[i, dictionary.index(word)] = docs[i][word]

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


def classify(trainFile, testFile):
    dictionary, trainingLabels, trainingMatrix = prepare(trainFile)
    conditionalProbTable = trainNB(trainingMatrix, trainingLabels)
    testLabels, testMatrix = translateDoc(testFile, dictionary)

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
classify("8category.training.txt", "8category.testing.txt")
toc()