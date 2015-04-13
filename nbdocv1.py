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


def getPriorDists(labels):
    priorDists = {0: 0, 1: 0}

    for i in range(len(labels)):
        priorDists[labels[i]] += 1

    for key in priorDists.keys():
        priorDists[key] /= float(len(labels));

    return priorDists


def confusionMatrix(actual, prediction):
    mat = zeros((10, 10))

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
        print 'digit', i, 'accuracy:', confMat[i, i] * 1.0 / sum(confMat[i, :])

    print 'Overall accuracy:', overallAccuracy(actual, prediction)
    return None


def countNB(trainData, trainLabels, m=1, n=1):
    # if we use a dict for last dimension, would be able to save more memory
    priorDists = getPriorDists(trainLabels)
    numOfTrainDigits = len(trainLabels)
    numOfFeatures = len(trainData[0])
    nFeatVals = 2**(m*n)

    condProbTable = zeros((len(priorDists), numOfFeatures, nFeatVals))

    for i in range(numOfTrainDigits):
        for f in range(numOfFeatures):
            condProbTable[trainLabels[i], f, trainData[i, f]] += 1

    return condProbTable, priorDists, numOfFeatures, nFeatVals


def trainNB(m=1, n=1, k=1, overlap=False):
    trainData = getData(prepare("trainingimages"), m, n, overlap)
    trainLabels = getLabels("traininglabels")

    countTable, priorDists, numOfFeatures, nFeatVals = countNB(trainData, trainLabels, m, n)
    likelihoods = zeros((len(priorDists), numOfFeatures, nFeatVals))

    for i in range(len(priorDists)):
        denom = priorDists[i] * len(trainLabels) + k * nFeatVals
        for f in range(numOfFeatures):
            for v in range(nFeatVals):
                likelihoods[i, f, v] = (countTable[i, f, v] + k)/denom

    return likelihoods, priorDists, numOfFeatures, nFeatVals


def classify(m=1, n=1, k=1, overlap=False):
    from collections import OrderedDict
    from math import log
    predictions = []
    model, priorDists, numOfFeatures, nFeatVals = trainNB(m, n, k, overlap)

    toc()
    print "done training"

    testData = getData(prepare("testimages"), m, n, overlap)
    testLabels = getLabels("testlabels")
    for case in range(len(testData)):
        probs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

        for dClass in probs.keys():
            probs[dClass] += log(priorDists[dClass])

            for f in range(numOfFeatures):
                probs[dClass] += log(model[dClass, f, testData[case, f]])

        predictions.append(list(OrderedDict(sorted(probs.items(), key=lambda x: x[1], reverse=True)))[0])

    analyze(testLabels, predictions)

    return predictions


tic()
dictionary, trainingLabels, trainingMatrix = prepare("test_email.txt")

print getPriorDists(trainingLabels)

toc()