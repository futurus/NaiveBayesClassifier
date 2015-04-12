__author__ = 'vunguyen'
dimension = 28
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


def replace(char):
    if char is ' ':
        return 0
    else:
        return 1


def prepare(finput):
    out = []
    lines = [line for line in open(finput)]

    for nDigit in range(len(lines)/dimension):
        digit = []
        for i in range(dimension):
            cline = list(lines[nDigit*dimension + i])
            cline.pop()  # remove new line char
            cline = map(replace, cline)
            digit.append(cline)
        out.append(digit)

    return array(out)


def getLabels(finput):
    return [int(line[0]) for line in open(finput)]


def getPriorDists(labels):
    priorDists = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    for i in range(len(labels)):
        priorDists[labels[i]] += 1

    for key in priorDists.keys():
        priorDists[key] /= float(len(labels));

    return priorDists


def enumerate(size):
    vals = []

    for i in range(2**size):
        vals.append(format(i, '0' + size.__str__() + 'b'))

    return vals


def featValues(m=1, n=1):
    featVals = []

    enums = enumerate(m * n)

    for i in range(len(enums)):
        featVals.append(reshape(map(int, enums[i]), (m, n)))

    return featVals


def digitToFeatures(digit, m=1, n=1):  # by default we look at a pixel
    features = []

    if m > digit.shape[0] or n > digit.shape[1]:
        return None

    for row in range(digit.shape[0]-m+1):
        for col in range(digit.shape[1]-n+1):
            features.append(digit[row:(row+m), col:(col+n)])

    return features


def getData(digits, m=1, n=1):
    trainData = []

    for i in range(len(digits)):
        trainData.append(digitToFeatures(digits[i], m, n))

    return array(trainData)


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

    for i in range(len(confMat)):
        print 'digit ', i, ' accuracy: ', confMat[i, i] * 1.0 / sum(confMat[i, :])

    print 'Overall accuracy: ', overallAccuracy(actual, prediction)
    return None


def countNB(trainData, trainLabels, m=1, n=1):
    priorDists = getPriorDists(trainLabels)
    numOfTrainDigits = len(trainLabels)
    numOfFeatures = len(trainData[0])
    featVals = featValues(m, n)

    condProbTable = zeros((len(priorDists), numOfFeatures, len(featVals)))

    for i in range(numOfTrainDigits):
        for f in range(numOfFeatures):
            for v in range(len(featVals)):
                if allclose(trainData[i, f], featVals[v]):
                    condProbTable[trainLabels[i], f, v] += 1
                    break

    return condProbTable, priorDists, numOfFeatures, featVals


def trainNB(m=1, n=1, k=1):
    trainData = getData(prepare("trainingimages"), m, n)
    trainLabels = getLabels("traininglabels")

    countTable, priorDists, numOfFeatures, featVals = countNB(trainData, trainLabels, m, n)
    likelihoods = zeros((len(priorDists), numOfFeatures, len(featVals)))

    for i in range(len(priorDists)):
        denom = priorDists[i] * len(trainLabels) + k * len(featVals)
        for f in range(numOfFeatures):
            for v in range(len(featVals)):
                likelihoods[i, f, v] = (countTable[i, f, v] + k)/denom

    return likelihoods, priorDists, numOfFeatures, featVals


def classify(testData, testLabels, m=1, n=1, k=1):
    from collections import OrderedDict
    from math import log
    predictions = []
    model, priorDists, numOfFeatures, featVals = trainNB(m, n, k)

    toc()
    print "done training"

    for case in range(len(testData)):
        probs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

        for dClass in probs.keys():
            probs[dClass] += log(priorDists[dClass])

            for f in range(numOfFeatures):
                for v in range(len(featVals)):
                    if allclose(testData[case, f], featVals[v]):
                        probs[dClass] += log(model[dClass, f, v])
                        break

        predictions.append(list(OrderedDict(sorted(probs.items(), key=lambda x: x[1], reverse=True)))[0])

    print confusionMatrix(testLabels, predictions)

    return predictions


def autotune(testData, testLabels, m=1, n=1):
    from collections import OrderedDict
    from math import log
    trainData = getData(prepare("trainingimages"), m, n)
    trainLabels = getLabels("traininglabels")
    countMat, priorDists, numOfFeatures, featVals = countNB(trainData, trainLabels, m, n)

    toc()
    print "done counting"

    accuracy = []

    for k in [x + 1 for x in range(50)]:
        toc()
        print 'testing k: ', k

        predictions = []
        for case in range(len(testData)):
            probs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

            for dClass in probs.keys():
                probs[dClass] += log(priorDists[dClass])
                denom = priorDists[dClass] * len(trainLabels) + k * len(featVals)

                for f in range(numOfFeatures):
                    for v in range(len(featVals)):
                        if allclose(testData[case, f], featVals[v]):
                            probs[dClass] += log((countMat[dClass, f, v] + k)/denom)
                            break

            predictions.append(list(OrderedDict(sorted(probs.items(), key=lambda x: x[1], reverse=True)))[0])

        accuracy.append(overallAccuracy(testLabels, predictions))

    print accuracy
    bestk = accuracy.index(max(accuracy))+1

    predictions = []
    for case in range(len(testData)):
        probs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

        for dClass in probs.keys():
            probs[dClass] += log(priorDists[dClass])
            denom = priorDists[dClass]*len(trainLabels) + bestk * len(featVals)

            for f in range(numOfFeatures):
                for v in range(len(featVals)):
                    if allclose(testData[case, f], featVals[v]):
                        probs[dClass] += log((countMat[dClass, f, v] + bestk)/denom)
                        break

        predictions.append(list(OrderedDict(sorted(probs.items(), key=lambda x: x[1], reverse=True)))[0])

    print confusionMatrix(testLabels, predictions)
    analyze(testLabels, predictions)

    return None


tic()
testData = getData(prepare("testimages"))
testLabels = getLabels("testlabels")
autotune(testData, testLabels)
toc()