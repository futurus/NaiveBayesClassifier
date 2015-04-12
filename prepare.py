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
        denom = priorDists[i]*len(trainLabels) + k * len(featVals)
        for f in range(numOfFeatures):
            for v in range(len(featVals)):
                likelihoods[i, f, v] = (countTable[i, f, v] + k)/denom

    return likelihoods, priorDists, numOfFeatures, featVals


def classify(testData, testLabels, m=1, n=1, k=1):
    from collections import OrderedDict
    from math import log
    classification = []
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

        ordered = OrderedDict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
        print ordered
        toc()
        classification.append(ordered[0][0])
        print '#case: ', case, '(actual, predicted): (', testLabels[case], ', ', classification[case], ')'


    return None


tic()
testData = getData(prepare("testimages"))
testLabels = getLabels("testlabels")
# print testData[0]
classify(testData, testLabels)
toc()


# trainData = getTrainData(prepare("testimages"), m=1, n=1)
# trainLabels = getLabels("testlabels")

# condTable = trainNB(trainData, trainLabels, k=0, m=1, n=1)

# likelihoods = trainNB()
# for i in range(len(condTable)):
# print likelihoods[0, 0]
# print likelihoods[0, 400]

# print len(trainData)
# print len(trainData[0])
# print trainData[0]
# print trainData[0][0]

# print createFeatures(out[0], 1, 1)[0] in featValues(m=1, n=1)

# print getLabels("testlabels")
# print getPriorDists(getLabels("testlabels"))