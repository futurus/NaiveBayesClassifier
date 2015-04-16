__author__ = 'vunguyen'
dimension = 28
from numpy import *
import time
from math import log


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


def bin2dec(a):
    return int(''.join(map(str, map(int, a.flatten()))), 2)


def digitToFeatures(digit, m=1, n=1, overlap=False):  # by default we look at a pixel
    features = []

    if m > digit.shape[0] or n > digit.shape[1]:
        print "m or n too large (max 28)"
        return None

    if overlap is False and ((28 % m != 0) or (28 % n != 0)):
        print "m or n is not a divisor of 28"
        return None

    if overlap is False:
        for row in range(0, digit.shape[0], m):
            for col in range(0, digit.shape[1], n):
                features.append(bin2dec(digit[row:(row+m), col:(col+n)]))
    else:
        for row in range(digit.shape[0]-m+1):
            for col in range(digit.shape[1]-n+1):
                features.append(bin2dec(digit[row:(row+m), col:(col+n)]))

    return features


def getData(digits, m=1, n=1, overlap=False):
    trainData = []

    for i in range(len(digits)):
        trainData.append(digitToFeatures(digits[i], m, n, overlap))

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

    print confMat

    for i in range(len(confMat)):
        print 'digit', i, 'accuracy:', confMat[i, i] * 1.0 / sum(confMat[i, :])

    print 'Overall accuracy:', overallAccuracy(actual, prediction)
    return None


def project(iArray, multiplier):
    from math import floor
    out = zeros((iArray.shape[0] * multiplier, iArray.shape[1] * multiplier))

    for i in range(len(out)):
        for j in range(len(out[i])):
            out[i, j] = iArray[floor(i / multiplier), floor(j / multiplier)]

    return out


def createGreyscale(out, matrix):
    import matplotlib.pyplot as plt
    # x = ((random.rand(28*28))*255).reshape(28, 28)
    # plt.gray()
    plt.imsave(out, project(matrix, 10))


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
    trainData = getData(prepare("p1/trainingimages"), m, n, overlap)
    trainLabels = getLabels("p1/traininglabels")

    countTable, priorDists, numOfFeatures, nFeatVals = countNB(trainData, trainLabels, m, n)
    likelihoods = zeros((len(priorDists), numOfFeatures, nFeatVals))

    for i in range(len(priorDists)):
        denom = priorDists[i] * len(trainLabels) + k * nFeatVals
        for f in range(numOfFeatures):
            for v in range(nFeatVals):
                likelihoods[i, f, v] = (countTable[i, f, v] + k)/denom

    return likelihoods, priorDists, numOfFeatures, nFeatVals


def logLikelihood(digit, likelihood):
    out = zeros((28, 28))

    for i in range(len(out)):
        for j in range(len(out[i])):
            out[i, j] = log(likelihood[digit, i * 28 + j, 1])

    return out


def oddsRatios(digit1, digit2):
    out = zeros((28, 28))

    for i in range(len(out)):
        for j in range(len(out[i])):
            out[i, j] = digit1[i, j] - digit2[i, j]

    return out


def classify(m=1, n=1, k=1, overlap=False):
    from collections import OrderedDict
    predictions = []
    model, priorDists, numOfFeatures, nFeatVals = trainNB(m, n, k, overlap)

    toc()
    print "done training"

    testData = getData(prepare("p1/testimages"), m, n, overlap)
    testLabels = getLabels("p1/testlabels")
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
# classify()

model, priorDists, numOfFeatures, nFeatVals = trainNB(m=1, n=1, k=1, overlap=False)

for digit1 in range(10):
    createGreyscale(digit1.__str__() + '.png', logLikelihood(digit1, model))
    for digit2 in range(10):
        createGreyscale(digit1.__str__() + ' over ' + digit2.__str__() + '.png', oddsRatios(logLikelihood(digit1, model), logLikelihood(digit2, model)))

toc()