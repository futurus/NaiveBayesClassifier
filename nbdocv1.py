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
            # bag of word
            featureMat[i, dictionary.index(word)] = docs[i][word]

            # set of word
            # if docs[i][word] > 0:
            #     featureMat[i, dictionary.index(word)] = 1
            # else:
            #     featureMat[i, dictionary.index(word)] = 0


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

    print "done training"
    toc()

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
classify("p2/8category.training.txt", "p2/8category.testing.txt")


# this part is for top 20 likelihoods and log odds ratios
# dictionary, trainingLabels, trainingMatrix = prepare("p2/8category.training.txt")
# conditionalProbTable = trainNB(trainingMatrix, trainingLabels)
#
# from collections import OrderedDict
#
# dict1 = {}
# dict2 = {}
# dict3 = {}
# dict4 = {}
# for word in dictionary:
#     dict1[word] = log(conditionalProbTable[1, dictionary.index(word)]/conditionalProbTable[3, dictionary.index(word)])
#     dict2[word] = log(conditionalProbTable[5, dictionary.index(word)]/conditionalProbTable[1, dictionary.index(word)])
#     dict3[word] = log(conditionalProbTable[7, dictionary.index(word)]/conditionalProbTable[3, dictionary.index(word)])
#     dict4[word] = log(conditionalProbTable[7, dictionary.index(word)]/conditionalProbTable[1, dictionary.index(word)])
#
# oDict1 = OrderedDict(sorted(dict1.items(), key=lambda x: x[1], reverse=True))
# oDict2 = OrderedDict(sorted(dict2.items(), key=lambda x: x[1], reverse=True))
# oDict3 = OrderedDict(sorted(dict3.items(), key=lambda x: x[1], reverse=True))
# oDict4 = OrderedDict(sorted(dict4.items(), key=lambda x: x[1], reverse=True))
#
# print list(oDict1)[0:20]
# print list(oDict2)[0:20]
# print list(oDict3)[0:20]
# print list(oDict4)[0:20]

toc()