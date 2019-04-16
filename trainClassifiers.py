import nltk
import math
import pandas
import argparse
import os
import pickle
from nltk.classify import DecisionTreeClassifier, MaxentClassifier, NaiveBayesClassifier
from nltk.classify.svm import SvmClassifier
from nltk.metrics import f_measure, precision, recall
import collections
import numpy
from sklearn.model_selection import KFold


classifierOptions = ["decisionTree", "NaiveBayes", "maxent", "sklearnExtraTreesClassifier",
                     "sklearnGradientBoostingClassifier", "sklearnRandomForestClassifier", "sklearnLogisticRegression",
                     "sklearnBernoulliNB", "sklearnMultinomialNB", "sklearnLinearSVC", "sklearnNuSVC", "sklearnSVC",
                     "sklearnDecisionTreeClassifier"]


def getWords(categorizedCorpus, category):
    for fileid in set(categorizedCorpus.fileids(categories=[category])):
        yield categorizedCorpus.words(fileids=[fileid])


def splitData(data, fraction):
    #expects list of data
    l = len(data)
    cutoff = int(math.ceil(l * fraction))
    return data[0:cutoff], data[cutoff:]


def loadData(data, fraction):
    trainData = {}
    testData = {}

    if data == "BBC":
        data = pandas.read_csv("bbc-text.csv")
        dataDict = {}

        for index, row in data.iterrows():
            if row["category"] not in dataDict:
                dataDict[row["category"]] = []
            dataDict[row["category"]].append(nltk.word_tokenize(row["text"]))

        for label in dataDict:
            trainData[label], testData[label] = splitData(dataDict[label], fraction)

    elif data == "SFU":
        corpus = nltk.corpus.reader.CategorizedPlaintextCorpusReader("corpus", ".*", cat_pattern=r'(\w+)/*', encoding='latin-1')
        trainData = {}
        testData = {}
        labels = corpus.categories()

        for label in labels:
            data = getWords(corpus, label)
            instances = [i for i in data if i]
            trainData[label], testData[label] = splitData(instances, fraction)
    else:
        raise ValueError("Data not supported")

    return trainData, testData, labels


def wordCountsFeature(words):
    return dict(probability.FreqDist((w for w in words)))


def bagOfWordsFeature(words):
    return dict([(word, True) for word in words])


def extractFeatures(label_instances, featx):
    feats = []
    for label, instances in label_instances.items():
        feats.extend([(featx(i), label) for i in instances])
    return feats


def makeClassifier(args):
    """ configurates classifiers with arguments

    :param trainer: String: Name of classifier
    :param args: Classifier Options
    :return: trainFunction of configurated classifier
    """
    print(args.classifier)
    trainArgs = {}

    if args.classifier not in classifierOptions:
        raise ValueError("classifier %s is not supported" % args.classifier)
    if args.classifier == "NaiveBayes":
        classifierTrain = NaiveBayesClassifier.train
    elif args.classifier == "maxent":
        classifierTrain = MaxentClassifier.train
        trainArgs['max_iter'] = args.maxIter
        trainArgs['min_ll'] = args.minll
        trainArgs['min_lldelta'] = args.minlldelta
    elif args.classifier == "decisionTree":
        classifierTrain = DecisionTreeClassifier.train
        trainArgs['binary'] = False
        trainArgs['entropy_cutoff'] = args.entropyCutoff
        trainArgs['depth_cutoff'] = args.depthCutoff
        trainArgs['support_cutoff'] = args.supportCutoff
    elif args.classifier == "sklearnExtraTreesClassifier":
        classifierTrain = scikitlearn.SklearnClassifier(ExtraTreesClassifier(criterion=args.criterion, max_feats=args.maxFeats, depth_cutoff=args.depthCutoff, n_estimators=args.nEstimators)).train
    elif args.classifier == "sklearnGradientBoostingClassifier":
        classifierTrain = scikitlearn.SklearnClassifier(GradientBoostingClassifier(learning_rate=args.learningRate, max_feats=args.maxFeats, depth_cutoff=args.depthCutoff, n_estimators=arrgs.nEstimators)).train
    elif args.classifier == "sklearnRandomForestClassifier":
        classifierTrain = scikitlearn.SklearnClassifier(RandomForestClassifier(criterion=args.criterion, max_feats=args.maxFeats, depth_cutoff=args.depthCutoff, n_estimators=args.nEstimators)).train
    elif args.classifier == "sklearnLogisticRegression":
        classifierTrain = scikitlearn.SklearnClassifier(LogisticRegression(penalty=args.penalty, C=arg.C)).train
    elif args.classifier == "sklearnBernoulliNB":
        classifierTrain == scikitlearn.SklearnClassifier(sklearnBernoulliNB(alpha=args.alpha)).train
    elif args.classifier == "sklearnMultinomialNB":
        classifierTrain = scikitlearn.SklearnClassifier(MultinomialNB(alpha=args.alpha)).train
    elif args.classifier == "sklearnLinearSVC":
        classifierTrain = scikitlearn.SklearnClassifier(LinearSVC(C=args.C, penalty=args.penalty, loss=args.loss)).train
    elif args.classifier == "sklearnNuSVC":
        classifierTrain = scikitlearn.SklearnClassifier(NuSVC(nu=args.nu, kernel=args.kernel)).train
    elif args.classifier == "sklearnSVC":
        classifierTrain = scikitlearn.SklearnClassifier(SVC(C=args.C, kernel=args.kernel)).train
    elif args.classifier == "sklearnDecisionTreeClassifier":
        classifierTrain = scikitlearn.SklearnClassifier(DecisionTreeClassifier(criterion=args.criterion, max_feats=args.maxFeats, depth_cutoff=args.DepthCutoff)).train


    def trainf(trainFeats):
        return classifierTrain(trainFeats, **trainArgs)
    return trainf




def safeClassifier(chunker, args):
    """ safes(pickles) classifierChunker

    :param chunker: chunker/cLassifier to be safed
    :param args: Arguments containing name of classifier
    """
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, "Classifiers")
    file = os.path.join(path, args.classifier)

    f = open(file, 'wb')
    pickle.dump(chunker, f)
    f.close()


def ref_test_sets(classifier, test_feats):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feat, label) in enumerate(test_feats):
        refsets[label].add(i)
        observed = classifier.classify(feat)
        testsets[observed].add(i)

    return refsets, testsets


def evaluate(classifier, evalFeats, labels):
    #old eval without cross Validation
    try:
        print('accuracy: %f' % nltk.classify.util.accuracy(classifier, evalFeats))
    except ZeroDivisionError:
        print('accuracy: 0')

    refsets, testsets = ref_test_sets(classifier, evalFeats)
    for label in labels:
        ref = refsets[label]
        test = testsets[label]
        print('%s precision: %f' % (label, precision(ref, test) or 0))
        print('%s recall: %f' % (label, recall(ref, test) or 0))
        print('%s f-measure: %f' % (label, f_measure(ref, test) or 0))


def crossVal(instances, trainf, testf, folds=10):
    #add shuffling? # random.shuffle(instances)
    kf = KFold(n_splits=folds)
    sum = 0
    for train, test in kf.split(instances):
        trainData = instances[train[0]:train[-1]]
        testData = instances[test[0]:test[-1]]
        classifier = trainf(trainData)
        sum += testf(classifier, testData)
    average = sum / folds
    return average


def train(args):
    """ trains a Classifier based on passed Arguments

       :param args: Arguments passed by user-> see main below
       """
    if args.classifier not in classifierOptions:
        raise ValueError("classifier %s is not supported" % args.classifier)
    trainData, testData, labels = loadData(args.corpus, args.fraction)
    featx = bagOfWordsFeature

    trainFeats = extractFeatures(trainData, featx)
    testFeats = extractFeatures(testData, featx)
    #print(trainFeats)

    trainf = makeClassifier(args)

    print(crossVal(trainFeats, trainf, nltk.classify.util.accuracy ,  folds=args.crossFold))

    #classifier = trainf(trainFeats) #old used to be instead of cross val
   # safeClassifier(classifier, args) #pickling not possible with cross val


    # if args.eval:
    #     # trainChunks = chunkTrees2trainChunks(evalChunkTrees)
    #     evaluate(classifier, testFeats, labels)




def addArguments():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Script that trains text-Classifiers')
    parser.add_argument("--corpus", default=r"Data\Corpus", help="Options: BBC, SFU")
    parser.add_argument("--fraction", default=0.7, help="split-fraction of data into train and test dataset")
    parser.add_argument("--classifier", default="all", help="Classifier to be used; Options:")  # TODO options
    parser.add_argument("--eval", action='store_true', default=True, help="do evaluation")
    parser.add_argument("--backoff", default="True", help="turn on/off backoff functionality for n-grams")
    parser.add_argument("--crossFold", default="10",  type=int, help="number of folds")

    maxentGroup = parser.add_argument_group("Maxent Classifier")
    maxentGroup.add_argument("-maxIter", default=10, type=int,
                             help="Terminate after default: %(default)d iterations.")
    maxentGroup.add_argument("--minll", default=0, type=float,
                             help="Terminate after the negative average log-likelihood drops under default: %(default)f")
    maxentGroup.add_argument("--minlldelta", default=0.1, type=float,
                             help="Terminate if a single iteration improvesnlog likelihood by less than default, default is %(default)f")

    decisiontreeGroup = parser.add_argument_group("Decision Tree Classifier")
    decisiontreeGroup.add_argument("--entropyCutoff", default=0.05, type=float,
                                   help="default: 0.05")
    decisiontreeGroup.add_argument("--depthCutoff", default=100, type=int,
                                   help="default: 100")
    decisiontreeGroup.add_argument("--supportCutoff", default=10, type=int,
                                   help="default: 10")

    sklearnGroup = parser.add_argument_group('sklearn Classifiers',
                                              'These options are used by the sklearn algorithms')
    sklearnGroup.add_argument('--alpha', type=float, default=1.0,
                               help='smoothing parameter for naive bayes classifiers, default is %(default)s')
    sklearnGroup.add_argument('--C', type=float, default=1.0,
                               help='penalty parameter, default is %(default)s')
    sklearnGroup.add_argument('--criterion', choices=['gini', 'entropy'],
                               default='gini', help='Split quality function, default is %(default)s')
    sklearnGroup.add_argument('--kernel', default='rbf',
                               choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                               help='kernel type for support vector machine classifiers, default is %(default)s')
    sklearnGroup.add_argument('--learningRate', type=float, default=0.1,
                               help='learning rate, default is %(default)s')
    sklearnGroup.add_argument('--loss', choices=['l1', 'l2'],
                               default='l2', help='loss function, default is %(default)s')
    sklearnGroup.add_argument('--nEstimators', type=int, default=10,
                               help='Number of trees for Decision Tree ensembles, default is %(default)s')
    sklearnGroup.add_argument('--nu', type=float, default=0.5,
                               help='upper bound on fraction of training errors & lower bound on fraction of support vectors, default is %(default)s')
    sklearnGroup.add_argument('--penalty', choices=['l1', 'l2'],
                               default='l2', help='norm for penalization, default is %(default)s')
    sklearnGroup.add_argument('--tfidf', default=False, action='store_true',
                               help='Use TfidfTransformer')
    sklearnGroup.add_argument('--maxFeats', default="auto",  help='max Feats')

    return parser.parse_args()



if  __name__ == '__main__':
    args = addArguments()

    if args.classifier == "all":
        for classifier in classifierOptions:
            args.classifier = classifier
            train(args)
    else:
        train(args)


