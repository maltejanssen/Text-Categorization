import nltk
import math
import pandas
import argparse
try:
	from nltk.compat import iteritems
except ImportError:
	def iteritems(d):
		return d.items()


classifierOptions =  ["1-gram", "2-gram", "3-gram", "decisionTree", "NaiveBayes", "Maxent"]

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

    return trainData, testData


def wordCountsFeature(words):
    return dict(probability.FreqDist((w for w in words)))


def bagOfWordsFeature(words):
    return dict([(word, True) for word in words])


def extractFeatures(label_instances, featx):
    feats = []
    for label, instances in iteritems(label_instances):
        feats.extend([(featx(i), label) for i in instances])
    return feats


def makeClassifier(trainer, args):
    """ configurates classifiers with arguments

    :param trainer: String: Name of classifier
    :param args: Classifier Options
    :return: trainFunction of configurated classifier
    """
    trainArgs = {}
    if trainer == "NaiveBayes":
        classifierTrain = NaiveBayes.train
    elif trainer == "maxent":
        classifierTrain = MaxentClassifier.train
        trainArgs['max_iter'] = args.maxIter
        trainArgs['min_ll'] = args.minll
        trainArgs['min_lldelta'] = args.minlldelta
    elif trainer == "decisionTree":
        classifierTrain = DecisionTreeClassifier.train
        trainArgs['binary'] = False
        trainArgs['entropy_cutoff'] = args.entropyCutoff
        trainArgs['depth_cutoff'] = args.depthCutoff
        trainArgs['support_cutoff'] = args.supportCutoff

    def train(trainFeats):
        return classifierTrain(trainFeats, **trainArgs)
    return train


def evaluate(evalFeats):
    return None


def train(args):
    """ trains a Classifier based on passed Arguments

       :param args: Arguments passed by user-> see main below
       """
    print("wtf")
    if args.classifier not in classifierOptions:
        raise ValueError("classifier %s is not supported" % args.classifier)
    trainData, testData = loadData(args.corpus, args.fraction)
    featx = bagOfWordsFeature

    trainFeats = extractFeatures(trainData, featx)
    testFeats = extractFeatures(testData, featx)

    classifier = makeClassifier(trainFeats, args)


    if args.eval:
        # trainChunks = chunkTrees2trainChunks(evalChunkTrees)
        eval = evaluate(testFeats)
        print(eval)



def addArguments():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Script that trains text-Classifiers')
    parser.add_argument("--corpus", default=r"Data\Corpus", help="Options: BBC, SFU")
    parser.add_argument("--fraction", default=0.7, help="split-fraction of data into train and test dataset")
    parser.add_argument("--classifier", default="all", help="Classifier to be used; Options:")  # TODO options
    parser.add_argument("--eval", action='store_true', default=True, help="do evaluation")
    parser.add_argument("--backoff", default="True", help="turn on/off backoff functionality for n-grams")

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

    return parser.parse_args()



if  __name__ == '__main__':
    args = addArguments()

    if args.classifier == "all":
        for classifier in classifierOptions:
            args.classifier = classifier
            train(args)
    else:
        train(args)


