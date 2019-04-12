import nltk
import math


def getWords(categorizedCorpus, category):
    for fileid in set(categorizedCorpus.fileids(categories=[category])):
        yield categorizedCorpus.words(fileids=[fileid])


def splitData(data, fraction):
    #expects list of data
    l = len(data)
    cutoff = int(math.ceil(l * fraction))
    return data[0:cutoff], data[cutoff:]


corpus = nltk.corpus.reader.CategorizedPlaintextCorpusReader("corpus", ".*" , cat_pattern=r'(\w+)/*', encoding='latin-1')

trainData = {}
testData = {}
labels = corpus.categories()

for label in labels:
    data = getWords(corpus, label)
    instances = [i for i in data if i]
    trainData[label], testData[label] = splitData(instances, 0.7)

print(trainData)