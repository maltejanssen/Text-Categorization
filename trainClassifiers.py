import nltk
import math
import pandas


def getWords(categorizedCorpus, category):
    for fileid in set(categorizedCorpus.fileids(categories=[category])):
        yield categorizedCorpus.words(fileids=[fileid])


def splitData(data, fraction):
    #expects list of data
    l = len(data)
    cutoff = int(math.ceil(l * fraction))
    return data[0:cutoff], data[cutoff:]






#SFU Data
corpus = nltk.corpus.reader.CategorizedPlaintextCorpusReader("corpus", ".*", cat_pattern=r'(\w+)/*', encoding='latin-1')
trainData = {}
testData = {}
labels = corpus.categories()

for label in labels:
    data = getWords(corpus, label)
    instances = [i for i in data if i]
    trainData[label], testData[label] = splitData(instances, 0.7)



#BBC data
data = pandas.read_csv("bbc-text.csv")
dataDict = {}



for index, row in data.iterrows():
    if row["category"] not in dataDict:
        dataDict[row["category"]] = []
    dataDict[row["category"]].append([nltk.word_tokenize(row["text"])])



print(dataDict)
# trainDataBBC = {}
# testDataBBC = {}
#
#
#
#
# trainDataBBC, testDataBBC = splitData(data, 0.7)
# print(trainDataBBC)

