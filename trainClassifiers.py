import nltk




reader = nltk.corpus.reader.CategorizedPlaintextCorpusReader("corpus", ".*" , cat_pattern=r'(\w+)/*')



