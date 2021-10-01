from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader
import pandas as pd
import numpy as np

class Word2VecTey:
    def __init__(self, size=100):

        self.size = size
        #wiki-english-20171001
        self.model = Word2Vec(sentences=gensim.downloader.load('text8'), vector_size=100, window=5, min_count=1, workers=4)

    def transform_many(self, phrases):
        dictTey = {}
        indexes = []
        columns = [x for x in range(self.size)]
        for index, phrase in enumerate(phrases):
            dictTey[index] = self.transform_one(phrase, self.size)
            indexes.append(index)

        return pd.DataFrame(data=dictTey.values(),
                            columns=columns, index=indexes)

    def transform_one(self, document):

        dictToBeReturned = {}

        self.fit(document)
        represent = None

        for p in document.split(" "):
            v = np.array(self.model.wv[p])
            if represent is None:
                represent = v
            else:
                represent = np.add(v, represent)

        represent /= len(document.split(" "))

        for count, value in enumerate([val for val in represent]):
            dictToBeReturned[count] = value
        return dictToBeReturned

    def fit(self, document):
        self.model.train(document, total_examples=len(document), epochs=1)

# teste = Word2VecTey()
# teste.transform_many(["213 321 32", "123 eu sou eu sou maluquice meu irmao"])
