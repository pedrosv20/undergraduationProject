from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from datetime import datetime
import numpy as np
import river
import pandas as pd
import sys
import os
import re
import ssl
import time
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from datetime import datetime

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context




class Word2VecTey:
    def __init__(self, size=100):
        self.size = size

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
        splited = [document.split(" ") for i in range(2)]
        model = Word2Vec(splited, vector_size=self.size, window=5, min_count=1, workers=4)

        represent = None

        for p in document.split(" "):
            v = np.array(model.wv[p])
            if represent is None:
                represent = v
            else:
                represent = np.add(v, represent)

        represent /= len(document.split(" "))

        for count, value in enumerate([val for val in represent]):
            dictToBeReturned[count] = value
        return dictToBeReturned


# teste = Word2VecTey()
# teste.transform_many(["213 321 32", "123 eu sou eu sou maluquice meu irmao"])


class BertEy:
    def __init__(self):
        self = self

    def transform_many(self, sentences):
        data = {}
        indexes = []
        columns = [x for x in range(384)]
        for index, sentence in enumerate(sentences):
            data[index] = self.transform_one(sentence)
            indexes.append(index)

        return pd.DataFrame(data=data.values(),
                            columns=columns, index=indexes)

    def transform_one(self, document):
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        sentence_embedding = model.encode(document)
        teste = {}
        for count, value in enumerate([embed for embed in sentence_embedding]):
            teste[count] = value
        return teste


class HashingTrickTey:

    def __init__(self, hashRange: int = 500):
        self.hashRange = hashRange

    def createHashMatrix(self, corpus: [str]):
        documentDict = {}
        count = 0
        for sentence in corpus:
            words = self.transformOne(sentence)
            documentDict[count] = words
            count += 1
        return documentDict

    def transform_one(self, document: str):
        documentDict = {}

        for i in range(self.hashRange):
            documentDict[i] = 0

        # print(document)
        document = document.lower()
        words = re.compile(r"(?u)\b\w\w+\b").findall(document)
        # print(document)
        # words = words.lower()

        for word in words:
            documentDict[hash(word) % self.hashRange] += 1
        return documentDict

    def transform_many(self, corpus: [str]):
        data = {}
        columns = []
        matrix = self.createHashMatrix(corpus)
        for x in matrix.values():
            columns += [*x]

        columns = set(columns)

        for doc in [*matrix]:
            data[doc] = [matrix[doc][i]
                        if i in matrix[doc].keys()
                        else 0
                        for i in columns]
        return pd.DataFrame(data=data.values(),
                            columns=columns, index=matrix.keys())



def loadDatasetTey(env = "SMSSpam"):
    if env == "SMSSpam":
        # Returns river textual stream dataset for binary classification
        return river.datasets.SMSSpam()
        
    elif env == "newsGroups":
        # Returns sklearn news textual database as a river stream
        newsgroups = fetch_20newsgroups(subset='train')

        X = pd.DataFrame({"text": newsgroups.data,
                          "target": newsgroups.target})
        # print(X.info)

        y = X.pop("target")

        return river.stream.iter_pandas(X = X, y=y)
    
    else:
        print("please enter valid dataset name")

print(datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
dataset = loadDatasetTey(env = "SMSSpam")
# print(dataset)
#
# Feature extractors
ht = HashingTrickTey()
bertey = BertEy()
word2Tey = Word2VecTey(size=100)


modelHT = river.naive_bayes.GaussianNB()
modelBert = river.naive_bayes.GaussianNB()
modelW2V = river.naive_bayes.GaussianNB()

metricHT = river.metrics.Accuracy()
metricBert = river.metrics.Accuracy()
metricW2V = river.metrics.Accuracy()





hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

    

for x, y in dataset:

    hashingTrick = ht.transform_one(document=x[list(x.keys())[0]])
    probs = modelHT.predict_proba_one(hashingTrick)
    if len(probs) > 0:
        y_pred = max(probs, key=lambda k: probs[k])
    else:
        y_pred = 0

    modelHT.learn_one(hashingTrick, y)
    metricHT.update(y, y_pred)

print("metricht", metricHT)

for x, y in dataset:
    w2tey = word2Tey.transform_one(document=x[list(x.keys())[0]])
    probs = modelW2V.predict_proba_one(w2tey)
    if len(probs) > 0:
        y_pred = max(probs, key=lambda k: probs[k])
    else:
        y_pred = 0

    modelW2V.learn_one(w2tey, y)
    metricW2V.update(y, y_pred)

print("metricw2v", metricW2V)

for x, y in dataset:
    berrrt = bertey.transform_one(document=x[list(x.keys())[0]])
    probs = modelBert.predict_proba_one(berrrt)
    if len(probs) > 0:
        y_pred = max(probs, key=lambda k: probs[k])
    else:
        y_pred = 0

    modelBert.learn_one(berrrt, y)
    metricBert.update(y, y_pred)

print("metricbert", metricBert)
print(datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))