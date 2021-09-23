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

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


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

bow = river.feature_extraction.BagOfWords()

modelHT = river.naive_bayes.GaussianNB()
modelBert = river.naive_bayes.GaussianNB()

metricHT = river.metrics.Accuracy()
metricBert = river.metrics.Accuracy()

bertey = BertEy()



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