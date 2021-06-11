from sklearn.datasets import fetch_20newsgroups
import numpy as np
import river
import pandas as pd
import sys
import os
import re
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

class HashingTrickTey():

    def __init__(self, hashRange: int = 50):
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

        print(document)
        document = document.lower()
        words = re.compile(r"(?u)\b\w\w+\b").findall(document)
        print(document)
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
        print(X.info)

        y = X.pop("target")

        return river.stream.iter_pandas(X = X, y=y)
    
    else:
        print("please enter valid dataset name")


dataset = loadDatasetTey(env = "SMSSpam")
print(dataset)

# Feature extractors
ht = HashingTrickTey()

bow = river.feature_extraction.BagOfWords()

model = river.naive_bayes.MultinomialNB()

metric = river.metrics.Accuracy()



hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

    

for x, y in dataset:
    
    hashingTrick = ht.transform_one(document=x[list(x.keys())[0]])

    probs = model.predict_proba_one(hashingTrick)
    if len(probs) > 0:
        y_pred = max(probs, key=lambda k: probs[k])
    else:
        y_pred = 0
    
    model.learn_one(hashingTrick, y)
    metric.update(y, y_pred)

print(metric)