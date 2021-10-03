from helper_functions import testFeatureExtractor
from helper_functions import loadDatasetTey
from bert import BertEy
from word2vec import Word2VecTey
from hashing_trick import HashingTrickTey
from river import naive_bayes, metrics, stream
import pandas as pd
import re
import time


# Feature extractors
ht = HashingTrickTey()
# bertey = BertEy()
word2Tey = Word2VecTey(size=100)


modelHT = naive_bayes.GaussianNB()
# modelBert = naive_bayes.GaussianNB()
modelW2V = naive_bayes.GaussianNB()

metricHT = metrics.Accuracy()
# metricBert = metrics.Accuracy()
metricW2V = metrics.Accuracy()

print("load dataset")

dataset = loadDatasetTey(env = "yelp")

# print(testFeatureExtractor([ht, word2Tey], [modelHT, modelW2V], [metricHT, metricW2V], dataset))


print("Hashing Trick")
start = time.time()
cont = 0
for instance, label in dataset:
    text_parameter = instance[list(instance.keys())[0]]

    # Removes special characters from text
    text_parameter = re.sub('\W+',' ', text_parameter )
    
    text_parameter = text_parameter.lower()

    ht.fit(text_parameter)
    extracted_features = ht.transform_one(text_parameter)

    probs = modelHT.predict_proba_one(extracted_features)

    if len(probs) > 0:
        y_pred = max(probs, key=lambda k: probs[k])
    else:
        y_pred = 0

    modelHT.learn_one(extracted_features, label)
    metricHT.update(label, y_pred)

    if cont > 500:
        break
    else:
        cont += 1

print("Hashing Trick",metricHT, "Time elapsed (s):", time.time() - start)

start = time.time()
cont = 0
for instance, label in dataset:
    text_parameter = instance[list(instance.keys())[0]]

    # Removes special characters from text
    text_parameter = re.sub('\W+',' ', text_parameter )
    
    text_parameter = text_parameter.lower()

    word2Tey.fit(text_parameter)
    extracted_features = word2Tey.transform_one(text_parameter)

    probs = modelW2V.predict_proba_one(extracted_features)

    if len(probs) > 0:
        y_pred = max(probs, key=lambda k: probs[k])
    else:
        y_pred = 0

    modelW2V.learn_one(extracted_features, label)
    metricW2V.update(label, y_pred)

    if cont > 500:
        break
    else:
        cont += 1

print("Word2Vec",metricW2V, "Time elapsed (s):", time.time() - start)