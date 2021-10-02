from helper_functions import testFeatureExtractor
from helper_functions import loadDatasetTey
from bert import BertEy
from word2vec import Word2VecTey
from hashing_trick import HashingTrickTey
from river import naive_bayes, metrics, stream
import pandas as pd


# Feature extractors
ht = HashingTrickTey()
bertey = BertEy()
word2Tey = Word2VecTey(size=100)


modelHT = naive_bayes.GaussianNB()
modelBert = naive_bayes.GaussianNB()
modelW2V = naive_bayes.GaussianNB()

metricHT = metrics.Accuracy()
metricBert = metrics.Accuracy()
metricW2V = metrics.Accuracy()

print("started")

dataset = loadDatasetTey(env = "yelp")

print(testFeatureExtractor([ht, word2Tey], [modelHT, modelW2V], [metricHT, metricW2V], dataset))