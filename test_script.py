from helper_functions import testFeatureExtractor
from helper_functions import loadDatasetTey
from bert import BertEy
from word2vec import Word2VecTey
from hashing_trick import HashingTrickTey
import river


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

dataset = loadDatasetTey(env = "SMSSpam")

print(testFeatureExtractor([ht, word2Tey], [modelHT, modelW2V], [metricHT, metricW2V], dataset))
