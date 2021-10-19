# from helper_functions import testFeatureExtractor
from helper_functions import loadDatasetTey
from bert import BertEy
from word2vec import Word2VecTey
from hashing_trick import HashingTrickTey
from river import naive_bayes, metrics, stream
import sys
import re
import time

totalInstances = 0

if "--totalInstances" in sys.argv:
    argumentIndex = sys.argv.index("--totalInstances")
    totalInstances = int(sys.argv[argumentIndex + 1])
    print(totalInstances)
if "-h" in sys.argv:
    print("Usage: \n\t--totalInstances : sets the number of instaces to be used on testing")
    exit()

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

# print(testFeatureExtractor([ht, word2Tey], [modelHT, modelW2V], [m    etricHT, metricW2V], dataset))

print("Start loading twitter dataset")

dataset = loadDatasetTey(env="twitter")
cache = stream.Cache()

print("Starting test routine for", totalInstances, "intances")
cont = 0
start = 0
for instance, label in cache(dataset, key="river_cache"):
    # Timer is started ghere so the time for loading the
    # dataset is not considered
    if cont == 0:
        cont += 1
        # print(instance)
        start = time.time()

    # Retrieve instance's textual parameter
    # It is expected the instance to be a dictionary and 
    # the first parameter (and only one) to be a text
    text_parameter = instance[list(instance.keys())[0]]

    # Removes special characters from text
    text_parameter = re.sub('\W+', ' ', text_parameter)

    text_parameter = text_parameter.lower()

    ht.fit(text_parameter)
    extracted_features = ht.transform_one(text_parameter)

    probs = modelHT.predict_proba_one(extracted_features)
    
    # {1: %, 2:%}
    # {"4", "5"}

    if len(probs) > 0:
        y_pred = max(probs, key=lambda k: probs[k])
    else:
        y_pred = 0

    modelHT.learn_one(extracted_features, label)
    metricHT.update(label, y_pred)

    if totalInstances != 0:
        if cont > totalInstances:
            break
        if cont % (totalInstances/10) == 0:
            print("y_pred", y_pred, "label", label)
            print("probs", probs)
            print("\t", cont, "of", totalInstances, "instances processed")
    cont += 1

print("Hashing Trick", metricHT, "Time elapsed (sec):", time.time() - start)

exit()
cont = 0
start = 0
for instance, label in cache(dataset, key="river_cache"):
    # We start timer here so the time for loading the
    # dataset is not considered
    if cont == 0:
        cont += 1
        # print(instance)
        start = time.time()

    # Retrieve instance's textual parameter
    # It is expected the instance to be a dictionary and 
    # the first parameter (and only one) to be a text
    text_parameter = instance[list(instance.keys())[0]]

    # Removes special characters from text
    text_parameter = re.sub('\W+', ' ', text_parameter)

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


    if totalInstances != 0:
        if cont > totalInstances:
            break
        if cont % (totalInstances/10) == 0:
            print("y_pred", y_pred, "label", label)
            print("\t", cont, "of", totalInstances, "instances processed")
    cont += 1

print("Word2Vec", metricW2V, "Time elapsed (s):", time.time() - start)

cont = 0
start = 0
for instance, label in cache(dataset, key="river_cache"):
    # We start timer here so the time for loading the
    # dataset is not considered
    if cont == 0:
        cont += 1
        # print(instance)
        start = time.time()

    # Retrieve instance's textual parameter
    # It is expected the instance to be a dictionary and 
    # the first parameter (and only one) to be a text
    text_parameter = instance[list(instance.keys())[0]]

    # Removes special characters from text
    text_parameter = re.sub('\W+', ' ', text_parameter)

    text_parameter = text_parameter.lower()

    bertey.fit(text_parameter)
    extracted_features = bertey.transform_one(text_parameter)

    probs = modelBert.predict_proba_one(extracted_features)

    if len(probs) > 0:
        y_pred = max(probs, key=lambda k: probs[k])
    else:
        y_pred = 0

    modelBert.learn_one(extracted_features, label)
    metricBert.update(label, y_pred)


    if totalInstances != 0:
        if cont > totalInstances:
            break
        if cont % (totalInstances/10) == 0:
            print("\t", cont, "of", totalInstances, "instances processed")
    cont += 1

print("BERT", metricBert, "Time elapsed (s):", time.time() - start)

cache.clear_all()