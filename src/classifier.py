#IMPORTS

# TO PLOT DISTRIBUTIONS
import matplotlib.pyplot as plt

# NUMPY
import numpy as np

# ML MODELS
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# ML TEXT HELPERS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# ML METRICS
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score



# THIS FUNCTION FILTERS AND RETURNS ALL THE REVIEWS' DOCS AND LABELS
def read_documents(fileName):
    all_docs = []
    all_labels = []

    with open(fileName, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split(" ", 3)
            all_docs.append(words[3])
            all_labels.append(words[1])

    return all_docs, all_labels

# CONVERT LABELS INTO INDICES
def getTargets(labels):
    targets = []
    for sentiment in labels:
        targets.append(labelToIndexDict[sentiment])
    return targets


# TRAINING FOR NAIVE BAYES MODEL
def trainNaiveBayes(trainDocs, trainTargets):
    return MultinomialNB().fit(trainDocs, trainTargets)


# TRAINING FOR DT MODEL
def traindt(trainDocs, trainTargets):
    return DecisionTreeClassifier(criterion="entropy").fit(trainDocs, trainTargets)


# GIVEN THE ML MODEL AND THE EVALUATION SET, MAKE THE PREDICTIONS
def classify(clf, evalDocs):
    return clf.predict(evalDocs)


# PLOTS THE CONFUSION MATRIX
def plotConfusionMatrix(clf, prediction):
    plot_confusion_matrix(clf, eval_docs_tfidf, evalTargets, labels=[i for i in range(len(targetTypes))])
    return confusion_matrix(evalTargets, prediction, labels=[i for i in range(len(targetTypes))])


# RETRIEVE OUTPUT CLASSIFICATION METRICS
def metrics(predicted):
    precision, recall, f1, support = precision_recall_fscore_support(evalTargets,
                                                                     predicted,
                                                                     labels=[i for i in range(len(targetTypes))],
                                                                     zero_division=0
                                                                     )
    return precision, recall, f1


# MEASURES THE PREDICTION ACCURACY OF THE MODEL
def accuracy(predicted):
    return accuracy_score(evalTargets, predicted)


def modelToFile(modelType, modelMetrics, prediction):
    # CREATE OUTPUT FILE
    file = open(modelType + "-" + textFileData, "w")

    file.write("LEGEND\n-----------------------\n")
    for index in range(len(targetTypes)):
        file.write("%s: %s\n" % (index, targetTypes[index]))
    file.write("\n\n")

    # WRITE CONFUSION MATRIX
    file.write("CONFUSION MATRIX\n")
    file.write("(row is true label, column is predicted label)\n-----------------------\n")
    file.write(np.array2string(modelMetrics[0]))
    file.write("\n\n\n")

    # WRITE PRECISION VALUES
    file.write("PRECISION VALUES\n-----------------------\n")
    for index in range(len(targetTypes)):
        file.write(targetTypes[index] + ": " + str(modelMetrics[1][index].item()) + "\n")
    file.write("\n\n")

    # WRITE RECALL VALUES
    file.write("RECALL VALUES\n-----------------------\n")
    for index in range(len(targetTypes)):
        file.write(targetTypes[index] + ": " + str(modelMetrics[2][index].item()) + "\n")
    file.write("\n\n")

    # WRITE F1-MEASURE VALUES
    file.write("F1-MEASURE VALUES\n-----------------------\n")
    for index in range(len(targetTypes)):
        file.write(targetTypes[index] + ": " + str(modelMetrics[3][index].item()) + "\n")
    file.write("\n\n")

    # WRITE ACCURACY
    file.write("PREDICTION ACCURACY\n-----------------------\n")
    file.write("The model's accuracy is: " + str(modelMetrics[4].item()) + "\n")
    file.write("\n\n")

    # WRITE REVIEW LINE NUMBER AND THE PREDICTION OF THE MODEL
    file.write(
        "BELOW ARE ALL THE PREDICTIONS MADE FOR EACH INSTANCES IN THE EVALUATION SET\n(0-indexed)\n--------------\n")
    index = split_point;
    for pred in prediction:
        file.write('%i, %s%s\n' % (index, pred, ' [Misclassified]' if evalTargets[index - split_point] != pred else ''))
        index += 1

    file.close()



# RETRIEVE ALL THE DATA FROM THE PASSED TEXT FILE
textFileData = ("all_sentiment_shuffled.txt")
all_docs, all_labels = read_documents(textFileData)

# FROM THE RETRIEVED DATA, SPLIT THE DATA INTO TRAINING AND EVALUATION SETS
split_point = int(0.80*len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

# FIND ALL UNIQUE LABELS
targetTypes = sorted(list(set(all_labels)))

# CREATE A DICTIONARY MAPPING LABELS TO THEIR INDEX
labelToIndexDict = dict()
for i in range(len(targetTypes)):
    labelToIndexDict[targetTypes[i]] = i



# RETRIEVE CONVERSION FOR TRAINING AND EVALUATION SETS
trainTargets = getTargets(train_labels)
evalTargets = getTargets(eval_labels)


# PLOT THE DISTRIBUTION OF THE NUMBER OF THE INSTANCES IN EACH CLASS
numberData = [all_labels.count(label) for label in targetTypes]
barGraph = plt.bar(targetTypes,numberData)

for i in range(0, len(barGraph), 2):
    barGraph[i].set_color('r')

plt.title('Number of instances in each class')

# ADD COUNT FOR EACH CLASS TO THE GRAPH
for i, v in enumerate(numberData):
    plt.text(plt.xticks()[0][i] - 0.10, v + 50, str(v))

plt.show()




# CONVERT EACH WORD OF THE VOCABULARY FOUND IN THE TRAINING
# TO A CORRESPONDING INDEX AND COUNT EACH WORD
# NOTE: TO SEE THE VOCABULARY OF THE TRAINING SET: print(countVect.get_feature_names())
# NOTE: TO OUTPUT THE INDEX OF A WORD IN THE VOCAB (from countVect.get_feature_names()): countVect.vocabulary_.get('WORD')
# UNDERSTANDING COUNTVECTORIZER()
# EX |(0, 23)     1|
# 0 is the index corresponding to the review, so 0 is the first review about a bad album
# 23 corresponds to the word at index 23 in countVect.get_feature_names()
# 1 is the number of times word 23 shows up in review 0
countVect = CountVectorizer()

### PREPARING TRAINING SET
train_docs_counts = countVect.fit_transform(train_docs)
tfidf_transformer = TfidfTransformer()
train_docs_tfidf = tfidf_transformer.fit_transform(train_docs_counts)
# TRAINING DONE

### PREPARING EVALUATION SET
eval_docs_counts = countVect.transform(eval_docs)
eval_docs_tfidf = tfidf_transformer.transform(eval_docs_counts)
# EVALUATION SET DONE


# NAIVES BAYES CLASSIFIER
nb = trainNaiveBayes(train_docs_tfidf, trainTargets)
nbPrediction = classify(nb,eval_docs_tfidf)
nbMetrics=[]

# CONFUSION MATRIX
nbMetrics.append(plotConfusionMatrix(nb,nbPrediction))

# RETRIEVE METRICS
nbPrecision, nbRecall, nbF1_measure = metrics(nbPrediction)
nbMetrics.append(nbPrecision)
nbMetrics.append(nbRecall)
nbMetrics.append(nbF1_measure)

# RETRIEVE ACCURACY OF PREDICTIONS
nbMetrics.append(accuracy(nbPrediction))


# DT CLASSIFIER
dt = traindt(train_docs_tfidf, trainTargets)
dtPrediction = classify(dt,eval_docs_tfidf)
dtMetrics=[]

# CONFUSION MATRIX
dtMetrics.append(plotConfusionMatrix(dt,dtPrediction))

# RETRIEVE METRICS
dtPrecision, dtRecall, dtF1_measure = metrics(dtPrediction)
dtMetrics.append(dtPrecision)
dtMetrics.append(dtRecall)
dtMetrics.append(dtF1_measure)

# RETRIEVE ACCURACY OF PREDICTIONS
dtMetrics.append(accuracy(dtPrediction))



# OUTPUT TO FILES
modelToFile("NaiveBayesClassifier", nbMetrics, nbPrediction)
modelToFile("DT", dtMetrics, dtPrediction)
