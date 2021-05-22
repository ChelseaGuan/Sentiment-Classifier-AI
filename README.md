# Sentiment-Classifier-AI

## About the Project
Worked collaboratively in a team of four developers to develop a sentiment classifier using two different machine learning models: Naive Bayes and Decision Tree.   
Trained the model with 80% of the sentiment data set, in this case, a set of reviews. The program uses the other 20% to test the models ability ot predict whether a review is positive or negative.   
The program also outputs a file with relevant metrics (e.g. accuracy, precision, recall, F1) and prediction results for each model.   

### Development
 This project was developed in the PyCharm IDE from JetBrains. The language used is Python.
 
### How Run the Code

Certain python libraries will need to be installed. To do so, open the command prompt, and execute from the current directory   
```pip install -r requirements.txt```

In the same command prompt, execute   
```python classifier.py```

### Sample output
The sample data set all_sentiment_shuffled.txt is located in src with the Python script.
Full sample outputs are provided in the 'Sample out files' folder.   

#### Sample histogram plotting the distribution of the number of the instances in each class
![Plot of the distribution of the number of the instances in each class](Sample%20output%20files/nb_of_instances_in_each_class_histogram.PNG)  

#### Output for the Naive Bayes classifier (NaiveBayesClassifier-all_sentiment_shuffled.txt)
```
LEGEND
-----------------------
0: neg
1: pos


CONFUSION MATRIX
(row is true label, column is predicted label)
-----------------------
[[1016  214]
 [ 194  959]]


PRECISION VALUES
-----------------------
neg: 0.8396694214876033
pos: 0.8175618073316283


RECALL VALUES
-----------------------
neg: 0.8260162601626017
pos: 0.8317432784041631


F1-MEASURE VALUES
-----------------------
neg: 0.8327868852459016
pos: 0.824591573516767


PREDICTION ACCURACY
-----------------------
The model's accuracy is: 0.8287872429710449


BELOW ARE ALL THE PREDICTIONS MADE FOR EACH INSTANCES IN THE EVALUATION SET
(0-indexed)
--------------
9531, 0
9532, 1
9533, 0
9534, 1
9535, 1
9536, 0
9537, 1
9538, 0
9539, 0
9540, 1 [Misclassified]
9541, 1 [Misclassified]
9542, 1
...
11905, 0
11906, 1
11907, 0
11908, 1 [Misclassified]
11909, 1 [Misclassified]
11910, 0 [Misclassified]
11911, 1
11912, 0
11913, 0
```
