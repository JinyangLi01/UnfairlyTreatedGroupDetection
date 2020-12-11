"""
The whole process:
- train a ml model
- use the model to test
- find mis-classified tuples
- apply NewAlg / NaiveAlg
"""


import pattern_count
import pandas as pd
import NewAlg_0_20201128 as newalg
import NaiveAlg_0_20201111 as naivealg
import time
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree



def Prediction(less_attribute_data, attributes):
    # splitting data arrays into two subsets: for training data and for testing data
    X_train, X_test, y_train, y_test = train_test_split(less_attribute_data[attributes], less_attribute_data['income'], test_size=0.5, random_state=1)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    X_test_res = X_test.copy()
    X_test_res['act'] = y_test  # actual result
    X_test_res['pred'] = X_test.apply(lambda x: clf.predict([x.to_list()]), axis=1)
    mis_class = X_test_res.loc[X_test_res['act'] != X_test_res['pred']]
    mis_class.drop('act', axis=1, inplace=True)
    mis_class.drop('pred', axis=1, inplace=True)
    mis_num = len(mis_class)

    OverallAccuracy = (1 - mis_num / len(X_test))  # accuracy
    AccThreshold = OverallAccuracy - 0.2  # Low accuracy threshold

    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(clf,
                       # feature_names=attributes,
                       class_names=['0', '1'],
                       filled=True,
                       rounded=True)
    fig.savefig("tree2.png")

    return mis_class, mis_num, OverallAccuracy, AccThreshold



def WholeProcess(original_data_file, selected_attributes, Thc, algorithm_function):
    original_data = pd.read_csv(original_data_file)
    selected_attributes.append('income')
    less_attribute_data = original_data[selected_attributes]
    selected_attributes.remove('income')
    mis_class_data, mis_class_num, OverallAccuracy, Tha = Prediction(less_attribute_data, selected_attributes)
    less_attribute_data.drop('income', axis=1, inplace=True)
    pattern_with_low_accuracy, num_patterns_checked, execution_time = algorithm_function(less_attribute_data, mis_class_data, Tha, Thc)
    return pattern_with_low_accuracy, num_patterns_checked, execution_time, OverallAccuracy, Tha, mis_class_data




original_data_file = "../InputData/CleanAdult2.csv"
Thc = 15
selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender']
pattern_with_low_accuracy, num_patterns_checked, execution_time, OverallAccuracy, Tha, mis_class_data = WholeProcess(original_data_file, selected_attributes, Thc, newalg.GraphTraverse)

print(pattern_with_low_accuracy, "\n", num_patterns_checked, execution_time, OverallAccuracy, Tha)




