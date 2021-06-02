"""
The whole process:
- train a ml model
- use the model to test
- find mis-classified tuples
- apply NewAlg / NaiveAlg
"""


from Algorithms import pattern_count
import pandas as pd
from Algorithms import NewAlg_0_20201128 as newalg
from Algorithms import NaiveAlg_0_20201111 as naivealg
import time
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def ComparePatternSets(set1, set2):
    len1 = len(set1)
    len2 = len(set2)
    if len1 != len2:
        return False
    for p in set1:
        found = False
        for q in set2:
            if newalg.PatternEqual(p, q):
                found = True
                break
        if found is False:
            return False
    return True

"""
predict with ml, return mis_classified data
"""
def Prediction(less_attribute_data, attributes, att_to_predict):
    # splitting data arrays into two subsets: for training data and for testing data
    X_train, X_test, y_train, y_test = train_test_split(less_attribute_data[attributes], less_attribute_data[att_to_predict],
                                                        test_size=0.5, random_state=1)
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
    return mis_class, mis_num, OverallAccuracy


"""
predict with ml, return TP, TN, FP, FN data
"""
def PredictionReturnTPTNFPFN(less_attribute_data, attributes, att_to_predict):
    # splitting data arrays into two subsets: for training data and for testing data
    X_train, X_test, y_train, y_test = train_test_split(less_attribute_data[attributes], less_attribute_data[att_to_predict],
                                                        test_size=0.5, random_state=1)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    X_test_res = X_test.copy()
    X_test_res['act'] = y_test  # actual result
    X_test_res['pred'] = X_test.apply(lambda x: clf.predict([x.to_list()]), axis=1)


    TP_data = X_test_res.loc[X_test_res['pred'] == 1]
    TP_data = TP_data.loc[TP_data['act'] == 1]
    TN_data = X_test_res.loc[X_test_res['pred'] == 0]
    TN_data = TN_data.loc[TN_data['act'] == 0]
    FP_data = X_test_res.loc[X_test_res['pred'] == 1]
    FP_data = FP_data.loc[FP_data['act'] == 0]
    FN_data = X_test_res.loc[X_test_res['pred'] == 0]
    FN_data = FN_data.loc[FN_data['act'] == 1]

    TP_data.drop('act', axis=1, inplace=True)
    TP_data.drop('pred', axis=1, inplace=True)
    TN_data.drop('act', axis=1, inplace=True)
    TN_data.drop('pred', axis=1, inplace=True)
    FP_data.drop('act', axis=1, inplace=True)
    FP_data.drop('pred', axis=1, inplace=True)
    FN_data.drop('act', axis=1, inplace=True)
    FN_data.drop('pred', axis=1, inplace=True)

    return TP_data, TN_data, FP_data, FN_data



def PredictWithML(original_data_file, selected_attributes, att_to_predict):
    original_data = pd.read_csv(original_data_file)
    print(len(original_data))
    selected_attributes.append(att_to_predict)
    less_attribute_data = original_data[selected_attributes]
    selected_attributes.remove(att_to_predict)
    mis_class_data, mis_class_num, overall_acc = Prediction(less_attribute_data, selected_attributes,
                                                                 att_to_predict)
    less_attribute_data.drop(att_to_predict, axis=1, inplace=True)
    return less_attribute_data, mis_class_data, overall_acc

"""
predict with ml
return TP, TN, FP, FN
"""
def PredictWithMLReturnTPTNFPFN(original_data_file, selected_attributes, att_to_predict):
    original_data = pd.read_csv(original_data_file)

    selected_attributes.append(att_to_predict)
    less_attribute_data = original_data[selected_attributes]
    selected_attributes.remove(att_to_predict)
    TP_data, TN_data, FP_data, FN_data = PredictionReturnTPTNFPFN(less_attribute_data, selected_attributes,
                                                                 att_to_predict)
    less_attribute_data.drop(att_to_predict, axis=1, inplace=True)
    return less_attribute_data, TP_data, TN_data, FP_data, FN_data

