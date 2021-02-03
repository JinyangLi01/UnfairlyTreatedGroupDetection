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


def Prediction(less_attribute_data, attributes, att_to_predict, difference_from_overall_acc=0.2):
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
    AccThreshold = OverallAccuracy - difference_from_overall_acc  # Low accuracy threshold
    return mis_class, mis_num, OverallAccuracy, AccThreshold



def WholeProcessWithOneAlgorithm(original_data_file, selected_attributes, Thc, time_limit, algorithm_function,
                                 att_to_predict, difference_from_overall_acc=0.2):
    original_data = pd.read_csv(original_data_file)
    selected_attributes.append(att_to_predict)
    less_attribute_data = original_data[selected_attributes]
    selected_attributes.remove(att_to_predict)
    mis_class_data, mis_class_num, overall_acc, Tha = Prediction(less_attribute_data, selected_attributes,
                                                                 att_to_predict,
                                                                 difference_from_overall_acc)
    less_attribute_data.drop(att_to_predict, axis=1, inplace=True)



    pattern_with_low_accuracy, num_calculation, execution_time, num_pattern_skipped_mis_c, num_pattern_skipped_whole_c \
        = algorithm_function(less_attribute_data, mis_class_data, Tha, Thc, time_limit)
    print("num_pattern_skipped_mis_c = {}, num_pattern_skipped_whole_c = {}".format(num_pattern_skipped_mis_c, num_pattern_skipped_whole_c))
    return pattern_with_low_accuracy, num_calculation, execution_time, \
           overall_acc, Tha, mis_class_data



def WholeProcessWithTwoAlgorithms(original_data_file, selected_attributes, Thc, time_limit, att_to_predict,
                                  difference_from_overall_acc=0.2):
    original_data = pd.read_csv(original_data_file)
    selected_attributes.append(att_to_predict)
    less_attribute_data = original_data[selected_attributes]
    selected_attributes.remove(att_to_predict)
    mis_class_data, mis_class_num, overall_acc, Tha = Prediction(less_attribute_data,
                                                                 selected_attributes,
                                                                 att_to_predict,
                                                                 difference_from_overall_acc)
    less_attribute_data.drop(att_to_predict, axis=1, inplace=True)
    pattern_with_low_accuracy1, num_calculation1, execution_time1, \
    num_pattern_skipped_mis_c1, num_pattern_skipped_whole_c1 = newalg.GraphTraverse(less_attribute_data,
                                                                                           mis_class_data, Tha, Thc,
                                                                                           time_limit)

    pattern_with_low_accuracy2, num_calculation2, execution_time2, _, _ = naivealg.NaiveAlg(less_attribute_data,
                                                                                           mis_class_data, Tha, Thc,
                                                                                           time_limit)


    # sanity check
    sanity_check = True
    if ComparePatternSets(pattern_with_low_accuracy1, pattern_with_low_accuracy2) is False:
        print("sanity check fails!")
        # print(len(pattern_with_low_accuracy1), "\n", pattern_with_low_accuracy1)
        # print(len(pattern_with_low_accuracy2), "\n", pattern_with_low_accuracy2)
        sanity_check = False
    return sanity_check, pattern_with_low_accuracy1, num_calculation1, execution_time1, \
    num_pattern_skipped_mis_c1, num_pattern_skipped_whole_c1, pattern_with_low_accuracy2, \
    num_calculation2, execution_time2, \
    overall_acc, Tha, mis_class_data


"""
original_data_file = "../../InputData/CreditcardDataset/credit_card_clients_categorized.csv"
Thc = 1
Tha = 0.05
time_limit = 60*10
#selected_attributes = ['limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_0', 'pay_2']
selected_attributes = ['limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_0']

pattern_with_low_accuracy2, num_calculation2, execution_time2, OverallAccuracy2, Tha2, mis_class_data2 = \
    WholeProcessWithOneAlgorithm(original_data_file, selected_attributes, Thc,
                                time_limit, newalg.GraphTraverse,
                                 'default payment next month', Tha)

print(pattern_with_low_accuracy2, "\n", num_calculation2, execution_time2, OverallAccuracy2, Tha2)


"""
