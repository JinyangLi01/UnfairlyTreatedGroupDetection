"""
New algorithm for minority group detection in general case
Search the graph top-down, generate children using the method in coverage paper to avoid redundancy.
Stop point 1: when finding a pattern satisfying the requirements
Stop point 2: when the cardinality is too small
"""

from itertools import combinations
import pandas as pd
from Algorithms import pattern_count
import time
from Algorithms import Predict_0_20210127 as predict
from Algorithms import NewAlgGeneral_0_20210412 as newalggeneral


def DFSattributes(cur, last, comb, pattern, all_p, mcdes, attributes):
    # print("DFS", attributes)
    if cur == last:
        # print("comb[{}] = {}".format(cur, comb[cur]))
        # print("{} {}".format(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max'])))
        for a in range(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max']) + 1):
            s = pattern.copy()
            s[comb[cur]] = a
            all_p.append(s)
        return
    else:
        # print("comb[{}] = {}".format(cur, comb[cur]))
        # print("{} {}".format(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max'])))
        for a in range(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max']) + 1):
            s = pattern.copy()
            s[comb[cur]] = a
            DFSattributes(cur + 1, last, comb, s, all_p, mcdes, attributes)


def AllPatternsInComb(comb, NumAttribute, mcdes, attributes):  # comb = [1,4]
    # print("All", attributes)
    all_p = []
    pattern = [-1] * NumAttribute
    DFSattributes(0, len(comb) - 1, comb, pattern, all_p, mcdes, attributes)
    return all_p


def num2string(pattern):
    st = ''
    for i in pattern:
        if i != -1:
            st += str(i)
        st += '|'
    st = st[:-1]
    return st


def P1DominatedByP2(P1, P2):
    length = len(P1)
    for i in range(length):
        if P1[i] == -1:
            if P2[i] != -1:
                return False
        if P1[i] != -1:
            if P2[i] != P1[i] and P2[i] != -1:
                return False
    return True

def PatternEqual(m, P):
    length = len(m)
    if len(P) != length:
        return False
    for i in range(length):
        if m[i] != P[i]:
            return False
    return True


# coverage of P among dataset D
def cov(P, D):
    cnt = 0
    for d in D:
        if P1DominatedByP2(d, P):
            cnt += 1
    return cnt


# whether a pattern P is dominated by MUP M
# except from P itself
def PDominatedByM(P, M):
    for m in M:
        if PatternEqual(m, P):
            continue
        if P1DominatedByP2(P, m):
            return True, m
    return False, None


def GenerateChildren(P, whole_data_frame, attributes):
    children = []
    length = len(P)
    i = 0
    for i in range(length-1, -1, -1):
        if P[i] != -1:
            break
    if P[i] == -1:
        i -= 1
    for j in range(i+1, length, 1):
        for a in range(int(whole_data_frame[attributes[j]]['min']), int(whole_data_frame[attributes[j]]['max'])+1):
            s = P.copy()
            s[j] = a
            children.append(s)
    return children




"""
Predictive parity:The fraction of correct positive prediction 
TP/(TP+FP) should be similar for all groups
The higher, the more unfairly treated
"""
def Predictive_parity(whole_data, TPdata, FPdata,
                      delta_Thf, Thc, time_limit):
    time1 = time.time()

    pc_whole_data = pattern_count.PatternCounter(whole_data, encoded=False)
    pc_whole_data.parse_data()
    pc_TP = pattern_count.PatternCounter(TPdata, encoded=False)
    pc_TP.parse_data()
    pc_FP = pattern_count.PatternCounter(FPdata, encoded=False)
    pc_FP.parse_data()


    whole_data_frame = whole_data.describe()
    attributes = whole_data_frame.columns.values.tolist()
    NumAttribute = len(attributes)
    index_list = list(range(0, NumAttribute))  # list[1, 2, ...13]
    denominator = len(TPdata) + len(FPdata)
    if denominator == 0:
        print("len(TPdata) + len(FPdata) = 0, exit")
        return [], 0, 0
    original_thf = len(TPdata) / denominator

    Thf = original_thf + delta_Thf
    print("Predictive_parity, original_thf = {}, Thf = {}".format(original_thf, Thf))

    num_patterns = 0
    pattern_with_low_accuracy = []
    for num_att in range(1, NumAttribute + 1):
        # print("----------------------------------------------------  num_att = ", num_att)
        comb_num_att = list(
            combinations(index_list, num_att))  # list of combinations of attribute index, length num_att
        overTime = False
        for comb in comb_num_att:
            if time.time() - time1 > time_limit:
                overTime = True
                break
            patterns = AllPatternsInComb(comb, NumAttribute, whole_data_frame, attributes)
            for p in patterns:
                st = num2string(p)
                num_patterns += 1
                whole_cardinality = pc_whole_data.pattern_count(st)
                if whole_cardinality < Thc:
                    continue
                tp = pc_TP.pattern_count(st)

                fp = pc_FP.pattern_count(st)
                if tp + fp == 0:
                    continue
                correct_positive_prediction = tp / (tp + fp)

                if correct_positive_prediction > Thf:
                    if PDominatedByM(p, pattern_with_low_accuracy)[0] is False:
                        # allDominatedByCurrentCandidateSet = False
                        pattern_with_low_accuracy.append(p)
        if overTime:
            print("naive alg overtime")
            break
        """
        # stop condition: if all patterns satisfying all conditions are dominated by pattern_with_low_accuracy, stop searching
        if allDominatedByCurrentCandidateSet:
            break
        """
    time2 = time.time()
    execution_time = time2 - time1
    # print("execution time = %s seconds" % execution_time)
    # print(len(pattern_with_low_accuracy))
    # print("num_patterns = ", num_patterns)
    return pattern_with_low_accuracy, num_patterns, execution_time


"""
False positive error rate balance (predictive equality)
The probability of a subject in the actual negative class to have a positive
predictive value FPR = FP/(FP+TN) is similar for all groups.
"""
def False_positive_error_rate_balance_greater_than(whole_data, FPdata, TNdata,
                                      delta_Thf, Thc, time_limit):
    time1 = time.time()

    pc_whole_data = pattern_count.PatternCounter(whole_data, encoded=False)
    pc_whole_data.parse_data()
    pc_FP = pattern_count.PatternCounter(FPdata, encoded=False)
    pc_FP.parse_data()
    pc_TN = pattern_count.PatternCounter(TNdata, encoded=False)
    pc_TN.parse_data()
    
    original_thf = len(FPdata) / (len(FPdata) + len(TNdata))
    Thf = original_thf + delta_Thf
    print("False_positive_error_rate_balance, original_thf = {}, Thf = {}".format(original_thf, Thf))

    whole_data_frame = whole_data.describe()
    attributes = whole_data_frame.columns.values.tolist()
    NumAttribute = len(attributes)
    index_list = list(range(0, NumAttribute))  # list[1, 2, ...13]

    num_patterns = 0
    pattern_with_low_accuracy = []
    for num_att in range(1, NumAttribute + 1):
        # print("----------------------------------------------------  num_att = ", num_att)
        comb_num_att = list(
            combinations(index_list, num_att))  # list of combinations of attribute index, length num_att
        overTime = False
        for comb in comb_num_att:
            if time.time() - time1 > time_limit:
                overTime = True
                break
            patterns = AllPatternsInComb(comb, NumAttribute, whole_data_frame, attributes)
            for p in patterns:
                st = num2string(p)
                num_patterns += 1
                whole_cardinality = pc_whole_data.pattern_count(st)
                if whole_cardinality < Thc:
                    continue
                fp = pc_FP.pattern_count(st)

                tn = pc_TN.pattern_count(st)
                if fp + tn == 0:
                    continue
                FPR = fp / (fp + tn)

                if FPR > Thf:
                    if PDominatedByM(p, pattern_with_low_accuracy)[0] is False:
                        # allDominatedByCurrentCandidateSet = False
                        pattern_with_low_accuracy.append(p)
        if overTime:
            print("naive alg overtime")
            break
        """
        # stop condition: if all patterns satisfying all conditions are dominated by pattern_with_low_accuracy, stop searching
        if allDominatedByCurrentCandidateSet:
            break
        """
    time2 = time.time()
    execution_time = time2 - time1
    # print("execution time = %s seconds" % execution_time)
    # print(len(pattern_with_low_accuracy))
    # print("num_patterns = ", num_patterns)
    return pattern_with_low_accuracy, num_patterns, execution_time


"""
False negative error rate balance (equal opportunity) 
Similar to the above, but considers the probability of falsely classifying
subject in the positive class as negative FNR = FN/(TP+FN)
"""


def False_negative_error_rate_balance_greater_than(whole_data, TPdata, FNdata,
                                      delta_Thf, Thc, time_limit):
    time1 = time.time()

    pc_whole_data = pattern_count.PatternCounter(whole_data, encoded=False)
    pc_whole_data.parse_data()
    pc_FN = pattern_count.PatternCounter(FNdata, encoded=False)
    pc_FN.parse_data()
    pc_TP = pattern_count.PatternCounter(TPdata, encoded=False)
    pc_TP.parse_data()

    whole_data_frame = whole_data.describe()
    attributes = whole_data_frame.columns.values.tolist()
    NumAttribute = len(attributes)
    index_list = list(range(0, NumAttribute))  # list[1, 2, ...13]

    original_thf = len(FNdata) / (len(TPdata) + len(FNdata))
    Thf = original_thf + delta_Thf
    print("False_negative_error_rate_balance, original_thf = {}, Thf = {}".format(original_thf, Thf))

    num_patterns = 0
    pattern_with_low_accuracy = []
    for num_att in range(1, NumAttribute + 1):
        # print("----------------------------------------------------  num_att = ", num_att)
        comb_num_att = list(
            combinations(index_list, num_att))  # list of combinations of attribute index, length num_att
        overTime = False
        for comb in comb_num_att:
            if time.time() - time1 > time_limit:
                overTime = True
                break
            patterns = AllPatternsInComb(comb, NumAttribute, whole_data_frame, attributes)
            for p in patterns:
                st = num2string(p)
                num_patterns += 1
                whole_cardinality = pc_whole_data.pattern_count(st)
                if whole_cardinality < Thc:
                    continue
                fn = pc_FN.pattern_count(st)
                tp = pc_TP.pattern_count(st)

                if fn + tp == 0:
                    continue
                FNR = fn / (fn + tp)

                if FNR > Thf:
                    if PDominatedByM(p, pattern_with_low_accuracy)[0] is False:
                        # allDominatedByCurrentCandidateSet = False
                        pattern_with_low_accuracy.append(p)
        if overTime:
            print("naive alg overtime")
            break
        """
        # stop condition: if all patterns satisfying all conditions are dominated by pattern_with_low_accuracy, stop searching
        if allDominatedByCurrentCandidateSet:
            break
        """
    time2 = time.time()
    execution_time = time2 - time1
    # print("execution time = %s seconds" % execution_time)
    # print(len(pattern_with_low_accuracy))
    # print("num_patterns = ", num_patterns)
    return pattern_with_low_accuracy, num_patterns, execution_time


"""
Equalized odds Combines the previous two definitions. All groups
should have both similar false positive error rate balance FP/(FP+TN)
and false negative error rate balance FN/(TP+FN)
"""
def Equalized_odds(whole_data, TPdata, TNdata, FPdata, FNdata,
                   delta_Thf, Thc, time_limit):
    time1 = time.time()
    delta_Thf_FPR, delta_Thf_FNR = delta_Thf
    pc_whole_data = pattern_count.PatternCounter(whole_data, encoded=False)
    pc_whole_data.parse_data()
    pc_TP = pattern_count.PatternCounter(TPdata, encoded=False)
    pc_TP.parse_data()
    pc_FP = pattern_count.PatternCounter(FPdata, encoded=False)
    pc_FP.parse_data()
    pc_TN = pattern_count.PatternCounter(TNdata, encoded=False)
    pc_TN.parse_data()
    pc_FN = pattern_count.PatternCounter(FNdata, encoded=False)
    pc_FN.parse_data()

    whole_data_frame = whole_data.describe()
    attributes = whole_data_frame.columns.values.tolist()
    NumAttribute = len(attributes)
    index_list = list(range(0, NumAttribute))  # list[1, 2, ...13]

    original_thf_FPR = len(FPdata) / (len(FPdata) + len(TNdata))
    original_thf_FNR = len(FNdata) / (len(TPdata) + len(FNdata))
    Thf_FPR = original_thf_FPR - delta_Thf_FPR
    Thf_FNR = original_thf_FNR + delta_Thf_FNR
    print("Equalized_odds, original_thf_FPR = {}, Thf_FPR = {}".format(original_thf_FPR, Thf_FPR))
    print("original_thf_FNR = {}, Thf_FNR = {}".format(original_thf_FNR, Thf_FNR))

    num_patterns = 0
    pattern_with_low_accuracy = []
    for num_att in range(1, NumAttribute + 1):
        # print("----------------------------------------------------  num_att = ", num_att)
        comb_num_att = list(
            combinations(index_list, num_att))  # list of combinations of attribute index, length num_att
        overTime = False
        for comb in comb_num_att:
            if time.time() - time1 > time_limit:
                overTime = True
                break
            patterns = AllPatternsInComb(comb, NumAttribute, whole_data_frame, attributes)
            for p in patterns:
                st = num2string(p)
                num_patterns += 1
                whole_cardinality = pc_whole_data.pattern_count(st)
                if whole_cardinality < Thc:
                    continue
                fp = pc_FP.pattern_count(st)
                fn = pc_FN.pattern_count(st)

                tp = pc_TP.pattern_count(st)
                if fn + tp == 0:
                    continue
                FNR = fn / (fn + tp)

                tn = pc_TN.pattern_count(st)
                if fp + tn == 0:
                    continue
                FPR = fp / (fp + tn)

                if not (FNR <= Thf_FNR and FPR >= Thf_FPR):
                    if PDominatedByM(p, pattern_with_low_accuracy)[0] is False:
                        # allDominatedByCurrentCandidateSet = False
                        pattern_with_low_accuracy.append(p)
        if overTime:
            print("naive alg overtime")
            break
        """
        # stop condition: if all patterns satisfying all conditions are dominated by pattern_with_low_accuracy, stop searching
        if allDominatedByCurrentCandidateSet:
            break
        """
    time2 = time.time()
    execution_time = time2 - time1
    # print("execution time = %s seconds" % execution_time)
    # print(len(pattern_with_low_accuracy))
    # print("num_patterns = ", num_patterns)
    return pattern_with_low_accuracy, num_patterns, execution_time


"""
All groups should have similar probability of subjects to be accurately predicted 
as positive TP/(TP+FP) and accurately predicted as negative TN/(TN+FN).
"""
def Conditional_use_accuracy_equality(whole_data, TPdata, TNdata, FPdata, FNdata,
                                      delta_Thf, Thc, time_limit):
    time1 = time.time()
    delta_Thf_FP, delta_Thf_FN = delta_Thf
    pc_whole_data = pattern_count.PatternCounter(whole_data, encoded=False)
    pc_whole_data.parse_data()
    pc_TP = pattern_count.PatternCounter(TPdata, encoded=False)
    pc_TP.parse_data()
    pc_FP = pattern_count.PatternCounter(FPdata, encoded=False)
    pc_FP.parse_data()
    pc_TN = pattern_count.PatternCounter(TNdata, encoded=False)
    pc_TN.parse_data()
    pc_FN = pattern_count.PatternCounter(FNdata, encoded=False)
    pc_FN.parse_data()

    whole_data_frame = whole_data.describe()
    attributes = whole_data_frame.columns.values.tolist()

    original_thf_FP = len(TPdata) / (len(FPdata) + len(TPdata))
    original_thf_FN = len(TNdata) / (len(TNdata) + len(FNdata))
    Thf_FP = original_thf_FP - delta_Thf_FP
    Thf_FN = original_thf_FN + delta_Thf_FN
    print("Conditional_use_accuracy_equality, original_thf_FP = {}, Thf_FP = {}".format(original_thf_FP, Thf_FP))
    print("original_thf_FN = {}, Thf_FN = {}".format(original_thf_FN, Thf_FN))

    NumAttribute = len(attributes)
    index_list = list(range(0, NumAttribute))  # list[1, 2, ...13]

    num_patterns = 0
    pattern_with_low_accuracy = []
    for num_att in range(1, NumAttribute + 1):
        # print("----------------------------------------------------  num_att = ", num_att)
        comb_num_att = list(
            combinations(index_list, num_att))  # list of combinations of attribute index, length num_att
        overTime = False
        for comb in comb_num_att:
            if time.time() - time1 > time_limit:
                overTime = True
                break
            patterns = AllPatternsInComb(comb, NumAttribute, whole_data_frame, attributes)
            for p in patterns:
                st = num2string(p)
                num_patterns += 1
                whole_cardinality = pc_whole_data.pattern_count(st)
                if whole_cardinality < Thc:
                    continue
                fp = pc_FP.pattern_count(st)
                tn = pc_TN.pattern_count(st)
                tp = pc_TP.pattern_count(st)
                if tp + fp == 0:
                    continue
                positive_prob = tp / (tp + fp)
                fn = pc_FN.pattern_count(st)
                if tn + fn == 0:
                    continue
                negative_prob = tn / (tn + fn)
                if not (positive_prob >= Thf_FP and negative_prob <= Thf_FN):
                    if PDominatedByM(p, pattern_with_low_accuracy)[0] is False:
                        # allDominatedByCurrentCandidateSet = False
                        pattern_with_low_accuracy.append(p)
        if overTime:
            print("naive alg overtime")
            break
        """
        # stop condition: if all patterns satisfying all conditions are dominated by pattern_with_low_accuracy, stop searching
        if allDominatedByCurrentCandidateSet:
            break
        """
    time2 = time.time()
    execution_time = time2 - time1
    # print("execution time = %s seconds" % execution_time)
    # print(len(pattern_with_low_accuracy))
    # print("num_patterns = ", num_patterns)
    return pattern_with_low_accuracy, num_patterns, execution_time


"""
This definition considers the ratio of error.
A classifier satisfies it if all groups have similar ratio of false negatives and false positives.
"""
def Treatment_equality(whole_data, TPdata, TNdata, FPdata, FNdata,
                       delta_Thf, Thc, time_limit):
    time1 = time.time()
    delta_Thf_ratio = delta_Thf
    pc_whole_data = pattern_count.PatternCounter(whole_data, encoded=False)
    pc_whole_data.parse_data()

    pc_FP = pattern_count.PatternCounter(FPdata, encoded=False)
    pc_FP.parse_data()

    pc_FN = pattern_count.PatternCounter(FNdata, encoded=False)
    pc_FN.parse_data()

    original_thf = len(FPdata) / len(FNdata)
    Thf_ratio = original_thf - delta_Thf
    print("Treatment_equality, original_thf = {}, Thf = {}".format(original_thf, Thf_ratio))

    whole_data_frame = whole_data.describe()
    attributes = whole_data_frame.columns.values.tolist()
    NumAttribute = len(attributes)
    index_list = list(range(0, NumAttribute))  # list[1, 2, ...13]

    num_patterns = 0
    pattern_with_low_accuracy = []
    for num_att in range(1, NumAttribute + 1):
        # print("----------------------------------------------------  num_att = ", num_att)
        comb_num_att = list(
            combinations(index_list, num_att))  # list of combinations of attribute index, length num_att
        overTime = False
        for comb in comb_num_att:
            if time.time() - time1 > time_limit:
                overTime = True
                break
            patterns = AllPatternsInComb(comb, NumAttribute, whole_data_frame, attributes)
            for p in patterns:
                st = num2string(p)
                num_patterns += 1
                whole_cardinality = pc_whole_data.pattern_count(st)
                if whole_cardinality < Thc:
                    continue
                fp = pc_FP.pattern_count(st)

                fn = pc_FN.pattern_count(st)

                if fn == 0:
                    continue

                ratio = fp / fn

                if ratio < Thf_ratio:
                    if PDominatedByM(p, pattern_with_low_accuracy)[0] is False:
                        # allDominatedByCurrentCandidateSet = False
                        pattern_with_low_accuracy.append(p)
        if overTime:
            print("naive alg overtime")
            break
        """
        # stop condition: if all patterns satisfying all conditions are dominated by pattern_with_low_accuracy, stop searching
        if allDominatedByCurrentCandidateSet:
            break
        """
    time2 = time.time()
    execution_time = time2 - time1
    # print("execution time = %s seconds" % execution_time)
    # print(len(pattern_with_low_accuracy))
    # print("num_patterns = ", num_patterns)
    return pattern_with_low_accuracy, num_patterns, execution_time



"""
whole_data: the original data file 
mis_class_data: file containing mis-classified tuples
Tha: delta fairness value 
Thc: size threshold
"""

def NaiveAlg(whole_data, TPdata, TNdata, FPdata, FNdata,
                  delta_thf, Thc, time_limit, fairness_definition = 0):
    time1 = time.time()

    pc_whole_data = pattern_count.PatternCounter(whole_data, encoded=False)
    pc_whole_data.parse_data()
    pc_TP = pattern_count.PatternCounter(TPdata, encoded=False)
    pc_TP.parse_data()
    pc_FP = pattern_count.PatternCounter(FPdata, encoded=False)
    pc_FP.parse_data()
    pc_TN = pattern_count.PatternCounter(TNdata, encoded=False)
    pc_TN.parse_data()
    pc_FN = pattern_count.PatternCounter(FNdata, encoded=False)
    pc_FN.parse_data()

    if fairness_definition == 0:
        return Predictive_parity(whole_data, TPdata, FPdata,
                  delta_thf, Thc, time_limit)
    elif fairness_definition == 1:
        return False_positive_error_rate_balance_greater_than(whole_data, FPdata, TNdata,
                  delta_thf, Thc, time_limit)
    elif fairness_definition == 2:
        return False_negative_error_rate_balance_greater_than(whole_data, TPdata, FNdata,
                  delta_thf, Thc, time_limit)
    elif fairness_definition == 3:
        return Equalized_odds(whole_data, TPdata, TNdata, FPdata, FNdata,
                  delta_thf, Thc, time_limit)
    elif fairness_definition == 4:
        return Conditional_use_accuracy_equality(whole_data, TPdata, TNdata, FPdata, FNdata,
                  delta_thf, Thc, time_limit)
    elif fairness_definition == 5:
        return Treatment_equality(whole_data, TPdata, TNdata, FPdata, FNdata,
                  delta_thf, Thc, time_limit)


#
# # age,workclass,education,educational-num,marital-status
# selected_attributes = ['age', 'workclass', 'education', 'educational-num', 'marital-status']
# original_data_file = "../../InputData/AdultDataset/CleanAdult2.csv"
#
# att_to_predict = 'income'
# time_limit = 20*60
#
# fairness_definition = 0
# delta_thf = 0.1
# thc = 3
#
# less_attribute_data, TP, TN, FP, FN = predict.PredictWithMLReturnTPTNFPFN(original_data_file,
#                                                                          selected_attributes,
#                                                                          att_to_predict)
#
#
# pattern_with_low_fairness1, num_patterns1, t1_ = NaiveAlg(less_attribute_data,
#                                                           TP, TN, FP, FN, delta_thf,
#                                                           thc, time_limit, 5)
#
# print(len(pattern_with_low_fairness1))
# print("time = {} s, num_patterns = {}".format(t1_, num_patterns1), "\n", pattern_with_low_fairness1)
#
# pattern_with_low_fairness2, num_patterns2, t2_ = newalggeneral.GraphTraverse(less_attribute_data,
#                                                           TP, TN, FP, FN, delta_thf,
#                                                           thc, time_limit, 5)
#
# print(len(pattern_with_low_fairness2))
# print("time = {} s, num_patterns = {}".format(t2_, num_patterns2), "\n", pattern_with_low_fairness2)
#
# print("1 in 2")
# for p in pattern_with_low_fairness1:
#     flag = False
#     for q in pattern_with_low_fairness2:
#         if PatternEqual(p, q):
#             flag = True
#     if not flag:
#         print(p)
#
# print("2 in 1")
# for p in pattern_with_low_fairness2:
#     flag = False
#     for q in pattern_with_low_fairness1:
#         if PatternEqual(p, q):
#             flag = True
#     if not flag:
#         print(p)
#
#
