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
whole_data: the original data file 
mis_class_data: file containing mis-classified tuples
Tha: threshold of accuracy 
Thc: threshold of cardinality
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
        return False_positive_error_rate_balance(whole_data, FPdata, TNdata,
                  delta_thf, Thc, time_limit)
    elif fairness_definition == 2:
        return False_negative_error_rate_balance(whole_data, TPdata, FNdata,
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
# pattern_with_low_fairness1, num_calculation1, t1_ = NaiveAlg(less_attribute_data,
#                                                           TP, TN, FP, FN, delta_thf,
#                                                           thc, time_limit, 5)
#
# print(len(pattern_with_low_fairness1))
# print("time = {} s, num_calculation = {}".format(t1_, num_calculation1), "\n", pattern_with_low_fairness1)
#
# pattern_with_low_fairness2, num_calculation2, t2_ = newalggeneral.GraphTraverse(less_attribute_data,
#                                                           TP, TN, FP, FN, delta_thf,
#                                                           thc, time_limit, 5)
#
# print(len(pattern_with_low_fairness2))
# print("time = {} s, num_calculation = {}".format(t2_, num_calculation2), "\n", pattern_with_low_fairness2)
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
