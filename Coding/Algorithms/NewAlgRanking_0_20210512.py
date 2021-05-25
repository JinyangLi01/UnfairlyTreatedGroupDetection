"""
New algorithm for minority group detection in general case
Search the graph top-down, generate children using the method in coverage paper to avoid redundancy.
Stop point 1: when finding a pattern satisfying the requirements
Stop point 2: when the cardinality is too small
"""

from Algorithms import pattern_count
import time
from Algorithms import Predict_0_20210127 as predict


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



def num2string(pattern):
    st = ''
    for i in pattern:
        if i != -1:
            st += str(i)
        st += '|'
    st = st[:-1]
    return st


# whether a pattern P is dominated by MUP M
# except from P itself
def PDominatedByM(P, M):
    for m in M:
        if PatternEqual(m, P):
            continue
        if P1DominatedByP2(P, m):
            return True, m
    return False, None


"""
Predictive parity:The fraction of correct positive prediction 
TP/(TP+FP) should be similar for all groups
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

    original_thf = len(TPdata) / (len(TPdata) + len(FPdata))
    Thf = original_thf - delta_Thf
    print("Predictive_parity, original_thf = {}, Thf = {}".format(original_thf, Thf))

    num_calculation = 0
    root = [-1] * (len(attributes))
    initial_children = GenerateChildren(root, whole_data_frame, attributes)
    S = initial_children
    pattern_with_low_fairness = []

    while len(S) > 0:
        if time.time() - time1 > time_limit:
            print("newalg overtime")
            break
        P = S.pop()
        st = num2string(P)

        whole_cardinality = pc_whole_data.pattern_count(st)
        num_calculation += 1
        if whole_cardinality < Thc:
            # pattern_skipped_whole_c.append(P)
            continue

        # time consuming!!
        tp = pc_TP.pattern_count(st)
        num_calculation += 1
        if tp == 0:
            continue
        fp = pc_TP.pattern_count(st)
        correct_positive_prediction = tp / (tp + fp)
        num_calculation += 1

        if correct_positive_prediction >= Thf:
            children = GenerateChildren(P, whole_data_frame, attributes)
            S = S + children
            continue
        # why this line?
        if PDominatedByM(P, pattern_with_low_fairness)[0] is False:
            pattern_with_low_fairness.append(P)
    time2 = time.time()
    # print(duration1, duration2, duration3, duration4, duration5, duration6)
    return pattern_with_low_fairness, num_calculation, time2 - time1


"""
False positive error rate balance (predictive equality)
The probability of a subject in the actual negative class to have a positive
predictive value FPR = FP/(FP+TN) is similar for all groups.
"""
def False_positive_error_rate_balance(whole_data, FPdata, TNdata,
                                      delta_Thf, Thc, time_limit):
    time1 = time.time()

    pc_whole_data = pattern_count.PatternCounter(whole_data, encoded=False)
    pc_whole_data.parse_data()
    pc_FP = pattern_count.PatternCounter(FPdata, encoded=False)
    pc_FP.parse_data()
    pc_TN = pattern_count.PatternCounter(TNdata, encoded=False)
    pc_TN.parse_data()

    whole_data_frame = whole_data.describe()
    attributes = whole_data_frame.columns.values.tolist()

    original_thf = len(FPdata) / (len(FPdata) + len(TNdata))
    Thf = original_thf - delta_Thf
    print("False_positive_error_rate_balance, original_thf = {}, Thf = {}".format(original_thf, Thf))

    num_calculation = 0
    root = [-1] * (len(attributes))
    initial_children = GenerateChildren(root, whole_data_frame, attributes)
    S = initial_children
    pattern_with_low_fairness = []

    while len(S) > 0:
        if time.time() - time1 > time_limit:
            print("newalg overtime")
            break
        P = S.pop()
        st = num2string(P)

        whole_cardinality = pc_whole_data.pattern_count(st)
        num_calculation += 1
        if whole_cardinality < Thc:
            # pattern_skipped_whole_c.append(P)
            continue

        # time consuming!!
        fp = pc_FP.pattern_count(st)
        num_calculation += 1
        if fp == 0:
            continue
        tn = pc_TN.pattern_count(st)
        FPR = fp / (fp + tn)
        num_calculation += 1

        if FPR >= Thf:
            children = GenerateChildren(P, whole_data_frame, attributes)
            S = S + children
            continue
        # why this line?
        if PDominatedByM(P, pattern_with_low_fairness)[0] is False:
            pattern_with_low_fairness.append(P)
    time2 = time.time()
    # print(duration1, duration2, duration3, duration4, duration5, duration6)
    return pattern_with_low_fairness, num_calculation, time2 - time1


"""
False negative error rate balance (equal opportunity) 
Similar to the above, but considers the probability of falsely classifying
subject in the positive class as negative FNR = FN/(TP+FN)
"""
def False_negative_error_rate_balance(whole_data, TPdata, FNdata,
                                      delta_Thf, Thc, time_limit):
    time1 = time.time()

    pc_whole_data = pattern_count.PatternCounter(whole_data, encoded=False)
    pc_whole_data.parse_data()
    pc_FN = pattern_count.PatternCounter(FNdata, encoded=False)
    pc_FN.parse_data()
    pc_TP = pattern_count.PatternCounter(TPdata, encoded=False)
    pc_TP.parse_data()

    original_thf = len(FNdata) / (len(TPdata) + len(FNdata))
    Thf = original_thf + delta_Thf
    print("False_negative_error_rate_balance, original_thf = {}, Thf = {}".format(original_thf, Thf))

    whole_data_frame = whole_data.describe()
    attributes = whole_data_frame.columns.values.tolist()

    num_calculation = 0
    root = [-1] * (len(attributes))
    initial_children = GenerateChildren(root, whole_data_frame, attributes)
    S = initial_children
    pattern_with_low_fairness = []

    while len(S) > 0:
        if time.time() - time1 > time_limit:
            print("newalg overtime")
            break
        P = S.pop()
        st = num2string(P)

        whole_cardinality = pc_whole_data.pattern_count(st)
        num_calculation += 1
        if whole_cardinality < Thc:
            # pattern_skipped_whole_c.append(P)
            continue

        # time consuming!!
        fn = pc_FN.pattern_count(st)
        num_calculation += 1
        if fn == 0:
            continue
        tp = pc_TP.pattern_count(st)
        FNR = fn / (fn + tp)
        num_calculation += 1

        if FNR <= Thf:
            children = GenerateChildren(P, whole_data_frame, attributes)
            S = S + children
            continue
        # why this line?
        if PDominatedByM(P, pattern_with_low_fairness)[0] is False:
            pattern_with_low_fairness.append(P)
    time2 = time.time()
    # print(duration1, duration2, duration3, duration4, duration5, duration6)
    return pattern_with_low_fairness, num_calculation, time2 - time1


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

    original_thf_FPR = len(FPdata) / (len(FPdata) + len(TNdata))
    original_thf_FNR = len(FNdata) / (len(TPdata) + len(FNdata))
    Thf_FPR = original_thf_FPR - delta_Thf_FPR
    Thf_FNR = original_thf_FNR + delta_Thf_FNR
    print("Equalized_odds, original_thf_FPR = {}, Thf_FPR = {}".format(original_thf_FPR, Thf_FPR))
    print("original_thf_FNR = {}, Thf_FNR = {}".format(original_thf_FNR, Thf_FNR))

    num_calculation = 0
    root = [-1] * (len(attributes))
    initial_children = GenerateChildren(root, whole_data_frame, attributes)
    S = initial_children
    pattern_with_low_fairness = []

    while len(S) > 0:
        if time.time() - time1 > time_limit:
            print("newalg overtime")
            break
        P = S.pop()
        st = num2string(P)

        whole_cardinality = pc_whole_data.pattern_count(st)
        num_calculation += 1
        if whole_cardinality < Thc:
            # pattern_skipped_whole_c.append(P)
            continue

        # time consuming!!
        fp = pc_FP.pattern_count(st)
        num_calculation += 1
        fn = pc_FN.pattern_count(st)
        num_calculation += 1

        tp = pc_TP.pattern_count(st)
        num_calculation += 1
        if fn + tp == 0:
            continue
        FNR = fn / (fn + tp)
        tn = pc_TN.pattern_count(st)
        num_calculation += 1
        if fp + tn == 0:
            continue
        FPR = fp / (fp + tn)

        if FNR <= Thf_FNR and FPR >= Thf_FPR:
            children = GenerateChildren(P, whole_data_frame, attributes)
            S = S + children
            continue
        # why this line?
        if PDominatedByM(P, pattern_with_low_fairness)[0] is False:
            pattern_with_low_fairness.append(P)
    time2 = time.time()
    # print(duration1, duration2, duration3, duration4, duration5, duration6)
    return pattern_with_low_fairness, num_calculation, time2 - time1


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

    num_calculation = 0
    root = [-1] * (len(attributes))
    initial_children = GenerateChildren(root, whole_data_frame, attributes)
    S = initial_children
    pattern_with_low_fairness = []

    while len(S) > 0:
        if time.time() - time1 > time_limit:
            print("newalg overtime")
            break
        P = S.pop()
        st = num2string(P)

        whole_cardinality = pc_whole_data.pattern_count(st)
        num_calculation += 1
        if whole_cardinality < Thc:
            # pattern_skipped_whole_c.append(P)
            continue

        # time consuming!!
        tp = pc_TP.pattern_count(st)
        num_calculation += 1
        tn = pc_TN.pattern_count(st)
        num_calculation += 1
        fp = pc_FP.pattern_count(st)
        num_calculation += 1

        if tp + fp == 0:
            continue
        positive_prob = tp / (tp + fp)

        fn = pc_FN.pattern_count(st)
        num_calculation += 1
        if tn + fn == 0:
            continue
        negative_prob = tn / (tn + fn)

        if positive_prob >= Thf_FP and negative_prob <= Thf_FN:
            children = GenerateChildren(P, whole_data_frame, attributes)
            S = S + children
            continue
        # why this line?
        if PDominatedByM(P, pattern_with_low_fairness)[0] is False:
            pattern_with_low_fairness.append(P)
    time2 = time.time()
    # print(duration1, duration2, duration3, duration4, duration5, duration6)
    return pattern_with_low_fairness, num_calculation, time2 - time1


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

    num_calculation = 0
    root = [-1] * (len(attributes))
    initial_children = GenerateChildren(root, whole_data_frame, attributes)
    S = initial_children
    pattern_with_low_fairness = []

    while len(S) > 0:
        if time.time() - time1 > time_limit:
            print("newalg overtime")
            break
        P = S.pop()
        st = num2string(P)

        whole_cardinality = pc_whole_data.pattern_count(st)
        num_calculation += 1
        if whole_cardinality < Thc:
            # pattern_skipped_whole_c.append(P)
            continue

        # time consuming!!
        fp = pc_FP.pattern_count(st)
        num_calculation += 1
        fn = pc_FN.pattern_count(st)
        num_calculation += 1

        if fn == 0:
            continue
        ratio = fp / fn

        if ratio >= Thf_ratio:
            children = GenerateChildren(P, whole_data_frame, attributes)
            S = S + children
            continue
        # why this line?
        if PDominatedByM(P, pattern_with_low_fairness)[0] is False:
            pattern_with_low_fairness.append(P)
    time2 = time.time()
    # print(duration1, duration2, duration3, duration4, duration5, duration6)
    return pattern_with_low_fairness, num_calculation, time2 - time1


"""
whole_data: the original data file 
mis_class_data: file containing mis-classified tuples
Tha: threshold of accuracy 
Thc: threshold of cardinality
"""
def GraphTraverse(whole_data, TPdata, TNdata, FPdata, FNdata,
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





# # age,workclass,education,educational-num,marital-status
# selected_attributes = ['age', 'workclass', 'education', 'educational-num', 'marital-status']
# # original_data_file = "../../InputData/AdultDataset/SmallDataset/SmallWhole_5_10.csv"
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
# pattern_with_low_fairness, num_calculation, t_ = GraphTraverse(less_attribute_data,
#                                                               TP, TN, FP, FN, delta_thf,
#                                                               thc, time_limit, 2)
#
# print(len(pattern_with_low_fairness))
# print("time = {} s, num_calculation = {}".format(t_, num_calculation), "\n", pattern_with_low_fairness)
#
# for p in pattern_with_low_fairness:
#     if PDominatedByM(p, pattern_with_low_fairness)[0]:
#         print(p)
