"""
New algorithm for group detection in ranking
fairness definition: the number of a group members in top-k is bounded by U_k, L_k, k_min <= k <= k_max
bounds for different k can be different, but all patterns have the same bounds

this alg has some problem, not the final used version
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
whole_data: the original data file 
mis_class_data: file containing mis-classified tuples
Tha: delta fairness value 
Thc: size threshold
"""
def GraphTraverse(whole_data, ranked_data, Thc, Lowerbounds, Upperbounds, k_min, k_max, time_limit):
    time1 = time.time()

    pc_whole_data = pattern_count.PatternCounter(ranked_data, encoded=False)
    pc_whole_data.parse_data()

    whole_data_frame = whole_data.describe()
    attributes = whole_data_frame.columns.values.tolist()

    num_patterns = 0
    root = [-1] * (len(attributes))
    S = [root]
    pattern_treated_unfairly = []
    patterns_top_k = []
    for k in range(k_min, k_max + 1):
        patterns_top_k[k-k_min] = pattern_count.PatternCounter(ranked_data[:k], encoded=False)

    while len(S) > 0:
        if time.time() - time1 > time_limit:
            print("newalg overtime")
            break
        P = S.pop()
        st = num2string(P)

        num_patterns += 1

        whole_cardinality = pc_whole_data.pattern_count(st)
        if whole_cardinality < Thc:
            continue

        treated_unfairly = False
        for k in range(k_min, k_max+1):
            num_top_k = patterns_top_k[k-k_min].pattern_count(st)
            if num_top_k < Lowerbounds[k-k_min] or num_top_k > Upperbounds[k-k_min]:
                if PDominatedByM(P, pattern_treated_unfairly)[0] is False:
                    pattern_treated_unfairly.append(P)
                    treated_unfairly = True
                    break

        if not treated_unfairly:
            children = GenerateChildren(P, whole_data_frame, attributes)
            S = S + children
            continue
    time2 = time.time()
    # print(duration1, duration2, duration3, duration4, duration5, duration6)
    return pattern_treated_unfairly, num_patterns, time2-time1





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
