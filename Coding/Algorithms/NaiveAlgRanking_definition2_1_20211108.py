"""
Naive algorithm for group detection in ranking
fairness definition: the number of a group members in top-k should be proportional to the group size, k_min <= k <= k_max

Expected output: most general patterns treated unfairly, either lower bound, or upper bound
Difference from definition 1: in definition 1, we find most specific patterns for upper bound,
but here we check upper bound together with lower bound, and don't do the part for upper bound.

naive alg:
same as NaiveAlgRanking_2_20200701.py
for each k, we iterate the whole process again, go top down
"""


"""
Go top-down, find two result sets: for lower bound and for upper bound
For lower bound: most general pattern
For upper bound: most specific pattern

Use alg from classification for k_min
and then, for k>k_min, only search locally.
When k increases by 1, there is only one tuple added into top-k
There are two groups of patterns we need to check:
1. patterns related to this additional tuple, starting from root
2. if bounds change, check parents of patterns in the result set for lower bound, upper bound are not affected

We don't use patterns_size_topk[st] to store the sizes, but still use patterns_size_whole
We compute the size every time of k
"""

import time
import pandas as pd
import numpy as np

from Algorithms import pattern_count


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
            # print(P, "domintated by", m)
            return True, m
    return False, None

def PDominatesM(P, M):
    for m in M:
        if PatternEqual(m, P):
            continue
        if P1DominatedByP2(m, P):
            return True, m
    return False, None


def GenerateChildren(P, whole_data_frame, ranked_data, attributes):
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
            # print(ranked_data.loc[3, attributes[j]], type(ranked_data.loc[3, attributes[j]]))
            if not isinstance(ranked_data.loc[3, attributes[j]], (int, np.integer)):
            #if type(whole_data_frame[attributes[j]]['min']) is not int:
                s[j] = float(a)
            children.append(s)
    return children


def CheckDominationAndAdd(pattern, pattern_treated_unfairly):
    to_remove = []
    for p in pattern_treated_unfairly:
        if PatternEqual(p, pattern):
            return
        if P1DominatedByP2(pattern, p):
            return
        elif P1DominatedByP2(p, pattern):
            to_remove.append(p)
    for p in to_remove:
        pattern_treated_unfairly.remove(p)
    pattern_treated_unfairly.append(pattern)




def string2num(st):
    p = list()
    idx = 0
    item = ''
    i = ''
    for i in st:
        if i == '|':
            if item == '':
                p.append(-1)
            else:
                p.append(int(item))
                item = ''
            idx += 1
        else:
            item += i
    if i != '|':
        p.append(int(item))
    else:
        p.append(-1)
    return p


def PatternInSet(p, set):
    if isinstance(p, str):
        p = string2num(p)
    for q in set:
        if PatternEqual(p, q):
            return True
    return False


"""
whole_data: the original data file 
mis_class_data: file containing mis-classified tuples
Tha: delta fairness value 
Thc: size threshold
"""

def NaiveAlg(ranked_data, attributes, Thc, alpha, k_min, k_max, time_limit):
    time0 = time.time()
    data_size = len(ranked_data)
    pc_whole_data = pattern_count.PatternCounter(ranked_data, encoded=False)
    pc_whole_data.parse_data()
    whole_data_frame = ranked_data.describe(include='all')
    num_patterns_visited = 0
    pattern_treated_unfairly = []
    overtime_flag = False

    q = [-1, -1, -1, -1, -1, 0, -1, -1, -1]
    for k in range(k_min, k_max):
        # print("k={}".format(k))
        # if q in pattern_treated_unfairly:
        #     print("q in!!")
        # else:
        #     print("q not in ... ")
        if overtime_flag:
            print("naive overtime, exiting the loop of k")
            break
        root = [-1] * (len(attributes))
        S = GenerateChildren(root, whole_data_frame, ranked_data, attributes)
        patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k], encoded=False)
        patterns_top_kmin.parse_data()

        # lower bound
        while len(S) > 0:
            if time.time() - time0 > time_limit:
                overtime_flag = True
                print("naive overtime")
                break
            P = S.pop(0)
            st = num2string(P)
            # if PatternEqual(P, q) and k == 33:
            #     print("st = {}\n".format(st))
            #     print("stop here naive alg")

            num_patterns_visited += 1
            whole_cardinality = pc_whole_data.pattern_count(st)
            # print("P={}, whole size={}".format(P, whole_cardinality))
            if whole_cardinality < Thc:
                continue
            num_top_k = patterns_top_kmin.pattern_count(st)
            lowerbound = (whole_cardinality / data_size - alpha) * k
            upperbound = (whole_cardinality / data_size + alpha) * k
            if num_top_k < lowerbound or num_top_k > upperbound:
                # if PatternEqual(P, [-1, -1, 1, -1]):
                #     print("k={}, pattern equal = {}, num_top_k = {}".format(k, P, num_top_k))
                CheckDominationAndAdd(P, pattern_treated_unfairly)
            else:
                children = GenerateChildren(P, whole_data_frame, ranked_data, attributes)
                S = children + S
                continue
    time1 = time.time()
    return pattern_treated_unfairly, num_patterns_visited, time1 - time0


#
# all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
#                   'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C',
#                   'failures_C', 'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C', 'higher_C',
#                   'internet_C', 'romantic_C', 'famrel_C', 'freetime_C', 'goout_C', 'Dalc_C', 'Walc_C',
#                   'health_C', 'absences_C', 'G1_C', 'G2_C', 'G3_C']
#
# selected_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C']
#
#
# """
# with the above 19 att,
# naive: 98s num_patterns_visited = 2335488
# optimized: 124s num_patterns_visited = 299559
# num of pattern_treated_unfairly_lowerbound = 85, num of pattern_treated_unfairly_upperbound = 18
# """
#
# original_data_file = r"../../InputData/StudentDataset/ForRanking_1/student-mat_cat_ranked.csv"
#
#
# ranked_data = pd.read_csv(original_data_file)
# ranked_data = ranked_data[selected_attributes]
#
#
# time_limit = 5 * 60
# k_min = 162
# k_max = 166
# Thc = 50
#
# List_k = list(range(k_min, k_max))
# #
# # def lowerbound(x):
# #     return 5 # int((x-3)/4)
# #
# # def upperbound(x):
# #     return 25 # int(3+(x-k_min+1)/3)
# #
# # Lowerbounds = [lowerbound(x) for x in List_k]
# # Upperbounds = [upperbound(x) for x in List_k]
# #
# # print(Lowerbounds, "\n", Upperbounds)
#
#
# alpha = 0.1
#
#
#
# pattern_treated_unfairly, num_patterns_visited, running_time = \
#     NaiveAlg(ranked_data, selected_attributes, Thc,
#                      alpha,
#                      k_min, k_max, time_limit)
#
# print("num_patterns_visited = {}".format(num_patterns_visited))
# print("time = {} s, num of pattern_treated_unfairly = {}".format(running_time,
#         len(pattern_treated_unfairly)), "\n", "patterns:\n",
#       "lower bound ", pattern_treated_unfairly)
#
# print("dominated by pattern_treated_unfairly:")
# for p in pattern_treated_unfairly:
#     if PDominatedByM(p, pattern_treated_unfairly)[0]:
#         print(p)
#
#
