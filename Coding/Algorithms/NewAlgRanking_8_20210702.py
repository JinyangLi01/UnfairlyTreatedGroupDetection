"""
New algorithm for minority group detection in general case
Search the graph top-down, generate children using the method in coverage paper to avoid redundancy.
Stop point 1: when finding a pattern satisfying the requirements
Stop point 2: when the cardinality is too small
"""


"""
Go top-down, find two result sets: for lower bound and for upper bound
For lower bound: most general pattern
For upper bound: most specific pattern
We don't use patterns_size_topk[st] to store the sizes
We compute the size every time of k
"""

import pandas as pd

from itertools import combinations
from Algorithms import pattern_count
import time
from Algorithms import Predict_0_20210127 as predict
from Algorithms import NaiveAlgRanking_2_20210701 as naiveranking


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
    length = len(P)
    if len(m) != length:
        return False
    for i in range(length):
        if m[i] != P[i]:
            return False
    return True

# increase size by 1 for children, and children's children
def AddSizeTopKMinOfChildren(children, patterns_top_kmin, patterns_size_topk, new_tuple, num_patterns_visited):
    while len(children) > 0:
        child = children.pop(0)
        num_patterns_visited += 1
        st = num2string(child)
        if st in patterns_size_topk:
            patterns_size_topk[st] += 1
        else:
            patterns_size_topk[st] = patterns_top_kmin.pattern_count(st) + 1
        new_children = GenerateChildrenRelatedToTuple(child, new_tuple)
        children = children + new_children
    return num_patterns_visited

def GenerateChildrenRelatedToTuple(P, new_tuple):
    children = []
    length = len(P)
    i = 0
    for i in range(length - 1, -1, -1):
        if P[i] != -1:
            break
    if P[i] == -1:
        i -= 1
    for j in range(i + 1, length, 1):
        s = P.copy()
        s[j] = new_tuple[j]
        children.append(s)
    return children


def GenerateChildren(P, whole_data_frame, attributes):
    children = []
    length = len(P)
    i = 0
    for i in range(length - 1, -1, -1):
        if P[i] != -1:
            break
    if P[i] == -1:
        i -= 1
    for j in range(i + 1, length, 1):
        for a in range(int(whole_data_frame[attributes[j]]['min']), int(whole_data_frame[attributes[j]]['max']) + 1):
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

def findParent(child, length):
    parent = child.copy()
    for i in range(length-1, -1, -1):
        if parent[i] != -1:
            parent[i] = -1
            break
    return parent


def findParentForStr(child):
    end = 0
    start = 0
    length = len(child)
    i = length - 1
    while i > -1:
        if child[i] != '|':
            end = i + 1
            i -= 1
            break
        i -= 1
    while i > -1:
        if child[i] == '|':
            start = i
            parent = child[:start+1] + child[end:]
            return parent
        i -= 1
    parent = child[end:]
    return parent

def GraphTraverse(ranked_data, attributes, Thc, Lowerbounds, Upperbounds, k_min, k_max, time_limit):
    # print("attributes:", attributes)
    time0 = time.time()

    pc_whole_data = pattern_count.PatternCounter(ranked_data, encoded=False)
    pc_whole_data.parse_data()

    whole_data_frame = ranked_data.describe(include='all')

    num_patterns_visited = 0
    num_att = len(attributes)
    root = [-1] * num_att
    root_str = '|' * (num_att-1)
    S = GenerateChildren(root, whole_data_frame, attributes)
    pattern_treated_unfairly_lowerbound = []
    pattern_treated_unfairly_upperbound = []
    patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k_min], encoded=False)
    patterns_top_kmin.parse_data()
    patterns_size_whole = dict()
    k = k_min
    patterns_searched_lowest_level_lowerbound = set()
    patterns_searched_lowest_level_upperbound = set()

    parent_candidate_for_upperbound = []


    # DFS
    # this part is the main time consumption
    while len(S) > 0:
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        P = S.pop(0)
        # if PatternEqual(P, [-1, -1, 1, -1]):
        #     print("k={}, pattern equal = {}".format(k, P))
        st = num2string(P)
        num_patterns_visited += 1
        whole_cardinality = pc_whole_data.pattern_count(st)
        patterns_size_whole[st] = whole_cardinality
        if whole_cardinality < Thc:
            if len(parent_candidate_for_upperbound) > 0:
                CheckDominationAndAddForUpperbound(parent_candidate_for_upperbound, pattern_treated_unfairly_upperbound)
                parent_candidate_for_upperbound = []
            parent = findParent(P, num_att)
            # patterns in patterns_searched_lowest_level all have valid whole cardinality
            # and are not in pattern_treated_unfairly
            # ================== time consuming =============
            if PatternEqual(parent, root) is False:
                parent_str = num2string(parent)
                patterns_searched_lowest_level_lowerbound.add(parent_str)
                patterns_searched_lowest_level_upperbound.add(parent_str)
            continue
        num_top_k = patterns_top_kmin.pattern_count(st)
        if num_top_k < Lowerbounds[k - k_min]:
            parent = findParent(P, num_att)
            parent_str = num2string(parent)
            if parent_str != root_str:
                patterns_searched_lowest_level_lowerbound.add(parent_str)
            CheckDominationAndAddForLowerbound(P, pattern_treated_unfairly_lowerbound)
        else:
            children = GenerateChildren(P, whole_data_frame, attributes)
            if len(children) == 0:
                patterns_searched_lowest_level_lowerbound.add(st)
            S = children + S
        if num_top_k > Upperbounds[k - k_min]:
            parent_candidate_for_upperbound = P
            children = GenerateChildren(P, whole_data_frame, attributes)
            S = children + S
            if len(children) == 0:
                CheckDominationAndAddForUpperbound(P, pattern_treated_unfairly_upperbound)
                parent_candidate_for_upperbound = []
        else:
            if len(parent_candidate_for_upperbound) > 0:
                CheckDominationAndAddForUpperbound(parent_candidate_for_upperbound, pattern_treated_unfairly_upperbound)
                parent_candidate_for_upperbound = []

    for k in range(k_min + 1, k_max):
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        patterns_top_k = pattern_count.PatternCounter(ranked_data[:k], encoded=False)
        patterns_top_k.parse_data()
        new_tuple = ranked_data.iloc[[k - 1]].values.flatten().tolist()
        # top down for related patterns
        ancestors, num_patterns_visited = AddNewTuple(new_tuple, Thc, pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound,
                                whole_data_frame, patterns_top_k, k, k_min, pc_whole_data, num_patterns_visited,
                    patterns_size_whole, Lowerbounds, Upperbounds, num_att, attributes)
        # suppose Lowerbounds and Upperbounds monotonically increases
        if Lowerbounds[k-k_min] > Lowerbounds[k-1-k_min] or Upperbounds[k-k_min] > Upperbounds[k-1-k_min]:
            num_patterns_visited, patterns_searched_lowest_level_lowerbound, patterns_searched_lowest_level_upperbound \
                = CheckCandidatesForBounds(ancestors, patterns_searched_lowest_level_lowerbound,
                                                            patterns_searched_lowest_level_upperbound, root, root_str,
                                                            pattern_treated_unfairly_lowerbound,
                                                            pattern_treated_unfairly_upperbound, k,
                                                            k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                                                            Lowerbounds, Upperbounds, num_att, whole_data_frame,
                                                            attributes, num_patterns_visited, Thc)

    time1 = time.time()
    return pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound, num_patterns_visited, time1 - time0

def CheckRepeatingAndAppend(pattern, pattern_lowest_level):
    for p in pattern_lowest_level:
        if PatternEqual(p, pattern):
            return
    pattern_lowest_level.append(pattern)


def CheckDominationAndAddForLowerbound(pattern, pattern_treated_unfairly):
    to_remove = []
    for p in pattern_treated_unfairly:
        # if PatternEqual(p, pattern):
        #     return
        if P1DominatedByP2(pattern, p):
            return
        elif P1DominatedByP2(p, pattern):
            to_remove.append(p)
    for p in to_remove:
        pattern_treated_unfairly.remove(p)
    pattern_treated_unfairly.append(pattern)

def CheckDominationAndAddForUpperbound(pattern, pattern_treated_unfairly):
    to_remove = []
    for p in pattern_treated_unfairly:
        if PatternEqual(p, pattern):
            return
        if P1DominatedByP2(pattern, p):
            to_remove.append(p)
        elif P1DominatedByP2(p, pattern):
            return
    for p in to_remove:
        pattern_treated_unfairly.remove(p)
    pattern_treated_unfairly.append(pattern)


# only need to check the lower bound of parents
def CheckCandidatesForBounds(ancestors, patterns_searched_lowest_level_lowerbound,
                                patterns_searched_lowest_level_upperbound, root, root_str,
                                pattern_treated_unfairly_lowerbound,
                                pattern_treated_unfairly_upperbound, k,
                                k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                                Lowerbounds, Upperbounds, num_att, whole_data_frame,
                                attributes, num_patterns_visited, Thc):
    to_remove = set()
    to_append = set()
    for st in patterns_searched_lowest_level_lowerbound: # st is a string
        num_patterns_visited += 1
        p = string2num(st)
        if p in ancestors or p in pattern_treated_unfairly_lowerbound:
            continue
        if st in patterns_size_whole:
            whole_cardinality = patterns_size_whole[st]
        else:
            whole_cardinality = pc_whole_data.pattern_count(st)
        if whole_cardinality < Thc:
            continue
        pattern_size_in_topk = patterns_top_k.pattern_count(st)
        if pattern_size_in_topk >= Lowerbounds[k - k_min]:
            continue

        child_str = st
        parent_str = findParentForStr(child_str)
        child = string2num(child_str)
        if parent_str == root_str:
            CheckDominationAndAddForLowerbound(child, pattern_treated_unfairly_lowerbound)
            to_remove.add(child_str)
            continue

        while parent_str != root_str:
            num_patterns_visited += 1
            pattern_size_in_topk = patterns_top_k.pattern_count(parent_str)

            if pattern_size_in_topk < Lowerbounds[k - k_min]:
                child_str = parent_str
                parent_str = findParentForStr(child_str)
            else:
                CheckDominationAndAddForLowerbound(child, pattern_treated_unfairly_lowerbound)
                to_remove.add(st)
                to_append.add(parent_str)
                break
        if parent_str == root_str:
            CheckDominationAndAddForLowerbound(child, pattern_treated_unfairly_lowerbound)
            continue
    for p_str in to_remove:
        patterns_searched_lowest_level_lowerbound.remove(p_str)
    patterns_searched_lowest_level_lowerbound = patterns_searched_lowest_level_lowerbound | to_append

    # to_remove = set()
    # to_append = set()
    # for st in patterns_searched_lowest_level_upperbound:
    #     p = string2num(st) #
    #     num_patterns_visited += 1
    #     if p in ancestors or p in pattern_treated_unfairly_upperbound:
    #         continue
    #     if st in patterns_size_whole:
    #         whole_cardinality = patterns_size_whole[st]
    #     else:
    #         whole_cardinality = pc_whole_data.pattern_count(st)
    #     if whole_cardinality < Thc:
    #         continue
    #     if st in patterns_size_topk:
    #         pattern_size_in_topk = patterns_size_topk[st]
    #     else:
    #         pattern_size_in_topk = patterns_top_kmin.pattern_count(st)
    #         patterns_size_topk[st] = pattern_size_in_topk
    #     if pattern_size_in_topk <= Upperbounds[k - k_min]:
    #         continue
    #
    #     parent_str = st
    #     to_remove.add(parent_str)
    #     children = GenerateChildren(p, whole_data_frame, attributes)
    #     parent = []
    #     if len(children) == 0:
    #         parent = string2num(parent_str)
    #         CheckDominationAndAddForUpperbound(parent, pattern_treated_unfairly_upperbound)
    #     add_for_upper_bound = False
    #     while len(children) != 0:
    #         child = children.pop(0)
    #         num_patterns_visited += 1
    #         child_str = num2string(child)
    #         if child_str in patterns_size_whole:
    #             whole_cardinality = patterns_size_whole[child_str]
    #         else:
    #             whole_cardinality = pc_whole_data.pattern_count(child_str)
    #         if whole_cardinality < Thc:
    #             continue
    #         if child_str in patterns_size_topk:
    #             pattern_size_in_topk = patterns_size_topk[child_str]
    #         else:
    #             pattern_size_in_topk = patterns_top_kmin.pattern_count(child_str)
    #             patterns_size_topk[child_str] = pattern_size_in_topk
    #         if pattern_size_in_topk > Upperbounds[k - k_min]:
    #             parent = child
    #             children_new = GenerateChildren(parent, whole_data_frame, attributes)
    #             children = children_new + children
    #             if len(children_new) == 0:
    #                 add_for_upper_bound = True
    #                 CheckDominationAndAddForUpperbound(parent, pattern_treated_unfairly_upperbound)
    #                 parent = []
    #         else:
    #             if len(parent) > 0:
    #                 add_for_upper_bound = True
    #                 CheckDominationAndAddForUpperbound(parent, pattern_treated_unfairly_upperbound)
    #                 to_append.add(child_str)
    #                 parent = []
    #     if not add_for_upper_bound:
    #         CheckDominationAndAddForUpperbound(p, pattern_treated_unfairly_upperbound)
    # for p_str in to_remove:
    #     patterns_searched_lowest_level_upperbound.remove(p_str)
    # patterns_searched_lowest_level_upperbound = patterns_searched_lowest_level_upperbound | to_append

    return num_patterns_visited, patterns_searched_lowest_level_lowerbound, patterns_searched_lowest_level_upperbound


# search top-down
def AddNewTuple(new_tuple, Thc, pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound,
                                whole_data_frame, patterns_top_k, k, k_min, pc_whole_data, num_patterns_visited,
                    patterns_size_whole, Lowerbounds, Upperbounds, num_att, attributes):

    ancestors = []
    root = [-1] * num_att
    children = GenerateChildrenRelatedToTuple(root, new_tuple)
    S = children
    parent_candidate_for_upperbound = []
    while len(S) > 0:

        P = S.pop(0)
        st = num2string(P)
        num_patterns_visited += 1
        add_children = False
        children = GenerateChildrenRelatedToTuple(P, new_tuple)
        if st in patterns_size_whole:
            whole_cardinality = patterns_size_whole[st]
        else:
            whole_cardinality = pc_whole_data.pattern_count(st)

        if whole_cardinality < Thc:
            if len(parent_candidate_for_upperbound) > 0:
                CheckDominationAndAddForUpperbound(parent_candidate_for_upperbound, pattern_treated_unfairly_upperbound)
                parent_candidate_for_upperbound = []
        else:
            num_top_k = patterns_top_k.pattern_count(st)
            if num_top_k < Lowerbounds[k - k_min]:
                CheckDominationAndAddForLowerbound(new_tuple, pattern_treated_unfairly_lowerbound)
            else:
                S = children + S
                ancestors = ancestors + children
                add_children = True
            if num_top_k > Upperbounds[k - k_min]:
                parent_candidate_for_upperbound = P
                if not add_children:
                    S = children + S
                    ancestors = ancestors + children
                if len(children) == 0:
                    CheckDominationAndAddForUpperbound(P, pattern_treated_unfairly_upperbound)
                    parent_candidate_for_upperbound = []
            else: # below the upper bound
                if len(parent_candidate_for_upperbound) > 0:
                    CheckDominationAndAddForUpperbound(parent_candidate_for_upperbound, pattern_treated_unfairly_upperbound)
                    parent_candidate_for_upperbound = []

    return ancestors, num_patterns_visited


#
# all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
#                   'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C',
#                   'failures_C', 'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C', 'higher_C',
#                   'internet_C', 'romantic_C', 'famrel_C', 'freetime_C', 'goout_C', 'Dalc_C', 'Walc_C',
#                   'health_C', 'absences_C', 'G1_C', 'G2_C', 'G3_C']
#
# selected_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C',
#                        'Pstatus_C', 'Medu_C', 'Fedu_C', 'Mjob_C', 'Fjob_C',
#                        'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C', 'failures_C',
#                        'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C']
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
# k_min = 10
# k_max = 50
# Thc = 50
#
# List_k = list(range(k_min, k_max))
#
# def lowerbound(x):
#     return 5 # int((x-3)/4)
#
# def upperbound(x):
#     return 25 # int(3+(x-k_min+1)/3)
#
# Lowerbounds = [lowerbound(x) for x in List_k]
# Upperbounds = [upperbound(x) for x in List_k]
#
# print(Lowerbounds, "\n", Upperbounds)
#
#
#
#
#
#
# print("start the new alg")
#
# pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound, num_patterns_visited, running_time = \
#     GraphTraverse(ranked_data, selected_attributes, Thc,
#                                                                      Lowerbounds, Upperbounds,
#                                                                      k_min, k_max, time_limit)
#
# print("num_patterns_visited = {}".format(num_patterns_visited))
# print("time = {} s, num of pattern_treated_unfairly_lowerbound = {}, num of pattern_treated_unfairly_upperbound = {} ".format(running_time,
#         len(pattern_treated_unfairly_lowerbound), len(pattern_treated_unfairly_upperbound)), "\n", "patterns:\n",
#       pattern_treated_unfairly_lowerbound, "\n", pattern_treated_unfairly_upperbound)
#
# print("dominated by pattern_treated_unfairly_lowerbound:")
# for p in pattern_treated_unfairly_lowerbound:
#     if PDominatedByM(p, pattern_treated_unfairly_lowerbound)[0]:
#         print(p)
#
#
#
#
#
# print("start the naive alg")
#
# pattern_treated_unfairly_lowerbound2, pattern_treated_unfairly_upperbound2, \
# num_patterns_visited2, running_time2 = naiveranking.NaiveAlg(ranked_data, selected_attributes, Thc,
#                                                                      Lowerbounds, Upperbounds,
#                                                                      k_min, k_max, time_limit)
#
#
# print("num_patterns_visited = {}".format(num_patterns_visited2))
# print("time = {} s, num of pattern_treated_unfairly_lowerbound = {}, num of pattern_treated_unfairly_upperbound = {} ".format(running_time2,
#         len(pattern_treated_unfairly_lowerbound2), len(pattern_treated_unfairly_upperbound2)), "\n", "patterns:\n",
#       pattern_treated_unfairly_lowerbound2, "\n", pattern_treated_unfairly_upperbound2)
#
#
#
#
#
# print("dominated by pattern_treated_unfairly2:")
# for p in pattern_treated_unfairly_lowerbound2:
#     t, m = PDominatedByM(p, pattern_treated_unfairly_lowerbound2)
#     if t:
#         print("{} dominated by {}".format(p, m))
#
#
#
# print("p in pattern_treated_unfairly_lowerbound but not in pattern_treated_unfairly_lowerbound2:")
# for p in pattern_treated_unfairly_lowerbound:
#     if p not in pattern_treated_unfairly_lowerbound2:
#         print(p)
#
#
# print("\n\n\n")
#
# print("p in pattern_treated_unfairly_lowerbound2 but not in pattern_treated_unfairly_lowerbound:")
# for p in pattern_treated_unfairly_lowerbound2:
#     if p not in pattern_treated_unfairly_lowerbound:
#         print(p)
#
#
# print("\n\n\n")
#
# print("p in pattern_treated_unfairly_upperbound but not in pattern_treated_unfairly_upperbound2:")
# for p in pattern_treated_unfairly_upperbound:
#     if p not in pattern_treated_unfairly_upperbound2:
#         print(p)
#
#
# print("\n\n\n")
#
# print("p in pattern_treated_unfairly_upperbound2 but not in pattern_treated_unfairly_upperbound:")
# for p in pattern_treated_unfairly_upperbound2:
#     if p not in pattern_treated_unfairly_upperbound:
#         print(p)
#

