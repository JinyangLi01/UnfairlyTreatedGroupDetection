"""
New algorithm for group detection in ranking
fairness definition: the number of a group members in top-k is bounded by U_k, L_k, k_min <= k <= k_max
bounds for different k can be different, but all patterns have the same bounds
This is the final algorithm for this fairness definition in ranking



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

This script is different from NewAlgRanking_8_20210702.py only for:
In function CheckCandidatesForBounds(), here I have checked_patterns to avoid checking same pattern twice.
But in NewAlgRanking_8_20210702.py I don't have this.
Their performance are almost the same.

"""

import pandas as pd

from itertools import combinations
from Algorithms import pattern_count
import time
from Algorithms import Predict_0_20210127 as predict
from Algorithms import NaiveAlgRanking_3_20211212 as naiveranking


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
    for i in range(length - 1, -1, -1):
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
            parent = child[:start + 1] + child[end:]
            return parent
        i -= 1
    parent = child[end:]
    return parent


def CheckRepeatingAndAppend(pattern, pattern_lowest_level):
    for p in pattern_lowest_level:
        if PatternEqual(p, pattern):
            return
    pattern_lowest_level.append(pattern)


def CheckDominationAndAddForLowerbound(pattern, pattern_treated_unfairly, dominated_by_lowerbound_result):
    to_remove = []
    for p in pattern_treated_unfairly:
        # if PatternEqual(p, pattern):
        #     return
        if P1DominatedByP2(pattern, p):
            if pattern not in dominated_by_lowerbound_result:
                dominated_by_lowerbound_result.append(pattern)
            return
        elif P1DominatedByP2(p, pattern):
            to_remove.append(p)
    for p in to_remove:
        pattern_treated_unfairly.remove(p)
        dominated_by_lowerbound_result.append(p)
    pattern_treated_unfairly.append(pattern)
    if pattern in dominated_by_lowerbound_result:
        dominated_by_lowerbound_result.remove(pattern)


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
                             result_set_lowerbound, result_set_upperbound, k,
                             k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                             Lowerbounds, Upperbounds, num_att, whole_data_frame,
                             attributes, num_patterns_visited, Thc, dominated_by_lowerbound_result):
    to_remove = set()
    to_append = set()
    checked_patterns = set()
    st = "|0||0|"
    if st in patterns_searched_lowest_level_lowerbound:
        print("in CheckCandidatesForBounds, {} in stop set".format(st))
    for st in patterns_searched_lowest_level_lowerbound:  # st is a string
        if st == "|0||0|":
            print("CheckCandidatesForBounds, st = {}".format(st))
        if st in checked_patterns:
            continue
        checked_patterns.add(st)
        num_patterns_visited += 1
        p = string2num(st)
        if PatternEqual(p, [-1, 0, -1, 0, -1]):
            print("CheckCandidatesForBounds, p = {}".format(p))
        if p in ancestors or p in result_set_lowerbound:  # already checked
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
        # if pattern_size_in_topk < Lowerbounds[k - k_min], remove child and add this parent
        # why do you only care about the parent generating this child, but not other parents?
        # because other parents will be handled by the child it generates by itself
        child_str = st
        parent_str = findParentForStr(child_str)
        child = string2num(child_str)
        if parent_str == root_str:
            CheckDominationAndAddForLowerbound(child, result_set_lowerbound, dominated_by_lowerbound_result)
            to_remove.add(child_str)  # child need removing
            continue
        checked_patterns.add(parent_str)
        # if parent is not root, we need to check until 1: the root, 2: we find a node that is above the lower bound
        # since lower bound may increase by 5, and parent is below the lower bound, grandparent may be too
        while parent_str != root_str:
            num_patterns_visited += 1
            pattern_size_in_topk = patterns_top_k.pattern_count(parent_str)
            if pattern_size_in_topk < Lowerbounds[k - k_min]:
                child_str = parent_str
                parent_str = findParentForStr(child_str)
                checked_patterns.add(parent_str)
            else:
                CheckDominationAndAddForLowerbound(child, result_set_lowerbound, dominated_by_lowerbound_result)
                to_remove.add(st)
                to_append.add(parent_str)
                break
        if parent_str == root_str:
            CheckDominationAndAddForLowerbound(child, result_set_lowerbound, dominated_by_lowerbound_result)
            continue
    for p_str in to_remove:
        patterns_searched_lowest_level_lowerbound.remove(p_str)
    patterns_searched_lowest_level_lowerbound = patterns_searched_lowest_level_lowerbound | to_append

    return num_patterns_visited, patterns_searched_lowest_level_lowerbound, \
           patterns_searched_lowest_level_upperbound, checked_patterns


def GraphTraverse(ranked_data, attributes, Thc, Lowerbounds, Upperbounds, k_min, k_max, time_limit):
    # print("attributes:", attributes)
    time0 = time.time()

    pc_whole_data = pattern_count.PatternCounter(ranked_data, encoded=False)
    pc_whole_data.parse_data()

    whole_data_frame = ranked_data.describe(include='all')

    num_patterns_visited = 0
    num_att = len(attributes)
    root = [-1] * num_att
    root_str = '|' * (num_att - 1)
    S = GenerateChildren(root, whole_data_frame, attributes)
    pattern_treated_unfairly_lowerbound = []  # looking for the most general patterns
    pattern_treated_unfairly_upperbound = []  # looking for the most specific patterns
    patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k_min], encoded=False)
    patterns_top_kmin.parse_data()
    patterns_size_whole = dict()
    k = k_min
    patterns_searched_lowest_level_lowerbound = set()
    patterns_searched_lowest_level_upperbound = set()

    parent_candidate_for_upperbound = []
    dominated_by_lowerbound_result = []

    result_set_lowerbound = []
    result_set_upperbound = []
    # DFS
    # this part is the main time consumption
    while len(S) > 0:
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        P = S.pop(0)
        if PatternEqual(P, [-1, 0, -1, -1, 1]):
            print("k={}, pattern equal = {}".format(k, P))
            print("\n")
        st = num2string(P)
        num_patterns_visited += 1
        add_children = False
        whole_cardinality = pc_whole_data.pattern_count(st)
        patterns_size_whole[st] = whole_cardinality
        if whole_cardinality < Thc:
            if len(parent_candidate_for_upperbound) > 0:  # there is a parent which is above upper bound
                CheckDominationAndAddForUpperbound(parent_candidate_for_upperbound, result_set_upperbound)
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
            CheckDominationAndAddForLowerbound(P, result_set_lowerbound, dominated_by_lowerbound_result)
        else:
            if P[num_att-1] != -1:
                patterns_searched_lowest_level_lowerbound.add(st)
            else:
                children = GenerateChildren(P, whole_data_frame, attributes)
                S = children + S
            add_children = True
        if num_top_k > Upperbounds[k - k_min]:
            parent_candidate_for_upperbound = P  # we need to store this so that if child is below upper bound, we put this into result set
            if not add_children:
                if P[num_att -1] != -1:
                    CheckDominationAndAddForUpperbound(P, result_set_upperbound)
                    parent_candidate_for_upperbound = []
                else:
                    children = GenerateChildren(P, whole_data_frame, attributes)
                    S = children + S
        else:
            if len(parent_candidate_for_upperbound) > 0:  # P is not above upperbound, so its parent should be added to the result set
                CheckDominationAndAddForUpperbound(parent_candidate_for_upperbound, result_set_upperbound)
                parent_candidate_for_upperbound = []
    pattern_treated_unfairly_upperbound.append(result_set_upperbound)
    pattern_treated_unfairly_lowerbound.append(result_set_lowerbound)

    for k in range(k_min + 1, k_max):
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        result_set_lowerbound=result_set_lowerbound.copy()
        result_set_upperbound=result_set_upperbound.copy()
        patterns_top_k = pattern_count.PatternCounter(ranked_data[:k], encoded=False)
        patterns_top_k.parse_data()
        new_tuple = ranked_data.iloc[[k - 1]].values.flatten().tolist()
        print("k={}, new tuple = {}".format(k, new_tuple))
        # top down for related patterns, using similar methods as k_min, add to result set if needed
        # ancestors are patterns checked in AddNewTuple() function, to avoid checking them again
        ancestors, num_patterns_visited = AddNewTuple(new_tuple, Thc, result_set_lowerbound, result_set_upperbound,
                                                      whole_data_frame, patterns_top_k, k, k_min, pc_whole_data,
                                                      num_patterns_visited,
                                                      patterns_size_whole, Lowerbounds, Upperbounds, num_att,
                                                      attributes, dominated_by_lowerbound_result)
        # suppose Lowerbounds and Upperbounds monotonically increases
        st = "|0||0|"
        if st in patterns_searched_lowest_level_lowerbound:
            print("after AddNewTuple, {} in stop set".format(st))
        checked_patterns = set()
        if Lowerbounds[k - k_min] > Lowerbounds[k - 1 - k_min] or Upperbounds[k - k_min] > Upperbounds[k - 1 - k_min]:
            num_patterns_visited, patterns_searched_lowest_level_lowerbound, patterns_searched_lowest_level_upperbound, \
            checked_patterns \
                = CheckCandidatesForBounds(ancestors, patterns_searched_lowest_level_lowerbound,
                                           patterns_searched_lowest_level_upperbound, root, root_str,
                                           result_set_lowerbound, result_set_upperbound, k,
                                           k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                                           Lowerbounds, Upperbounds, num_att, whole_data_frame,
                                           attributes, num_patterns_visited, Thc, dominated_by_lowerbound_result)
        # check dominated_by_lowerbound_result
        move_out = []
        move_in = []
        for p in dominated_by_lowerbound_result:
            can_add = True
            if p in ancestors:
                continue
            if PDominatedByM(p, move_in)[0]:
                continue
            for q in result_set_lowerbound:
                if P1DominatedByP2(p, q):
                    can_add = False
                    break
                elif P1DominatedByP2(q, p):
                    move_out.append(q)
            if can_add:
                remove_from_movein = []
                for t in move_in:
                    if P1DominatedByP2(t, p):
                        remove_from_movein.append(t)
                for t in remove_from_movein:
                    move_in.remove(t)
                move_in.append(p)
        for s in move_out:
            result_set_lowerbound.remove(s)
            if s not in dominated_by_lowerbound_result:
                dominated_by_lowerbound_result.append(s)
        for s in move_in:
            result_set_lowerbound.append(s)
            dominated_by_lowerbound_result.remove(s)

        pattern_treated_unfairly_upperbound.append(result_set_upperbound)
        pattern_treated_unfairly_lowerbound.append(result_set_lowerbound)
    time1 = time.time()
    return pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound, num_patterns_visited, time1 - time0


# def CheckDominationAndAdd(move_in, move_out):



# search top-down to go over all patterns related to new_tuple
# using similar checking methods as k_min
# add to result set if they are outliers
def AddNewTuple(new_tuple, Thc, result_set_lowerbound, result_set_upperbound,
                whole_data_frame, patterns_top_k, k, k_min, pc_whole_data, num_patterns_visited,
                patterns_size_whole, Lowerbounds, Upperbounds, num_att, attributes, dominated_by_lowerbound_result):
    ancestors = []
    root = [-1] * num_att
    children = GenerateChildrenRelatedToTuple(root, new_tuple)  # pattern with one deternimistic attribute
    S = children
    parent_candidate_for_upperbound = []
    add_to_lowerbound = []

    while len(S) > 0:
        P = S.pop(0)
        st = num2string(P)
        if PatternEqual(P, [-1, 0, -1, -1, 1]):
            print("AddNewTuple, P={}".format(P))
            print("\n")
        num_patterns_visited += 1
        add_children = False
        children = []
        if st in patterns_size_whole:
            whole_cardinality = patterns_size_whole[st]
        else:
            whole_cardinality = pc_whole_data.pattern_count(st)

        if whole_cardinality < Thc:
            if len(parent_candidate_for_upperbound) > 0:
                if parent_candidate_for_upperbound not in result_set_upperbound:
                    CheckDominationAndAddForUpperbound(parent_candidate_for_upperbound, result_set_upperbound)
                parent_candidate_for_upperbound = []
        else:
            num_top_k = patterns_top_k.pattern_count(st)
            if num_top_k < Lowerbounds[k - k_min]:
                if P not in result_set_lowerbound:
                    add_to_lowerbound.append(P)
                    # CheckDominationAndAddForLowerbound(P, result_set_lowerbound)
            else:
                if P in result_set_lowerbound:
                    result_set_lowerbound.remove(P)
                    children = GenerateChildren(P, whole_data_frame, attributes)
                elif P in dominated_by_lowerbound_result:
                    dominated_by_lowerbound_result.remove(P)
                    children = GenerateChildren(P, whole_data_frame, attributes)
                else:
                    if P[num_att - 1] == -1:
                        children = GenerateChildrenRelatedToTuple(P, new_tuple)
                if len(children) != 0:
                    S = children + S
                    ancestors = ancestors + children
                    add_children = True
            if num_top_k > Upperbounds[k - k_min]:
                parent_candidate_for_upperbound = P
                if not add_children and P[num_att - 1] == -1:
                    children = GenerateChildrenRelatedToTuple(P, new_tuple)
                    S = children + S
                    ancestors = ancestors + children
                if P[num_att - 1] != -1:  # no children
                    CheckDominationAndAddForUpperbound(P, result_set_upperbound)
                    parent_candidate_for_upperbound = []
            else:  # below the upper bound
                if len(parent_candidate_for_upperbound) > 0:
                    CheckDominationAndAddForUpperbound(parent_candidate_for_upperbound,
                                                       result_set_upperbound)
                    parent_candidate_for_upperbound = []
    for p in add_to_lowerbound:
        CheckDominationAndAddForLowerbound(p, result_set_lowerbound, dominated_by_lowerbound_result)
    return ancestors, num_patterns_visited

#
# all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
#                   'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C',
#                   'failures_C', 'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C', 'higher_C',
#                   'internet_C', 'romantic_C', 'famrel_C', 'freetime_C', 'goout_C', 'Dalc_C', 'Walc_C',
#                   'health_C', 'absences_C', 'G1_C', 'G2_C', 'G3_C']
#
# selected_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
#                        'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C']
#
#
#
#
# original_data_file = r"../../InputData/StudentDataset/ForRanking_1/student-mat_cat_ranked.csv"
#
# ranked_data = pd.read_csv(original_data_file)
# ranked_data = ranked_data[selected_attributes]
#
# time_limit = 10 * 60
# k_min = 5
# k_max = 65
# Thc = 50
#
# List_k = list(range(k_min, k_max))
#
#
# def lowerbound(x):
#     return 5  # int((x-3)/4)
#
#
# def upperbound(x):
#     return 25  # int(3+(x-k_min+1)/3)
#
#
# Lowerbounds = [lowerbound(x) for x in List_k]
# Upperbounds = [upperbound(x) for x in List_k]
#
# print(Lowerbounds, "\n", Upperbounds)
#
# print("start the new alg")
#
# pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound, num_patterns_visited, running_time = \
#     GraphTraverse(ranked_data, selected_attributes, Thc,
#                   Lowerbounds, Upperbounds,
#                   k_min, k_max, time_limit)
#
#
# print("num_patterns_visited = {}".format(num_patterns_visited))
# print("time = {} s".format(running_time))
# for k in range(0, k_max - k_min):
#     print("k = {}, num = {}, patterns =".format(k + k_min, len(pattern_treated_unfairly_lowerbound[k])),
#           pattern_treated_unfairly_lowerbound[k])
#
#
#
#
#
# print("start the naive alg")
#
# pattern_treated_unfairly_lowerbound2, pattern_treated_unfairly_upperbound2, \
# num_patterns_visited2, running_time2 = naiveranking.NaiveAlg(ranked_data, selected_attributes, Thc,
#                                                              Lowerbounds, Upperbounds,
#                                                              k_min, k_max, time_limit)
#
# print("num_patterns_visited = {}".format(num_patterns_visited2))
# print("time = {} s".format(running_time2))
# for k in range(0, k_max - k_min):
#     print("k = {}, num = {}, patterns =".format(k + k_min, len(pattern_treated_unfairly_lowerbound2[k])),
#           pattern_treated_unfairly_lowerbound2[k])
#
#
#
#
# k_printed = False
# print("p in pattern_treated_unfairly_lowerbound but not in pattern_treated_unfairly_lowerbound2:")
# for k in range(0, k_max - k_min):
#     for p in pattern_treated_unfairly_lowerbound[k]:
#         if p not in pattern_treated_unfairly_lowerbound2[k]:
#             if k_printed is False:
#                 print("k=", k + k_min)
#                 k_printed = True
#             print(p)
#
#
#
#
# k_printed = False
# print("p in pattern_treated_unfairly_lowerbound2 but not in pattern_treated_unfairly_lowerbound:")
# for k in range(0, k_max - k_min):
#     for p in pattern_treated_unfairly_lowerbound2[k]:
#         if p not in pattern_treated_unfairly_lowerbound[k]:
#             if k_printed is False:
#                 print("k=", k + k_min)
#                 k_printed = True
#             print(p)
#
# print("\n\n\n")
#
# k_printed = False
#
# print("p in pattern_treated_unfairly_upperbound but not in pattern_treated_unfairly_upperbound2:")
# for k in range(0, k_max - k_min):
#     for p in pattern_treated_unfairly_upperbound[k]:
#         if p not in pattern_treated_unfairly_upperbound2[k]:
#             if k_printed is False:
#                 print("k=", k + k_min)
#                 k_printed = True
#             print(p)
#
#
#
#
# k_printed = False
# print("p in pattern_treated_unfairly_upperbound2 but not in pattern_treated_unfairly_upperbound:")
# for k in range(0, k_max - k_min):
#     for p in pattern_treated_unfairly_upperbound2[k]:
#         if p not in pattern_treated_unfairly_upperbound[k]:
#             if k_printed is False:
#                 print("k=", k + k_min)
#                 k_printed = True
#             print(p)
#
#



