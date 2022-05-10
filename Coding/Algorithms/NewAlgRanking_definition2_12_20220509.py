"""
Naive algorithm for group detection in ranking
fairness definition: the number of a group members in top-k should be proportional to the group size, k_min <= k
We don't include k_max here !

Expected output: most general patterns treated unfairly, w.r.t. lower bound
Difference from definition 1: in definition 1, we find most specific patterns for upper bound,
most general for lower bound, and all patterns have same bounds.

But here, patterns have different bounds depending on their sizes. We only do lower bound.

"""

"""
naive alg:
AniveAlgRanking_definition2_3...
for each k, we iterate the whole process again, go top down.
"""

"""
This alg:
lowerbound = alpha * whole_cardinality * k / data_size

difference from 11:
store string instead of pattern list in result_set

"""

import time
import math
import numpy as np
import pandas as pd
from Algorithms import NaiveAlgRanking_definition2_5_20220506 as naiveranking
from Algorithms import pattern_count
from sortedcontainers import SortedDict
import cProfile
import pstats
import logging


def P1DominatedByP2ForStr(str1, str2, num_att):
    if str1 == str2:
        return True
    num_separator = num_att - 1
    start_pos1 = 0
    start_pos2 = 0
    for i in range(num_separator):
        p1 = str1.find("|", start_pos1)
        p2 = str2.find("|", start_pos2)
        s1 = str1[start_pos1:p1]
        s2 = str2[start_pos2:p2]
        if s1 != s2 and s2 != '':
            return False
        start_pos1 = p1 + 1
        start_pos2 = p2 + 1
    s1 = str1[start_pos1:]
    s2 = str2[start_pos2:]
    if s1 != s2 and s2 != '':
        return False
    return True


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


# for a tuple, there is one parent generating it
# but it has more than one parent, and those who doesn't generate this tuple, their size also increase
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


def GenerateChildrenAndChildrenRelatedToNewTuple(P, whole_data_frame, attributes, new_tuple):
    children = []
    children_related_to_new_tuple = []
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
            if s[j] == new_tuple[j]:
                children_related_to_new_tuple.append(s)
    return children, children_related_to_new_tuple


def num2string(pattern):
    st = ''
    for i in pattern:
        if i != -1:
            st += str(i)
        st += '|'
    st = st[:-1]
    return st


# string to num when string has ' '
# def string2num(st):
#     p = []
#     items = st.split('|')
#     for i in items:
#         if i == ' ':
#             p.append(-1)
#         else:
#             p.append(int(i))
#     return p


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


# find parent when string has ' '
# def findParentForStr(child):
#     end = 0
#     length = len(child)
#     i = length - 1
#     while i > -1:
#         if child[i] != '|' and child[i] != ' ':
#             end = i + 1
#             i -= 1
#             break
#         i -= 1
#     while i > -1:
#         if child[i] == '|':
#             start = i
#             parent = child[:start + 1] + ' ' + child[end:]
#             return parent
#         i -= 1
#     parent = ' ' + child[end:]
#     return parent
#


# find parent when string doesn't have ' '
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


# closest ancestor must be the ancestor with smallest k value
# smallest_ancestor != "" if and only if there is an ancestor having the same k value
class Node:
    # init method or constructor
    def __init__(self, pattern, st, smallest_valid_k):
        self.pattern = pattern
        self.st = st
        self.smallest_valid_k = smallest_valid_k
        # string of ancestor with smallest k, "" means itself
        # in case of same k, smallest_ancestor points to the ancestor rather than the node itself
        # since when we reach that k, all these nodes need updating
        # self.smallest_ancestor = smallest_ancestor
        # whether this node has the smallest k in the path from the root
        # self.self_smallest_k = self_smallest_k  # must be true. It may have children with smaller k but it doesn't know


# find the closest ancestor of pattern p in nodes_dict
# by checking each of p's ancestor in nodes_dict
def Find_closest_ancestor(string_set, st, num_att):
    if st in string_set:
        return True, st
    original_st = st
    length = len(st)
    j = length - 1
    i = length - 1
    find = False
    while True:
        if i < 0:
            if find:
                parent_str = st[j + 1:]
                if parent_str in string_set:
                    return True, parent_str
                else:
                    return False, original_st
            else:
                return False, original_st
        if find is False and st[i] == "|":
            i -= 1
            j -= 1
            continue
        elif find is False and st[i] != "|":
            j = i
            find = True
            i -= 1
            continue
        elif find and st[i] != "|":
            i -= 1
            continue
        else:
            parent_str = st[:i + 1] + st[j + 1:]
            if parent_str in string_set:
                return True, parent_str
            else:
                st = parent_str
                j = i - 1
                i -= 1
                continue
    return False, original_st


# assumption: p is not in nodes_dict, and we don't know its ancestor
# in this function, we find the ancestor with the smallest k for pattern p
# and only add p if p has a smaller k
# if the k is same, don't add p
# this function is executed during a top-down search, so p's descendants are not in nodes_dict
def Add_node_to_set(nodes_dict, k_dict, smallest_valid_k, p, st, num_att):
    att = 0
    end = 0
    length = len(st)
    i = length - 1
    original_st = st
    while i > -1:
        if st[i] != '|':
            end = i + 1
            i -= 1
            break
        if st[i] == '|':
            att += 1
        i -= 1
    while att < num_att:
        while i > -1:
            if st[i] == '|':
                start = i
                parent = st[:start + 1] + st[end:]
                if parent in nodes_dict.keys():
                    if nodes_dict[parent].smallest_valid_k > smallest_valid_k:
                        nodes_dict[original_st] = Node(p, original_st, smallest_valid_k)
                        k_dict[smallest_valid_k].append(original_st)
                        return
                    else:  # smallest_valid_k is larger than the k value of an ancestor
                        return
                st = parent
                i -= 1
                break
            i -= 1
        att += 1
    # no ancestors in nodes_dict
    nodes_dict[original_st] = Node(p, original_st, smallest_valid_k)
    k_dict[smallest_valid_k].append(original_st)


def Update_or_add_node_w_smaller_k(nodes_dict, k_dict, smallest_valid_k, p, st):
    if st in nodes_dict.keys():
        k_dict[nodes_dict[st].smallest_valid_k].remove(st)
        k_dict[smallest_valid_k].append(st)
        nodes_dict[st].smallest_valid_k = smallest_valid_k
    else:
        k_dict[smallest_valid_k].append(st)
        nodes_dict[st] = Node(p, st, smallest_valid_k)


def Check_and_remove_a_larger_k(nodes_dict, k_dict, p, st):
    if st in nodes_dict.keys():
        old_k = nodes_dict[st].smallest_valid_k
        nodes_dict.pop(st)
        k_dict[old_k].remove(st)


# whether a is an ancestor of b, a and b are string
def A_is_ancestor_of_B(a, b):
    if len(a) >= len(b):
        return False
    length = len(a)  # len(b) should >= len(a)
    find_undeterministic = False
    i = 0
    for i in range(length):
        if a[i] != b[i]:
            if a[i] != "|":
                return False
            else:
                find_undeterministic = True
                break
    for j in range(i, length):
        if a[j] != "|":
            return False
    return True


def PatternInSet(p, set):
    if isinstance(p, str):
        p = string2num(p)
    for q in set:
        if PatternEqual(p, q):
            return True
    return False




def AddDominatedToLowerbound(pattern, pattern_treated_unfairly, dominated_by_result):
    to_remove = []
    to_remove_from_dominated_by_result = []
    for p in pattern_treated_unfairly:
        # if PatternEqual(p, pattern):
        #     return
        if P1DominatedByP2(pattern, p):
            return False
        elif P1DominatedByP2(p, pattern):
            to_remove.append(p)
    for p in to_remove:
        pattern_treated_unfairly.remove(p)
    pattern_treated_unfairly.append(pattern)
    return True

#
# # return whether it is added or not, patterns are stored as list
# def CheckDominationAndAddForLowerbound(pattern, pattern_treated_unfairly, dominated_by_result):
#     to_remove = []
#     for p in pattern_treated_unfairly:
#         # if PatternEqual(p, pattern):
#         #     return
#         if P1DominatedByP2(pattern, p):
#             if pattern not in dominated_by_result:
#                 dominated_by_result.append(pattern)
#             return False
#         elif P1DominatedByP2(p, pattern):
#             to_remove.append(p)
#             if p not in dominated_by_result:
#                 dominated_by_result.append(p)
#     for p in to_remove:
#         pattern_treated_unfairly.remove(p)
#     if pattern in dominated_by_result:
#         dominated_by_result.remove(pattern)
#     pattern_treated_unfairly.append(pattern)
#     return True
#


# return whether it is added or not, strings are stored rather than patterns
def CheckDominationAndAddForLowerbound(pattern_st, pattern_treated_unfairly, dominated_by_result, num_att):
    to_remove = []
    for st in pattern_treated_unfairly:
        # if PatternEqual(p, pattern):
        #     return
        if P1DominatedByP2ForStr(pattern_st, st, num_att):
            if pattern_st not in dominated_by_result:
                dominated_by_result.append(pattern_st)
            return False
        elif P1DominatedByP2ForStr(st, pattern_st, num_att):
            to_remove.append(st)
            if st not in dominated_by_result:
                dominated_by_result.append(st)
    for st in to_remove:
        pattern_treated_unfairly.remove(st)
    if pattern_st in dominated_by_result:
        dominated_by_result.remove(pattern_st)
    pattern_treated_unfairly.append(pattern_st)
    return True


def Remove_descendants_str(c_str, patterns_to_search_lowest_level):
    to_remove = set()
    for st in patterns_to_search_lowest_level:
        if A_is_ancestor_of_B(c_str, st):
            to_remove.add(st)
    for r in to_remove:
        patterns_to_search_lowest_level.remove(r)


# Stop set: doesn't allow ancestor and children, but allow dominance between others.
# We can only put a node itself into the stop set, whether it is in result set or not.
# Nodes in stop set:
# 1. size is too small
# 2. already in result set
# 3. others.
def GraphTraverse(ranked_data, attributes, Thc, alpha, k_min, k_max, time_limit):
    time0 = time.time()
    data_size = len(ranked_data)
    pc_whole_data = pattern_count.PatternCounter(ranked_data, encoded=False)
    pc_whole_data.parse_data()
    whole_data_frame = ranked_data.describe(include='all')
    num_patterns_visited = 0
    num_att = len(attributes)
    root = [-1] * num_att
    S = GenerateChildren(root, whole_data_frame, attributes)
    root_str = '|' * (num_att - 1)
    store_children = {root_str: S}
    pattern_treated_unfairly = []  # looking for the most general patterns
    patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k_min], encoded=False)
    patterns_top_kmin.parse_data()
    patterns_size_whole = dict()
    k_dict = dict()
    dominated_by_result = []

    # this dict stores all patterns, indexed by num2string(p)
    nodes_dict = SortedDict()
    time_setup1 = 0
    time_Add_node_to_set = 0
    # DFS
    # this part is the main time consumption

    result_set = []
    for k in range(0, k_max + 2):
        k_dict[k] = []
    k = k_min
    while len(S) > 0:
        # if time.time() - time0 > time_limit:
        #     print("newalg overtime")
        #     break
        time1 = time.time()
        P = S.pop(0)
        st = num2string(P)
        # print("GraphTraverse, st = {}".format(st))
        if st == '|0|2|':
            print("stop here")
        num_patterns_visited += 1
        whole_cardinality = pc_whole_data.pattern_count(st)
        patterns_size_whole[st] = whole_cardinality
        time2 = time.time()
        time_setup1 += time2 - time1
        if whole_cardinality < Thc:
            continue
        num_top_k = patterns_top_kmin.pattern_count(st)
        smallest_valid_k = math.floor(num_top_k * data_size / (alpha * whole_cardinality))
        if smallest_valid_k > k_max:
            smallest_valid_k = k_max + 1
        elif smallest_valid_k < k_min:
            smallest_valid_k = k_min - 1
        # lowerbound = (whole_cardinality / data_size - alpha) * k
        lowerbound = alpha * whole_cardinality * k / data_size
        # print("pattern {}, lb = {}, smallest_valid_k = {}".format(P, lowerbound, smallest_valid_k))
        if num_top_k < lowerbound:
            CheckDominationAndAddForLowerbound(st, result_set, dominated_by_result, num_att)
            Add_node_to_set(nodes_dict, k_dict, smallest_valid_k, P, st, num_att)
        else:
            if P[num_att - 1] == -1:
                if st in store_children:
                    children = store_children[st]
                else:
                    children = GenerateChildren(P, whole_data_frame, attributes)
                    store_children[st] = children
                S = children + S
            # maintain sets for k values only for a node not in result set.
            # so now we add this node to nodes_dict
            # smallest k before which lower bound is ok
            if st not in nodes_dict.keys():
                time11 = time.time()
                Add_node_to_set(nodes_dict, k_dict, smallest_valid_k, P, st, num_att)
                time12 = time.time()
                time_Add_node_to_set += time12 - time11
            else:
                raise Exception("st is impossible to be in nodes_dict.keys()")
    time1 = time.time()
    print("time for k_min = {}".format(time1 - time0))
    print("finish kmin")
    # print(result_set)
    pattern_treated_unfairly.append(result_set)

    for k in range(k_min + 1, k_max):
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        time1 = time.time()
        patterns_top_k = pattern_count.PatternCounter(ranked_data[:k], encoded=False)
        patterns_top_k.parse_data()
        new_tuple = ranked_data.iloc[[k - 1]].values.flatten().tolist()
        # print("k={}, new tuple = {}".format(k, new_tuple))
        # top down for related patterns, using similar methods as k_min, add to result set if needed
        # ancestors are patterns checked in AddNewTuple() function, to avoid checking them again
        result_set = pattern_treated_unfairly[k - 1 - k_min].copy()
        # print("before add new tuple")
        # if '||2||' in result_set:
        #     print("||2|| is in result set")
        # if '||2||' in dominated_by_result:
        #     print("||2|| is in dominated by result set")
        ancestors, num_patterns_visited = AddNewTuple(new_tuple, Thc,
                                                      whole_data_frame, patterns_top_k, k, k_min, k_max,
                                                      pc_whole_data,
                                                      num_patterns_visited,
                                                      patterns_size_whole, alpha, num_att,
                                                      data_size, nodes_dict, k_dict, attributes,
                                                      result_set, dominated_by_result, store_children)

        time2 = time.time()
        # print("after addnewtuple")
        # if st in nodes_dict.keys():
        #     print("k of {} = {}".format(st, nodes_dict[st].smallest_valid_k))
        # else:
        #     print("st not in nodes_dict")
        # print("time for AddNewTuple = {}".format(time2-time1))
        # print("result_set after AddNewTuple: ", result_set)
        time3 = time.time()
        # print("after add new tuple")
        # if '||2||' in result_set:
        #     print("||2|| is in result set")
        # if '||2||' in dominated_by_result:
        #     print("||2|| is in dominated by result set")
        to_added_to_dominated_by_result = []
        to_remove_from_dominated_by_result = []
        for d in dominated_by_result:
            to_remove_from_result_set = []
            d_dominated_by_result_set = False
            for st in result_set:
                if P1DominatedByP2ForStr(d, st, num_att):
                    d_dominated_by_result_set = True
                    break
                elif P1DominatedByP2ForStr(st, d, num_att):
                    to_remove_from_result_set.append(st)
                    to_added_to_dominated_by_result.append(st)
            if not d_dominated_by_result_set:
                for p in to_remove_from_result_set:
                    result_set.remove(p)
                result_set.append(d)
                to_remove_from_dominated_by_result.append(d)
        for d in to_remove_from_dominated_by_result:
            if d in to_added_to_dominated_by_result:
                to_added_to_dominated_by_result.remove(d)
            else:
                dominated_by_result.remove(d)
        for d in to_added_to_dominated_by_result:
            dominated_by_result.append(d)
        for st in k_dict[k-1]:
            if st in ancestors:
                continue
            if st in result_set:
                continue
            CheckDominationAndAddForLowerbound(st, result_set, dominated_by_result, num_att)
        time4 = time.time()
        # print("time for CheckCandidatesForKValues = {}".format(time4 - time3))
        # print("result_set after CheckCandidatesForKValues: ", result_set)
        pattern_treated_unfairly.append(result_set)
    time1 = time.time()
    return pattern_treated_unfairly, num_patterns_visited, time1 - time0


# search top-down to go over all patterns related to new_tuple
# using similar checking methods as k_min
# add to result set if they are outliers
# need to update k values for these patterns

# for lower bound, when k and s_k increase by 1 at the same time, Sk/k is still larger than Sd(p)-Sd-alpha
# So if a pattern is above lower bound, after this, it is still above lower bound
# No patterns will be added to result set for lower bound here. Some will be added to upper bound result set
# but the smallest k values may maintain unchanged, may also increase
# thus, for lower bound, we only need to check for k values;
# for upper bound, we only need to check whether it is above upper bound
def AddNewTuple(new_tuple, Thc, whole_data_frame, patterns_top_k, k, k_min, k_max, pc_whole_data, num_patterns_visited,
                patterns_size_whole, alpha, num_att, data_size, nodes_dict, k_dict, attributes,
                result_set, dominated_by_result, store_children):
    ancestors = []
    root = [-1] * num_att
    S = GenerateChildrenRelatedToTuple(root, new_tuple)  # pattern with one deterministic attribute
    # if the k values increases, go to function () without generating children
    # otherwise, generating children and add children to queue
    K_values = [k_max] * len(S)
    time_smaller_than_lb = 0
    time_check_k = 0
    time_generate_children = 0
    time_generate_children1 = 0
    time_generate_children2 = 0
    time_update_k_list = 0
    while len(S) > 0:
        P = S.pop(0)
        st = num2string(P)
        smallest_valid_k_ancestor = K_values.pop(0)
        # if PatternEqual(P, [-1, -1, -1, -1, -1, 1, -1, -1, -1, 0]):
        #     print("st={}".format(st))
        #     print("\n")
        # print("in addnewtuple, st={}".format(st))
        time1 = time.time()
        children = []
        ancestors.append(P)
        num_patterns_visited += 1
        if st in patterns_size_whole:
            whole_cardinality = patterns_size_whole[st]
        else:
            whole_cardinality = pc_whole_data.pattern_count(st)
        if whole_cardinality < Thc:
            continue
        else:
            # special case: this pattern itself is in the result set
            num_top_k = patterns_top_k.pattern_count(st)
            lowerbound = alpha * whole_cardinality * k / data_size
            # lowerbound = (whole_cardinality / data_size - alpha) * k
            smallest_valid_k = math.floor(num_top_k * data_size / (alpha * whole_cardinality))
            if smallest_valid_k > k_max:
                smallest_valid_k = k_max + 1
            elif smallest_valid_k < k_min:
                smallest_valid_k = k_min - 1
            if num_top_k < lowerbound:
                Update_or_add_node_w_smaller_k(nodes_dict, k_dict, smallest_valid_k, P, st)
                if st in result_set:
                    continue
                CheckDominationAndAddForLowerbound(st, result_set, dominated_by_result, num_att)
                    # print("added {} to result set when k = {}".format(P, k))
                time2 = time.time()
                time_smaller_than_lb += time2 - time1
                # print("time 1-2 in AddNewTuple = {}".format(time2 - time1))
                continue
            time3 = time.time()
            time_smaller_than_lb += time3 - time1
            if smallest_valid_k_ancestor > smallest_valid_k:
                Update_or_add_node_w_smaller_k(nodes_dict, k_dict, smallest_valid_k, P, st)
            else:
                Check_and_remove_a_larger_k(nodes_dict, k_dict, P, st)
            time4 = time.time()
            time_check_k += time4 - time3
            time5 = time.time()
            if st in result_set:
                result_set.remove(st)
            else:
                if st in dominated_by_result:
                    dominated_by_result.remove(st)
            time56 = time.time()
            if P[num_att - 1] == -1:
                if st in store_children:
                    children = store_children[st]
                else:
                    children = GenerateChildren(P, whole_data_frame, attributes)
                    store_children[st] = children
                S = S + children
            time6 = time.time()
            if P[num_att - 1] != -1:
                time_generate_children1 += time56 - time5
                time_generate_children2 += time6 - time56
                time_generate_children += time6 - time5
                time_update_k_list += time.time() - time6
                continue
            if smallest_valid_k_ancestor > smallest_valid_k:
                K_values = K_values + [smallest_valid_k] * len(children)
            else:
                K_values = K_values + [smallest_valid_k_ancestor] * len(children)
            time7 = time.time()
            time_generate_children1 += time56 - time5
            time_generate_children2 += time6 - time56
            time_generate_children += time6 - time5
            time_update_k_list += time7 - time6

    print(time_smaller_than_lb, time_check_k, time_generate_children1, time_generate_children2, time_update_k_list)
    return ancestors, num_patterns_visited


all_attributes = ["age_binary", "sex_binary", "race_C", "MarriageStatus_C", "juv_fel_count_C",
                  "decile_score_C", "juv_misd_count_C", "juv_other_count_C", "priors_count_C", "days_b_screening_arrest_C",
                  "c_days_from_compas_C", "v_decile_score_C", "c_charge_degree_C", "start_C", "end_C",
                  "event_C"]

# 11 att, 17s VS 35s
# 12 att, 70 VS 88

selected_attributes = all_attributes[:13]

original_data_file = r"../../InputData/CompasData/ForRanking/LargeDatasets/compas_data_cat_necessary_att_ranked.csv"

ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]

time_limit = 5 * 60
k_min = 10
k_max = 30
Thc = 50

List_k = list(range(k_min, k_max))

alpha = 0.8

logger = logging.getLogger('MyLogger')

print("start the new alg")

pattern_treated_unfairly, num_patterns_visited, running_time = \
    GraphTraverse(ranked_data, selected_attributes, Thc,
                  alpha, k_min, k_max, time_limit)

print("num_patterns_visited = {}".format(num_patterns_visited))
print("time = {} s".format(running_time))
# for k in range(0, k_max - k_min):
#     print("k = {}, num = {}, patterns =".format(k + k_min, len(pattern_treated_unfairly[k])),
#           pattern_treated_unfairly[k])


print("start the naive alg")

pattern_treated_unfairly2, num_patterns_visited2, running_time2 = \
    naiveranking.NaiveAlg(ranked_data, selected_attributes, Thc,
                          alpha,
                          k_min, k_max, time_limit)

print("num_patterns_visited = {}".format(num_patterns_visited2))
print("time = {} s".format(running_time2))
# for k in range(0, k_max - k_min):
#     print("k = {}, num = {}, patterns =".format(k + k_min, len(pattern_treated_unfairly2[k])),
#           pattern_treated_unfairly2[k])


k_printed = False
print("p in pattern_treated_unfairly but not in pattern_treated_unfairly2:")
for k in range(0, k_max - k_min):
    for p in pattern_treated_unfairly[k]:
        if p not in pattern_treated_unfairly2[k]:
            if k_printed is False:
                print("k=", k + k_min)
                k_printed = True
            print(p)

print("\n\n\n")

k_printed = False
print("p in pattern_treated_unfairly2 but not in pattern_treated_unfairly:")
for k in range(0, k_max - k_min):
    for p in pattern_treated_unfairly2[k]:
        if p not in pattern_treated_unfairly[k]:
            if k_printed is False:
                print("k=", k + k_min)
                k_printed = True
            print(p)
