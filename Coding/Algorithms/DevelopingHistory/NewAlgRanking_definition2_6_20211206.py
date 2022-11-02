"""
Naive algorithm for group detection in ranking
fairness definition: the number of a group members in top-k should be proportional to the group size, k_min <= k <= k_max


bounds for different k can be different, and even for the same k, different patterns may have different bounds
This definition makes it different that, pattern p is above its lower bound, but its ancestor may not

So for alg in ranking in NewAlgRanking_8_20210702.py, CheckCandidatesForBounds(), we need to go up until the root.
We can't stop at a node whose size is above the lower bound.
Similarly, for upper bound, we can't stop at a node where it is above the upper bound/below the upper bound.
Its child may be also above the upper bound.
We need to go all the way to leaves, unless the size is too small.

We maintain a dict for nodes whose k is smallest locally.
How to find nearest ancestor: check each ancestor and see whether the ancestor is in this dict
For each node in this dict, if its descendant is in the dict, the descendant must have a smaller or equal k value


Why the problem of k only applies to lower bound: since we start from k_min to k_max.
If we do from k_max to k_min, it would be upper bound problem.

I maintain two stop sets, not allowing coexistence of ancestors and descendants
1. patterns in result set and whose size is too small
2. patterns are likely to violate conditions in the future


Problem:
To avoid coexistence of ancestors and descendants, every time when I want to add a pattern to stop set,
I need to check against each one in the set, which is toooooo time-consuming.

Stop set should allow dominance, so we use set(), add(), no need to check anything.

Two options:
1. maintain one stop set, like definition 1.
2. maintain two stop sets, like NewAlgRanking_definition2_3

Solution I'm implementing in this script:
Like in definition 1, only maintain one stop set, and allow dominance. (set(), add(), no need to check anything)

Do I use this stop set to decide when I need to stop searching in AddNewTuple()????


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
import math
import numpy as np
import pandas as pd
import NaiveAlgRanking_definition2_2_20211202 as naiveranking
from Algorithms import pattern_count
from sortedcontainers import SortedDict


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
        if s1 != s2 and s2 != " ":
            return False
        start_pos1 = p1 + 1
        start_pos2 = p2 + 1
    s1 = str1[start_pos1:]
    s2 = str2[start_pos2:]
    if s1 != s2 and s2 != " ":
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


#string to num when string has ' '
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
    def __init__(self, pattern, st, smallest_valid_k, smallest_ancestor, self_smallest_k):
        self.pattern = pattern
        self.st = st
        self.smallest_valid_k = smallest_valid_k
        # string of ancestor with smallest k, "" means itself
        # in case of same k, smallest_ancestor points to the ancestor rather than the node itself
        # since when we reach that k, all these nodes need updating
        self.smallest_ancestor = smallest_ancestor
        # whether this node has the smallest k in the path from the root
        self.self_smallest_k = self_smallest_k # must be true. It may have children with smaller k but it doesn't know



# find the closest ancestor of pattern p in nodes_dict_lowerbound
# by checking each of p's ancestor in nodes_dict_lowerbound
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
                parent_str = st[j+1:]
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
            parent_str = st[:i+1] + st[j+1:]
            if parent_str in string_set:
                return True, parent_str
            else:
                st = parent_str
                j = i - 1
                i -= 1
                continue
    return False, original_st


# assumption: p is not in nodes_dict_lowerbound
# in this function, we find the ancestor with the smallest k for pattern p
# and only add p if p has a smaller k
# if the k is same, don't add p
# this function is executed during a top-down search, so p's descendants are not in nodes_dict_lowerbound
def Add_node_to_set_lowerbound(nodes_dict_lowerbound, k_dict, smallest_valid_k, p, st, num_att):
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
                if parent in nodes_dict_lowerbound.keys():
                    if nodes_dict_lowerbound[parent].smallest_valid_k > smallest_valid_k:
                        nodes_dict_lowerbound[original_st] = Node(p, original_st, smallest_valid_k, parent, True)
                        k_dict[smallest_valid_k].append(original_st)
                        return
                    else: # smallest_valid_k is larger than the k value of an ancestor
                        return
                st = parent
                i -= 1
                break
            i -= 1
        att += 1
    # no ancestors in nodes_dict_lowerbound
    nodes_dict_lowerbound[original_st] = Node(p, original_st, smallest_valid_k, "", True)
    k_dict[smallest_valid_k].append(original_st)




def Add_node_to_set_upperbound(nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k, p, st, num_att, add_only=False):
    if add_only:
        nodes_dict_upperbound[st] = Node(p, st, largest_invalid_k, "", True)
        k_dict_upperbound[largest_invalid_k].append(st)
        return
    ancestor_st = ""
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
    added = False
    find_ancestor = False
    while att < num_att:
        while i > -1:
            if st[i] == '|':
                start = i
                parent = st[:start + 1] + st[end:]
                if parent in nodes_dict_upperbound.keys():
                    if nodes_dict_upperbound[parent].smallest_valid_k < largest_invalid_k:
                        nodes_dict_upperbound[original_st] = Node(p, original_st, largest_invalid_k, parent, True)
                        k_dict_upperbound[largest_invalid_k].append(original_st)
                        return
                    if nodes_dict_upperbound[parent].smallest_valid_k > largest_invalid_k:
                        k_dict_upperbound[nodes_dict_upperbound[parent].smallest_valid_k].pop(parent)
                        nodes_dict_upperbound.pop(parent)
                    if nodes_dict_upperbound[parent].smallest_valid_k == largest_invalid_k:
                        return
                st = parent
                i -= 1
                break
            i -= 1
        att += 1
    # no ancestors in nodes_dict_lowerbound
    nodes_dict_upperbound[original_st] = Node(p, original_st, largest_invalid_k, "", True)
    k_dict_upperbound[largest_invalid_k].append(original_st)



def Update_or_add_node_w_smaller_k(nodes_dict_lowerbound, k_dict, smallest_valid_k, p, st, smallest_ancestor):
    if st in nodes_dict_lowerbound.keys():
        k_dict[nodes_dict_lowerbound[st].smallest_valid_k].remove(st)
        k_dict[smallest_valid_k].append(st)
        nodes_dict_lowerbound[st].smallest_valid_k = smallest_valid_k
        nodes_dict_lowerbound[st].smallest_ancestor = smallest_ancestor
    else:
        k_dict[smallest_valid_k].append(st)
        nodes_dict_lowerbound[st] = Node(p, st, smallest_valid_k, smallest_ancestor, True)




def Update_or_add_node_w_larger_k(nodes_dict_upperbound, k_dict_upperbound, smallest_valid_k, p, st, smallest_ancestor):
    if st in nodes_dict_upperbound.keys():
        k_dict_upperbound[nodes_dict_upperbound[st].smallest_valid_k].remove(st)
        k_dict_upperbound[smallest_valid_k].append(st)
        nodes_dict_upperbound[st].smallest_valid_k = smallest_valid_k
        nodes_dict_upperbound[st].smallest_ancestor = smallest_ancestor
    else:
        k_dict_upperbound[smallest_valid_k].append(st)
        nodes_dict_upperbound[st] = Node(p, st, smallest_valid_k, smallest_ancestor, True)





# # for upperbound k value
# def Update_or_add_node_w_new_k(nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k_anc, p_ub, st_ub,
#                                p, st, largest_invalid_k):
#     if largest_invalid_k == largest_invalid_k_anc:
#         return
#     elif largest_invalid_k < largest_invalid_k_anc:
#         # go up until the ancestor whose k is < largest_invalid_k
#         ancestor = st_ub
#         while (largest_invalid_k_anc >= largest_invalid_k and ancestor != ""):
#             new_ancestor_node = nodes_dict_upperbound[ancestor].smallest_ancestor
#             k_dict_upperbound[nodes_dict_upperbound[ancestor].smallest_valid_k].remove(ancestor)
#             nodes_dict_upperbound.pop(ancestor)
#             largest_invalid_k_anc = new_ancestor_node.smallest_valid_k
#             ancestor = new_ancestor_node.st
#         if st in nodes_dict_upperbound.keys():
#             k_dict_upperbound[nodes_dict_upperbound[st].smallest_valid_k].remove(st)
#             k_dict_upperbound[largest_invalid_k].append(st)
#             nodes_dict_upperbound[st].smallest_valid_k = largest_invalid_k
#             nodes_dict_upperbound[st].smallest_ancestor = ancestor
#         else:
#             k_dict_upperbound[nodes_dict_upperbound[st].smallest_valid_k].remove(st)
#             k_dict_upperbound[largest_invalid_k].append(st)
#             nodes_dict_upperbound[st] = Node(p, st, largest_invalid_k, ancestor, True)
#     else: # largest_invalid_k > largest_invalid_k_anc
#         if st in nodes_dict_upperbound.keys():
#             k_dict_upperbound[nodes_dict_upperbound[st].smallest_valid_k].remove(st)
#             k_dict_upperbound[largest_invalid_k].append(st)
#             nodes_dict_upperbound[st].smallest_valid_k = largest_invalid_k
#             nodes_dict_upperbound[st].smallest_ancestor = st_ub
#         else:
#             k_dict_upperbound[nodes_dict_upperbound[st].smallest_valid_k].remove(st)
#             k_dict_upperbound[largest_invalid_k].append(st)
#             nodes_dict_upperbound[st] = Node(p, st, largest_invalid_k, st_ub, True)
#





def Check_and_remove_a_larger_k_lb(nodes_dict_lowerbound, k_dict, p, st):
    if st in nodes_dict_lowerbound.keys():
        old_k = nodes_dict_lowerbound[st].smallest_valid_k
        nodes_dict_lowerbound.pop(st)
        k_dict[old_k].remove(st)



# whether a is an ancestor of b, a and b are string
def A_is_ancestor_of_B(a, b):
    if len(a) >= len(b):
        return False
    length = len(a) # len(b) should >= len(a)
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


def Remove_descendants_str(c_str, patterns_to_search_lowest_level):
    to_remove = set()
    for st in patterns_to_search_lowest_level:
        if A_is_ancestor_of_B(c_str, st):
            to_remove.add(st)
    for r in to_remove:
        patterns_to_search_lowest_level.remove(r)





# p and st is the ancestor pattern so far with smallest k value
# c is the pattern to check, and it is sure that they deserve checking
def For_node_related_to_new_tuple(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k, p_lb, st_lb,
                                  nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k, p_ub, st_ub, c,
                                  data_size, alpha, patterns_top_k, whole_data_frame, attributes,
                                  k, new_tuple, ancestors, patterns_size_whole, pc_whole_data, result_set, num_att):
    ancestors.append(c)
    if c in result_set:
        return
    # check whether pattern c violates the conditions
    c_str = num2string(c)
    num_top_k_of_c = patterns_top_k.pattern_count(c_str)
    if c_str in patterns_size_whole:
        whole_cardinality_of_c = patterns_size_whole[c_str]
    else:
        whole_cardinality_of_c = pc_whole_data.pattern_count(c_str)
    if whole_cardinality_of_c < Thc:
        return
    lowerbound = (whole_cardinality_of_c / data_size - alpha) * k
    upperbound = (whole_cardinality_of_c / data_size + alpha) * k
    if whole_cardinality_of_c / data_size - alpha <= 0:
        smallest_valid_k_of_c = k_max + 1
    else:
        smallest_valid_k_of_c = math.floor(num_top_k_of_c / ((whole_cardinality_of_c / data_size) - alpha))
    if smallest_valid_k_of_c < k:
        raise Exception("smallest_valid_k_of_c < k")
    largest_invalid_k_of_c = math.ceil(num_top_k_of_c / ((whole_cardinality_of_c / data_size) + alpha))
    if largest_invalid_k_of_c > k_max:
        largest_invalid_k_of_c = k_max + 1
    if num_top_k_of_c < lowerbound or num_top_k_of_c > upperbound:
        CheckDominationAndAddForLowerbound(c, result_set)
        if smallest_valid_k_of_c < smallest_valid_k:
            Update_or_add_node_w_smaller_k(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k_of_c, c, c_str, st_lb)
        if largest_invalid_k_of_c > largest_invalid_k:
            Update_or_add_node_w_larger_k(nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k_of_c, c, c_str, st_ub)
        return
    # update k value for c, and continue checking for its children

    children, children_related_to_new_tuple = GenerateChildrenAndChildrenRelatedToNewTuple(c, whole_data_frame,
                                                                                           attributes, new_tuple)

    if smallest_valid_k_of_c < smallest_valid_k:
        Update_or_add_node_w_smaller_k(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k_of_c, c, c_str, st_lb)
        if largest_invalid_k_of_c > largest_invalid_k:
            Update_or_add_node_w_larger_k(nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k_of_c, c, c_str, st_ub)
            for grandc in children:
                if grandc in children_related_to_new_tuple:
                    For_node_related_to_new_tuple(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k_of_c, c, c_str,
                                                  nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k_of_c, c, c_str,
                                                  grandc,
                                                   data_size, alpha, patterns_top_k, whole_data_frame,
                                                   attributes, k, new_tuple, ancestors,
                                                   patterns_size_whole, pc_whole_data, result_set, num_att)
                else:
                    For_node_not_related_to_new_tuple(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k_of_c, c, c_str,
                                                  nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k_of_c, c, c_str,
                                                  grandc,
                                                   data_size, alpha, patterns_top_k, whole_data_frame,
                                                   attributes, k, new_tuple, ancestors,
                                                   patterns_size_whole, pc_whole_data, result_set, num_att)
        else:
            for grandc in children:
                if grandc in children_related_to_new_tuple:
                    For_node_related_to_new_tuple(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k_of_c, c, c_str,
                                                  nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k, p_ub, st_ub,
                                                  grandc,
                                                   data_size, alpha, patterns_top_k, whole_data_frame,
                                                   attributes, k, new_tuple, ancestors,
                                                   patterns_size_whole, pc_whole_data, result_set, num_att)
                else:
                    For_node_not_related_to_new_tuple(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k_of_c, c, c_str,
                                                  nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k, p_ub, st_ub,
                                                  grandc,
                                                   data_size, alpha, patterns_top_k, whole_data_frame,
                                                   attributes, k, new_tuple, ancestors,
                                                   patterns_size_whole, pc_whole_data, result_set, num_att)
    else:
        Check_and_remove_a_larger_k_lb(nodes_dict_lowerbound, k_dict_lowerbound, c, c_str)
        if largest_invalid_k_of_c > largest_invalid_k:
            Update_or_add_node_w_larger_k(nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k_of_c, c, c_str, st_ub)
            for grandc in children:
                if grandc in children_related_to_new_tuple:
                    For_node_related_to_new_tuple(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k, p_lb, st_lb,
                                                  nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k_of_c, c, c_str,
                                                  grandc,
                                                   data_size, alpha, patterns_top_k, whole_data_frame,
                                                   attributes, k, new_tuple, ancestors,
                                                   patterns_size_whole, pc_whole_data, result_set, num_att)
                else:
                    For_node_not_related_to_new_tuple(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k, p_lb, st_lb,
                                                  nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k_of_c, c, c_str,
                                                  grandc,
                                                   data_size, alpha, patterns_top_k, whole_data_frame,
                                                   attributes, k, new_tuple, ancestors,
                                                   patterns_size_whole, pc_whole_data, result_set, num_att)
        else:
            for grandc in children:
                if grandc in children_related_to_new_tuple:
                    For_node_related_to_new_tuple(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k, p_lb, st_lb,
                                                  nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k, p_ub, st_ub,
                                                  grandc,
                                                   data_size, alpha, patterns_top_k, whole_data_frame,
                                                   attributes, k, new_tuple, ancestors,
                                                   patterns_size_whole, pc_whole_data, result_set, num_att)
                else:
                    For_node_not_related_to_new_tuple(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k, p_lb, st_lb,
                                                  nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k, p_ub, st_ub,
                                                  grandc,
                                                   data_size, alpha, patterns_top_k, whole_data_frame,
                                                   attributes, k, new_tuple, ancestors,
                                                   patterns_size_whole, pc_whole_data, result_set, num_att)





# TODO: do i need to consider upperbound k values here? do i need to do anything to them?
# note isn't related to new tuple, so k doesn't change,
# but the smallest k value among this route may change
# and this tuple may need to be added to result set
# This function is to check whether this pattern should be added to result set, and see whether smallest k value changes
# c is a pattern which is either above, or in the stop set
# p and st is the current pattern with smallest k value in the ancestors
def For_node_not_related_to_new_tuple(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k, p_lb, st_lb,
                                      nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k, p_ub, st_ub,
                                      c,
                                      data_size, alpha, patterns_top_k, whole_data_frame, attributes, k, new_tuple,
                                      ancestors, patterns_size_whole, pc_whole_data, result_set, num_att):
    ancestors.append(c)
    if c in result_set:
        return
    c_str = num2string(c)
    if c_str in patterns_size_whole:
        whole_cardinality_of_c = patterns_size_whole[c_str]
    else:
        whole_cardinality_of_c = pc_whole_data.pattern_count(c_str)
    if whole_cardinality_of_c < Thc:
        return
    if c_str in nodes_dict_lowerbound.keys():
        if nodes_dict_lowerbound[c_str].smallest_valid_k == k or nodes_dict_upperbound[c_str].smallest_valid_k < k:
            CheckDominationAndAddForLowerbound(c, result_set)
            return
        if nodes_dict_lowerbound[c_str].smallest_valid_k < smallest_valid_k: # no need to go deeper for lower bound
            # TODO: Do i need to delete cstr and its descendents from nodes_dict_lowerbound?
            nodes_dict_lowerbound[c_str].smallest_ancestor = st_lb
            nodes_dict_lowerbound[c_str].self_smallest_k = True
            # if nodes_dict_upperbound[c_str].smallest_valid_k < largest_invalid_k: # no need to go deeper for upperbound
            #     # TODO: Do i need to do something here?? is it necessary to pop? not popping unnecessary patterns for now (ub)
            #     return

        else:
            k_dict_lowerbound[nodes_dict_lowerbound[c_str].smallest_valid_k].remove(c_str)
            nodes_dict_lowerbound.pop(c_str)
            children = GenerateChildren(c, whole_data_frame, attributes)
            for child in children:
                For_node_not_related_to_new_tuple(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k, p_lb, st_lb,
                                                  nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k, p_ub, st_ub,
                                                  child,
                                                  data_size, alpha, patterns_top_k, whole_data_frame,
                                                  attributes, k, new_tuple, ancestors, patterns_size_whole,
                                                  pc_whole_data, result_set, num_att)
    else:
        num_top_k_of_c = patterns_top_k.pattern_count(c_str)
        if whole_cardinality_of_c / data_size - alpha <= 0:
            smallest_valid_k_c = k_max + 1
        else:
            smallest_valid_k_c = math.floor(num_top_k_of_c / ((whole_cardinality_of_c / data_size) - alpha))
        largest_invalid_k_of_c = math.ceil(num_top_k_of_c / ((whole_cardinality_of_c / data_size) + alpha))
        if largest_invalid_k_of_c > k_max:
            largest_invalid_k_of_c = k_max + 1
        lowerbound = (whole_cardinality_of_c / data_size - alpha) * k
        upperbound = (whole_cardinality_of_c / data_size + alpha) * k
        if num_top_k_of_c < lowerbound or num_top_k_of_c > upperbound:
            CheckDominationAndAddForLowerbound(c, result_set)
            Add_node_to_set_lowerbound(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k_c, c, c_str, num_att)
            if c_str not in nodes_dict_upperbound.keys():
                Add_node_to_set_upperbound(nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k_of_c, c, c_str,
                                           num_att, True)
            return

        if smallest_valid_k_c >= smallest_valid_k:
            children = GenerateChildren(c, whole_data_frame, attributes)
            for child in children:
                For_node_not_related_to_new_tuple(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k, p_lb, st_lb,
                                                  nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k, p_ub, st_ub,
                                                  child,
                                                  data_size, alpha, patterns_top_k, whole_data_frame,
                                                  attributes, k, new_tuple, ancestors, patterns_size_whole,
                                                  pc_whole_data, result_set, num_att)
        else:
            k_dict_lowerbound[smallest_valid_k_c].append(c_str)
            nodes_dict_lowerbound[c_str] = Node(c, c_str, smallest_valid_k_c, st_lb, True)
            children = GenerateChildren(c, whole_data_frame, attributes)
            for child in children:
                For_node_not_related_to_new_tuple(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k_c, c, c_str,
                                                  nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k, p_ub, st_ub,
                                                  child,
                                                  data_size, alpha, patterns_top_k, whole_data_frame,
                                                  attributes, k, new_tuple, ancestors, patterns_size_whole,
                                                  pc_whole_data, result_set, num_att)




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
    root_str = '|' * (num_att - 1)
    S = GenerateChildren(root, whole_data_frame, attributes)
    pattern_treated_unfairly = []  # looking for the most general patterns
    patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k_min], encoded=False)
    patterns_top_kmin.parse_data()
    patterns_size_whole = dict()

    # this dict stores all patterns, indexed by num2string(p)
    nodes_dict_lowerbound = SortedDict()
    nodes_dict_upperbound = SortedDict()
    k_dict_lowerbound = dict()
    k_dict_upperbound = dict()
    time_setup1 = 0
    time_Add_node_to_set = 0
    # DFS
    # this part is the main time consumption

    result_set = []
    for k in range(0, k_max+2):
        k_dict_lowerbound[k] = []
        k_dict_upperbound[k] = []
    k = k_min
    while len(S) > 0:
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        time1 = time.time()
        P = S.pop(0)
        st = num2string(P)
        # print("GraphTraverse, st = {}".format(st))
        if st == "|0||":
            print("GraphTraverse, st={}".format(st))
            print("\n")
        num_patterns_visited += 1
        whole_cardinality = pc_whole_data.pattern_count(st)
        patterns_size_whole[st] = whole_cardinality
        time2 = time.time()
        time_setup1 += time2-time1
        if whole_cardinality < Thc:
            continue
        num_top_k = patterns_top_kmin.pattern_count(st)
        if whole_cardinality / data_size - alpha <= 0:
            smallest_valid_k = k_max + 1
        else:
            smallest_valid_k = math.floor(num_top_k / ((whole_cardinality / data_size) - alpha))
        if smallest_valid_k > k_max:
            smallest_valid_k = k_max + 1
        lowerbound = (whole_cardinality / data_size - alpha) * k
        upperbound = (whole_cardinality / data_size + alpha) * k
        if num_top_k < lowerbound or num_top_k > upperbound:
            CheckDominationAndAddForLowerbound(P, result_set)
            Add_node_to_set_lowerbound(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k,
                                       P, st, num_att)
            if num_top_k > upperbound:
                largest_invalid_k = math.ceil(num_top_k / ((whole_cardinality / data_size) + alpha))
                if largest_invalid_k > k_max:
                    largest_invalid_k = k_max + 1
                Add_node_to_set_upperbound(nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k,
                                           P, st, num_att, True)
        else:
            if P[num_att-1] != -1:
                children = GenerateChildren(P, whole_data_frame, attributes)
                S = children + S
            # maintain sets for k values only for a node not in result set.
            # so now we add the this node to nodes_dict_lowerbound
            # smallest k before which lower bound is ok

            if st not in nodes_dict_lowerbound.keys():
                time11 = time.time()
                Add_node_to_set_lowerbound(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k, P, st, num_att)
                time12 = time.time()
                time_Add_node_to_set += time12 - time11
    time1 = time.time()
    print("time for k_min = {}".format(time1 - time0))
    print("finish kmin")
    print(result_set)
    # print("patterns in nodes_dict_lowerbound:")
    # for st in nodes_dict_lowerbound.keys():
    #     print(st)
    pattern_treated_unfairly.append(result_set)

    for k in range(k_min + 1, k_max):
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        patterns_top_k = pattern_count.PatternCounter(ranked_data[:k], encoded=False)
        patterns_top_k.parse_data()
        new_tuple = ranked_data.iloc[[k - 1]].values.flatten().tolist()
        print("k={}, new tuple = {}".format(k, new_tuple))
        # top down for related patterns, using similar methods as k_min, add to result set if needed
        # ancestors are patterns checked in AddNewTuple() function, to avoid checking them again
        ancestors, num_patterns_visited, result_set = AddNewTuple(new_tuple, Thc,
                                                      whole_data_frame, patterns_top_k, k, k_min, pc_whole_data,
                                                      num_patterns_visited,
                                                      patterns_size_whole, alpha, num_att,
                                                      data_size, nodes_dict_lowerbound, k_dict_lowerbound,
                                                      nodes_dict_upperbound, k_dict_upperbound, attributes)

        num_patterns_visited = CheckCandidatesForKValues(nodes_dict_lowerbound, k_dict_lowerbound, ancestors, result_set,
                                        root, root_str, k,
                                        k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                                        alpha, num_att, whole_data_frame,
                                        attributes, num_patterns_visited, Thc, data_size)
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
def AddNewTuple(new_tuple, Thc, whole_data_frame, patterns_top_k, k, k_min, pc_whole_data, num_patterns_visited,
                patterns_size_whole, alpha, num_att, data_size, nodes_dict_lowerbound, k_dict_lowerbound,
                nodes_dict_upperbound, k_dict_upperbound, attributes):
    ancestors = []
    root = [-1] * num_att
    root_str = '|' * (num_att - 1)
    S = GenerateChildrenRelatedToTuple(root, new_tuple)  # pattern with one deterministic attribute
    # if the k values increases, go to function () without generating children
    # otherwise, generating children and add children to queue
    result_set = []
    while len(S) > 0:
        P = S.pop(0)
        st = num2string(P)
        if st == "|||0" and k == 11:
            print("st={}".format(st))
            print("\n")
        # print("in addnewtuple, st={}".format(st))
        time1 = time.time()
        ancestors.append(st)
        parent = findParent(P, num_att)
        parent_str = num2string(parent)
        num_patterns_visited += 1
        children, children_related_to_new_tuple = GenerateChildrenAndChildrenRelatedToNewTuple(P,
                                                                                               whole_data_frame,
                                                                                               attributes,
                                                                                               new_tuple)
        if st in patterns_size_whole:
            whole_cardinality = patterns_size_whole[st]
        else:
            whole_cardinality = pc_whole_data.pattern_count(st)
        if whole_cardinality < Thc:
            continue
        else:
            # special case: this pattern itself is in the result set
            num_top_k = patterns_top_k.pattern_count(st)
            lowerbound = (whole_cardinality / data_size - alpha) * k
            upperbound = (whole_cardinality / data_size + alpha) * k
            if whole_cardinality / data_size - alpha <= 0:
                smallest_valid_k = k_max + 1
            else:
                smallest_valid_k = math.floor(num_top_k / ((whole_cardinality / data_size) - alpha))
            if smallest_valid_k > k_max:
                smallest_valid_k = k_max + 1
            largest_invalid_k = math.ceil(num_top_k / ((whole_cardinality / data_size) + alpha))
            if largest_invalid_k > k_max:
                largest_invalid_k = k_max + 1
            if num_top_k < lowerbound or num_top_k > upperbound:
                CheckDominationAndAddForLowerbound(P, result_set)
                Update_or_add_node_w_smaller_k(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k, P, st, "")
                if num_top_k > upperbound:
                    Update_or_add_node_w_larger_k(nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k, P, st,
                                                   "")
                continue
            # smallest k before which lower bound is ok
            time5 = time.time()
            old_k_lowerbound = math.floor((num_top_k - 1) / ((whole_cardinality / data_size) - alpha))
            if old_k_lowerbound <= 0:
                old_k_lowerbound = k_max + 1
            old_k_upperbound = math.floor((num_top_k - 1) / ((whole_cardinality / data_size) + alpha))
            # we need to check k values for this node
            time6 = time.time()
            # print("time for setting up before Update_k_value is {}".format(time6 - time5))
            if smallest_valid_k > old_k_lowerbound or largest_invalid_k > old_k_upperbound:
                # this node only has one deterministic attribute, so if it is not in nodes_dict_lowerbound, it should be added
                if st not in nodes_dict_lowerbound.keys():
                    nodes_dict_lowerbound[st] = Node(P, st, smallest_valid_k, "", True)
                else:
                    nodes_dict_lowerbound[st].smallest_valid_k = smallest_valid_k
                for c in children:
                    if c in children_related_to_new_tuple:
                        For_node_related_to_new_tuple(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k, P, st,
                                                      nodes_dict_upperbound, k_dict_upperbound, largest_invalid_k, P, st,
                                                      c,
                                                      data_size, alpha, patterns_top_k, whole_data_frame,
                                                      attributes, k, new_tuple, ancestors,
                                                      patterns_size_whole, pc_whole_data, result_set, num_att)
                    else:
                        For_node_not_related_to_new_tuple(nodes_dict_lowerbound, k_dict_lowerbound, smallest_valid_k, P, st, c,
                                                      data_size, alpha, patterns_top_k, whole_data_frame,
                                                      attributes, k, new_tuple, ancestors,
                                                      patterns_size_whole, pc_whole_data, result_set, num_att)

                # time2 = time.time()
                # print("time for Update_k_value is {}".format(time2 - time1))
    return ancestors, num_patterns_visited, result_set




# check whether k values exceeds the smallest k for a pattern
# we check all patterns in the stop set, and go up
def CheckCandidatesForKValues(nodes_dict_lowerbound, k_dict, ancestors, result_set,
                             root, root_str, k,
                             k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                             alpha, num_att, whole_data_frame,
                             attributes, num_patterns_visited, Thc, data_size):
    for k_value in k_dict.keys():
        if k_value >= k:
            break
        for st in k_dict[k_value]:
            num_patterns_visited += 1
            if st in ancestors:
                continue
            p = nodes_dict_lowerbound[st].pattern
            CheckDominationAndAddForLowerbound(p, result_set)
    return num_patterns_visited

all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
                  'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C',
                  'failures_C', 'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C', 'higher_C',
                  'internet_C', 'romantic_C', 'famrel_C', 'freetime_C', 'goout_C', 'Dalc_C', 'Walc_C',
                  'health_C', 'absences_C', 'G1_C', 'G2_C', 'G3_C']

selected_attributes = ['school_C', 'sex_C', 'age_C', 'address_C']


original_data_file = r"../../../InputData/StudentDataset/ForRanking_1/student-mat_cat_ranked.csv"


ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]


time_limit = 5 * 60
k_min = 10
k_max = 13
Thc = 30

List_k = list(range(k_min, k_max))

alpha = 0.1


print("start the new alg")

pattern_treated_unfairly, num_patterns_visited, running_time = \
    GraphTraverse(ranked_data, selected_attributes, Thc,
                 alpha,
                 k_min, k_max, time_limit)

print("num_patterns_visited = {}".format(num_patterns_visited))
print("time = {} s".format(running_time))
for k in range(0, k_max-k_min):
    print("k = {}, num = {}, patterns =".format(k+k_min, len(pattern_treated_unfairly[k])), pattern_treated_unfairly[k])




print("start the naive alg")

pattern_treated_unfairly2, num_patterns_visited2, running_time2 = \
    naiveranking.NaiveAlg(ranked_data, selected_attributes, Thc,
                                                             alpha,
                                                             k_min, k_max, time_limit)


print("num_patterns_visited = {}".format(num_patterns_visited2))
print("time = {} s".format(running_time2))
for k in range(0, k_max-k_min):
    print("k = {}, num = {}, patterns =".format(k+k_min, len(pattern_treated_unfairly2[k])), pattern_treated_unfairly2[k])





print("p in pattern_treated_unfairly but not in pattern_treated_unfairly2:")
for k in range(0, k_max-k_min):
    print("k=", k+k_min)
    for p in pattern_treated_unfairly[k]:
        if p not in pattern_treated_unfairly2[k]:
            print(p)


print("\n\n\n")


print("p in pattern_treated_unfairly2 but not in pattern_treated_unfairly:")
for k in range(0, k_max-k_min):
    print("k=", k+k_min)
    for p in pattern_treated_unfairly2[k]:
        if p not in pattern_treated_unfairly[k]:
            print(p)



