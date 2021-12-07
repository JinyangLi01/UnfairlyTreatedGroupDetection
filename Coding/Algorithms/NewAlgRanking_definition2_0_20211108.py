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

Current Problem:
- How to update a k value? If a node has a new k, which may be larger, we need to check all its descendant???

Why the problem of k only applies to lower bound: since we start from k_min to k_max.j If we do from k_max to k_min, it would be upper bound problem.


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
import NaiveAlgRanking_definition2_0_20211108 as naiveranking
from Algorithms import pattern_count


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
        self.self_smallest_k = self_smallest_k # must be true. It may have children with smaller k but it doesn't know



# find the closest ancestor of pattern p in nodes_dict
# by checking each of p's ancestor in nodes_dict
def Find_closest_ancestor(nodes_dict, p, st, num_att):
    if st in nodes_dict.keys():
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
                if parent_str in nodes_dict:
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
            if parent_str in nodes_dict:
                return True, parent_str
            else:
                st = parent_str
                j = i - 1
                i -= 1
                continue
    return False, original_st

# before executing this function, we need to check whether this node has already been in nodes_dict
# in this function, we find the ancestor with the smallest k for pattern p
# and point p to this ancestor if p's k is larger
# otherwise, point p to itself
def Add_node_to_set(nodes_dict, smallest_valid_k, p, st, parent, parent_str, root, root_str, num_att):
    if st == "0||||||":
        print("St={}".format(st))
    if parent_str == root_str:
        nodes_dict[st] = Node(p, st, smallest_valid_k, "", True)
        return
    # find ancestor with the smallest k
    find, ancestor_st = Find_closest_ancestor(nodes_dict, p, st, num_att)
    if find:
        node = nodes_dict[ancestor_st]
        while node.smallest_ancestor != "":
            node = nodes_dict[node.smallest_ancestor]
        if node.smallest_valid_k > smallest_valid_k:
            nodes_dict[st] = Node(p, st, smallest_valid_k, "", True)
        elif node.smallest_valid_k == smallest_valid_k:
            nodes_dict[st] = Node(p, st, smallest_valid_k, ancestor_st, True)
    else:
        nodes_dict[st] = Node(p, st, smallest_valid_k, "", True)


# a pattern in nodes_set doesn't have any ancestors with smaller k also in nodes_set !!!
# TODO: how to update k values...
# since all patterns in this branch will be traversed in a top-down fashion
# for each pattern, we just need to find its closest ancestor, also find whether it itself is in nodes_set
# and compare the k value
def Update_k_value(nodes_dict, smallest_valid_k, p, st, parent, parent_str, root, root_str, num_att):
    # print(nodes_dict)
    find, ancestor_st = Find_closest_ancestor(nodes_dict, p, st, num_att)
    if find:
        ancestor_node = nodes_dict[ancestor_st]
        if st in nodes_dict.keys():
            nodes_dict[st].smallest_valid_k = smallest_valid_k
            if ancestor_node.smallest_valid_k > smallest_valid_k:
                nodes_dict[st].smallest_ancestor = ""
                nodes_dict[st].self_smallest_k = True
            elif ancestor_node.smallest_valid_k == smallest_valid_k:
                nodes_dict[st].smallest_ancestor = ancestor_node.st
                nodes_dict[st].self_smallest_k = True
            else:
                nodes_dict.pop(st)
        else:
            if ancestor_node.smallest_valid_k > smallest_valid_k:
                nodes_dict[st] = Node(p, st, smallest_valid_k, "", True)
            elif ancestor_node.smallest_valid_k == smallest_valid_k:
                nodes_dict[st] = Node(p, st, smallest_valid_k, ancestor_st, True)
    else:
        nodes_dict[st] = Node(p, st, smallest_valid_k, "", True)


#
# def Update_k_value(nodes_dict, smallest_valid_k, p, st, parent, parent_str, root, root_str):
#     # print(nodes_dict)
#     nodes_dict[st].smallest_valid_k = smallest_valid_k
#     if parent_str == root_str:
#         nodes_dict[st].smallest_ancestor = ""
#         nodes_dict[st].self_smallest_k = True
#         return
#     # find ancestor with the smallest k
#     node = nodes_dict[parent_str]
#     while node.smallest_ancestor != "":
#         node = nodes_dict[node.smallest_ancestor]
#     if node.smallest_valid_k > smallest_valid_k:
#         nodes_dict[st].smallest_ancestor = ""
#         nodes_dict[st].self_smallest_k = True
#     elif node.smallest_valid_k == smallest_valid_k:
#         nodes_dict[st].smallest_ancestor = node.st
#         nodes_dict[st].self_smallest_k = True
#     else:
#         nodes_dict[st].smallest_ancestor = node.st
#         nodes_dict[st].self_smallest_k = False

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



def GraphTraverse(ranked_data, attributes, Thc, alpha, k_min, k_max, time_limit):
    # print("attributes:", attributes)
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
    pattern_treated_unfairly_lowerbound = []  # looking for the most general patterns
    pattern_treated_unfairly_upperbound = []  # looking for the most specific patterns
    patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k_min], encoded=False)
    patterns_top_kmin.parse_data()
    patterns_size_whole = dict()
    k = k_min
    patterns_searched_lowest_level_lowerbound = set()
    patterns_searched_lowest_level_upperbound = set()
    # this dict stores all patterns, indexed by num2string(p)
    nodes_dict = dict()
    nodes_list = [] # sorted, parents are before children

    parent_candidate_for_upperbound = []
    # DFS
    # this part is the main time consumption
    while len(S) > 0:
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        P = S.pop(0)
        st = num2string(P)
        # print("st={}, lower bound".format(st))
        # if st == "0||||1||4":
        #     print("st={}".format(st))
        num_patterns_visited += 1
        whole_cardinality = pc_whole_data.pattern_count(st)
        patterns_size_whole[st] = whole_cardinality
        parent = findParent(P, num_att)
        parent_str = num2string(parent)
        children = []
        added_children = False
        if whole_cardinality < Thc:
            if len(parent_candidate_for_upperbound) > 0:  # there is a parent which is above upper bound
                CheckDominationAndAddForUpperbound(parent_candidate_for_upperbound, pattern_treated_unfairly_upperbound)
                parent_candidate_for_upperbound = []
            # patterns in patterns_searched_lowest_level all have valid whole cardinality
            # and are not in pattern_treated_unfairly
            # ================== time consuming ===============
            if parent_str != root_str:
                patterns_searched_lowest_level_lowerbound.add(parent_str)
                patterns_searched_lowest_level_upperbound.add(parent_str)
            continue
        num_top_k = patterns_top_kmin.pattern_count(st)
        lowerbound = (whole_cardinality / data_size - alpha) * k
        if num_top_k < lowerbound:
            if parent_str != root_str:
                patterns_searched_lowest_level_lowerbound.add(parent_str)
            CheckDominationAndAddForLowerbound(P, pattern_treated_unfairly_lowerbound)
            # maintain sets for k values only for a node not in result set.
            # so now we add the parent to nodes_dict
            # smallest k before which lower bound is ok
            if parent_str != root_str:
                whole_cardinality = patterns_size_whole[parent_str]
                if whole_cardinality / data_size - alpha <= 0:
                    smallest_valid_k = k_max + 1
                else:
                    smallest_valid_k = math.floor(num_top_k / ((whole_cardinality / data_size) - alpha))
                parent_of_parent = findParent(parent, num_att)
                if parent_str not in nodes_dict.keys():
                    Add_node_to_set(nodes_dict, smallest_valid_k, parent, parent_str, parent_of_parent,
                                num2string(parent_of_parent), root, root_str, num_att)
        else:
            children = GenerateChildren(P, whole_data_frame, attributes)
            if len(children) == 0:
                patterns_searched_lowest_level_lowerbound.add(st)
            S = children + S
            added_children = True
            # maintain sets for k values only for a node not in result set.
            # so now we add the this node to nodes_dict
            # smallest k before which lower bound is ok
            if whole_cardinality / data_size - alpha <= 0:
                smallest_valid_k = k_max + 1
            else:
                smallest_valid_k = math.floor(num_top_k / ((whole_cardinality / data_size) - alpha))
            if st not in nodes_dict.keys():
                Add_node_to_set(nodes_dict, smallest_valid_k, P, st, parent, parent_str, root, root_str, num_att)

        print("upperbound")
        upperbound = (whole_cardinality / data_size + alpha) * k
        if num_top_k > upperbound:
            parent_candidate_for_upperbound = P  # we need to store this so that if child is below upper bound, we put this into result set
            if added_children is False:
                children = GenerateChildren(P, whole_data_frame, attributes)
                S = children + S
                added_children = True
            if len(children) == 0:  # P is in result set
                CheckDominationAndAddForUpperbound(P, pattern_treated_unfairly_upperbound)
                parent_candidate_for_upperbound = []
        else:
            if added_children is False:
                children = GenerateChildren(P, whole_data_frame, attributes)
                S = children + S
                added_children = True
            if len(children) == 0:  # P is in result set
                if len(parent_candidate_for_upperbound) > 0:  # add its ancestor to the result set
                    CheckDominationAndAddForUpperbound(parent_candidate_for_upperbound, pattern_treated_unfairly_upperbound)
                    parent_candidate_for_upperbound = []
            # if len(parent_candidate_for_upperbound) > 0:  # P is not above upperbound, so its parent should be added to the result set
            #     CheckDominationAndAddForUpperbound(parent_candidate_for_upperbound, pattern_treated_unfairly_upperbound)
            #     parent_candidate_for_upperbound = []
    print("finish kmin")
    # if PatternInSet("0|0|||0", pattern_treated_unfairly_lowerbound):
    #     print("after kmin, in lowerbound\n".format(k))
    for k in range(k_min + 1, k_max):
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        print("k={}".format(k))
        if PatternInSet("0|0|||0", pattern_treated_unfairly_lowerbound):
            print("k={}, in lowerbound\n".format(k))
            print("stop here!")
        patterns_top_k = pattern_count.PatternCounter(ranked_data[:k], encoded=False)
        patterns_top_k.parse_data()
        new_tuple = ranked_data.iloc[[k - 1]].values.flatten().tolist()
        # top down for related patterns, using similar methods as k_min, add to result set if needed
        # ancestors are patterns checked in AddNewTuple() function, to avoid checking them again
        ancestors, num_patterns_visited = AddNewTuple(new_tuple, Thc, pattern_treated_unfairly_lowerbound,
                                                      pattern_treated_unfairly_upperbound,
                                                      whole_data_frame, patterns_top_k, k, k_min, pc_whole_data,
                                                      num_patterns_visited,
                                                      patterns_size_whole, alpha, num_att,
                                                      data_size, nodes_dict)
        # but in naive, it doesn't
        # check all patterns in the stop set whether the current k is the smallest k
        num_patterns_visited, patterns_searched_lowest_level_lowerbound, patterns_searched_lowest_level_upperbound \
            = CheckCandidatesForKValues(nodes_dict, ancestors, patterns_searched_lowest_level_lowerbound,
                                           patterns_searched_lowest_level_upperbound, root, root_str,
                                           pattern_treated_unfairly_lowerbound,
                                           pattern_treated_unfairly_upperbound, k,
                                           k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                                           alpha, num_att, whole_data_frame,
                                           attributes, num_patterns_visited, Thc, data_size)
        if PatternInSet("0|0|||0", pattern_treated_unfairly_lowerbound):
            print("@@@@@@@@@@@")
    time1 = time.time()
    return pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound, num_patterns_visited, time1 - time0

def PatternInSet(p, set):
    if isinstance(p, str):
        p = string2num(p)
    for q in set:
        if PatternEqual(p, q):
            return True
    return False

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



# check whether k values exceeds the smallest k for a pattern
def CheckCandidatesForKValues(nodes_dict, ancestors, patterns_searched_lowest_level_lowerbound,
                             patterns_searched_lowest_level_upperbound, root, root_str,
                             pattern_treated_unfairly_lowerbound,
                             pattern_treated_unfairly_upperbound, k,
                             k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                             alpha, num_att, whole_data_frame,
                             attributes, num_patterns_visited, Thc, data_size):
    to_remove = set()
    to_append = set()
    for st in patterns_searched_lowest_level_lowerbound:  # st is a string
        # if st == "0|0|||0":
        #     print("CheckCandidatesForKValues, st = {}".format(st))
        num_patterns_visited += 1
        p = string2num(st)
        if p in ancestors or p in pattern_treated_unfairly_lowerbound:  # already checked
            continue
        find, nearest_ancestor_str = Find_closest_ancestor(nodes_dict, p, st, num_att)
        if find:
            nearest_ancestor_node = nodes_dict[nearest_ancestor_str]
        else:
            raise ValueError('ancestors in nodes_dict not found')
        if nearest_ancestor_node.smallest_valid_k >= k:
            continue
        # nearest_ancestor.smallest_valid_k < k, this should be added to result set
        while nearest_ancestor_node.smallest_ancestor != "":
            nearest_ancestor_node = nodes_dict[nearest_ancestor_node.smallest_ancestor]
            nodes_dict.pop(nearest_ancestor_str)
            nearest_ancestor_str = nearest_ancestor_node.st

        child_str = nearest_ancestor_str
        parent_str = findParentForStr(child_str)
        child = nearest_ancestor_node
        if parent_str == root_str:
            CheckDominationAndAddForLowerbound(child, pattern_treated_unfairly_lowerbound)
            to_remove.add(st)  # child need removing
        else:
            CheckDominationAndAddForLowerbound(child, pattern_treated_unfairly_lowerbound)
            to_remove.add(st)
            to_append.add(parent_str)
    for p_str in to_remove:
        patterns_searched_lowest_level_lowerbound.remove(p_str)
    patterns_searched_lowest_level_lowerbound = patterns_searched_lowest_level_lowerbound | to_append

    return num_patterns_visited, patterns_searched_lowest_level_lowerbound, patterns_searched_lowest_level_upperbound


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
    for st in patterns_searched_lowest_level_lowerbound:  # st is a string
        num_patterns_visited += 1
        p = string2num(st)
        if p in ancestors or p in pattern_treated_unfairly_lowerbound:  # already checked
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
        # if pattern_size_in_topk < Lowerbounds[k - k_min], remove thild and add this parent
        child_str = st
        parent_str = findParentForStr(child_str)
        child = string2num(child_str)
        if parent_str == root_str:
            CheckDominationAndAddForLowerbound(child, pattern_treated_unfairly_lowerbound)
            to_remove.add(child_str)  # child need removing
            continue

        # if parent is not root, we need to check until 1: the root, 2: we find a node that is above the lower bound
        # since lower bound may increase by 5, and parent is below the lower bound, grandparent may be too
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

    return num_patterns_visited, patterns_searched_lowest_level_lowerbound, patterns_searched_lowest_level_upperbound


# search top-down to go over all patterns related to new_tuple
# using similar checking methods as k_min
# add to result set if they are outliers
# need to update k values for these patterns
def AddNewTuple(new_tuple, Thc, pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound,
                whole_data_frame, patterns_top_k, k, k_min, pc_whole_data, num_patterns_visited,
                patterns_size_whole, alpha, num_att, data_size, nodes_dict):
    ancestors = []
    root = [-1] * num_att
    root_str = '|' * (num_att - 1)
    children = GenerateChildrenRelatedToTuple(root, new_tuple)  # pattern with one deternimistic attribute
    S = children
    parent_candidate_for_upperbound = []
    while len(S) > 0:
        P = S.pop(0)
        st = num2string(P)
        if "0|0|||||" == st:
            print("st={}\n".format(st))
            print("stop here!")
        print("st={}".format(st))
        parent = findParent(P, num_att)
        parent_str = num2string(parent)
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
            print("lower bound")
            num_top_k = patterns_top_k.pattern_count(st)
            lowerbound = (whole_cardinality / data_size - alpha) * k
            if num_top_k < lowerbound:
                CheckDominationAndAddForLowerbound(new_tuple, pattern_treated_unfairly_lowerbound)
            else:
                S = children + S
                ancestors = ancestors + children
                add_children = True
            # smallest k before which lower bound is ok
            if whole_cardinality / data_size - alpha <= 0:
                smallest_valid_k = k_max + 1
            else:
                smallest_valid_k = math.floor(num_top_k / ((whole_cardinality / data_size) - alpha))
            if st in nodes_dict.keys():
                Update_k_value(nodes_dict, smallest_valid_k, P, st, parent, parent_str, root, root_str, num_att)
            else:
                Add_node_to_set(nodes_dict, smallest_valid_k, P, st, parent, parent_str, root, root_str, num_att)
            print("upper bound")
            upperbound = (whole_cardinality / data_size + alpha) * k
            if num_top_k > upperbound:
                parent_candidate_for_upperbound = P
                if not add_children:
                    S = children + S
                    ancestors = ancestors + children
                if len(children) == 0:
                    CheckDominationAndAddForUpperbound(P, pattern_treated_unfairly_upperbound)
                    parent_candidate_for_upperbound = []
            else:  # below the upper bound
                if len(parent_candidate_for_upperbound) > 0:
                    CheckDominationAndAddForUpperbound(parent_candidate_for_upperbound,
                                                       pattern_treated_unfairly_upperbound)
                    parent_candidate_for_upperbound = []

    return ancestors, num_patterns_visited


all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
                  'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C',
                  'failures_C', 'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C', 'higher_C',
                  'internet_C', 'romantic_C', 'famrel_C', 'freetime_C', 'goout_C', 'Dalc_C', 'Walc_C',
                  'health_C', 'absences_C', 'G1_C', 'G2_C', 'G3_C']

selected_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C']



original_data_file = r"../../InputData/StudentDataset/ForRanking_1/student-mat_cat_ranked.csv"


ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]


time_limit = 5 * 60
k_min = 10
k_max = 15
Thc = 50

List_k = list(range(k_min, k_max))


alpha = 0.06


print("start the new alg")

pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound, num_patterns_visited, running_time = \
    GraphTraverse(ranked_data, selected_attributes, Thc,
                 alpha,
                 k_min, k_max, time_limit)

print("num_patterns_visited = {}".format(num_patterns_visited))
print("time = {} s, num of pattern_treated_unfairly_lowerbound = {}, num of pattern_treated_unfairly_upperbound = {} ".format(running_time,
        len(pattern_treated_unfairly_lowerbound), len(pattern_treated_unfairly_upperbound)), "\n", "patterns:\n",
      pattern_treated_unfairly_lowerbound, "\n", pattern_treated_unfairly_upperbound)

print("dominated by pattern_treated_unfairly_lowerbound:")
for p in pattern_treated_unfairly_lowerbound:
    if PDominatedByM(p, pattern_treated_unfairly_lowerbound)[0]:
        print(p)





print("start the naive alg")

pattern_treated_unfairly_lowerbound2, pattern_treated_unfairly_upperbound2, \
num_patterns_visited2, running_time2 = naiveranking.NaiveAlg(ranked_data, selected_attributes, Thc,
                                                             alpha,
                                                             k_min, k_max, time_limit)


print("num_patterns_visited = {}".format(num_patterns_visited2))
print("time = {} s, num of pattern_treated_unfairly_lowerbound = {}, num of pattern_treated_unfairly_upperbound = {} ".format(running_time2,
        len(pattern_treated_unfairly_lowerbound2), len(pattern_treated_unfairly_upperbound2)), "\n", "patterns:\n",
      pattern_treated_unfairly_lowerbound2, "\n", pattern_treated_unfairly_upperbound2)




print("dominated by pattern_treated_unfairly2:")
for p in pattern_treated_unfairly_lowerbound2:
    t, m = PDominatedByM(p, pattern_treated_unfairly_lowerbound2)
    if t:
        print("{} dominated by {}".format(p, m))



print("p in pattern_treated_unfairly_lowerbound but not in pattern_treated_unfairly_lowerbound2:")
for p in pattern_treated_unfairly_lowerbound:
    if p not in pattern_treated_unfairly_lowerbound2:
        print(p)


print("\n\n\n")

print("p in pattern_treated_unfairly_lowerbound2 but not in pattern_treated_unfairly_lowerbound:")
for p in pattern_treated_unfairly_lowerbound2:
    if p not in pattern_treated_unfairly_lowerbound:
        print(p)


print("\n\n\n")

print("p in pattern_treated_unfairly_upperbound but not in pattern_treated_unfairly_upperbound2:")
for p in pattern_treated_unfairly_upperbound:
    if p not in pattern_treated_unfairly_upperbound2:
        print(p)


print("\n\n\n")

print("p in pattern_treated_unfairly_upperbound2 but not in pattern_treated_unfairly_upperbound:")
for p in pattern_treated_unfairly_upperbound2:
    if p not in pattern_treated_unfairly_upperbound:
        print(p)



