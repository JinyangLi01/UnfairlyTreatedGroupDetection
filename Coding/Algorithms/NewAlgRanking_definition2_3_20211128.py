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

I maintain two stop sets:
1. patterns in result set and whose size is too small
2. patterns are likely to violate conditions in the future



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
import NaiveAlgRanking_definition2_1_20211108 as naiveranking
from Algorithms import pattern_count


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
        if s1 != s2 and s2 != "":
            return False
        start_pos1 = p1 + 1
        start_pos2 = p2 + 1
    s1 = str1[start_pos1:]
    s2 = str2[start_pos2:]
    if s1 != s2 and s2 != "":
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
        # whether this node has the smallest k in the path from the root
        self.self_smallest_k = self_smallest_k # must be true. It may have children with smaller k but it doesn't know



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


# before executing this function, we need to check whether this node has already been in nodes_dict
# in this function, we find the ancestor with the smallest k for pattern p
# and point p to this ancestor if p's k is larger
# otherwise, point p to itself
def Add_node_to_set(nodes_dict, smallest_valid_k, p, st, parent, parent_str, root, root_str, num_att):
    if parent_str == root_str:
        nodes_dict[st] = Node(p, st, smallest_valid_k, "", True)
        return
    # find ancestor with the smallest k
    find, ancestor_st = Find_closest_ancestor(nodes_dict.keys(), st, num_att)
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


def Check_duplicate_and_add(pattern_list, new_pattern):
    for p in pattern_list:
        if PatternEqual(p, new_pattern):
            return
    pattern_list.append(new_pattern)


def PatternInSet(p, set):
    if isinstance(p, str):
        p = string2num(p)
    for q in set:
        if PatternEqual(p, q):
            return True
    return False




# Instead of checking dominance, we should check whether this node is
def CheckDominationAndAddForStopSetStr(str, patterns_searched_lowest_level_lowerbound):
    to_remove = []
    for p in patterns_searched_lowest_level_lowerbound:
        if str == p:
            return
        elif A_is_ancestor_of_B(str, p):
            to_remove.append(p)
        elif A_is_ancestor_of_B(p, str):
            return
    for q in to_remove:
        patterns_searched_lowest_level_lowerbound.remove(q)
    patterns_searched_lowest_level_lowerbound.append(str)



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

# this is used when we add a new tuple, and the k is increased by 1
# we only care about k values for nodes above stop set
#
# a pattern in nodes_set doesn't have any ancestors with smaller k also in nodes_set !!!
# since all patterns in this branch will be traversed in a top-down fashion
# for each pattern, we just need to find its closest ancestor, also find whether it itself is in nodes_set
# and compare the k value

# Makes sure that parameter p is not in result set and not in or below patterns_no_need_searching
def Update_k_value(nodes_dict, smallest_valid_k, p, st, parent, parent_str, root, root_str, num_att,
                   pattern_treated_unfairly_lowerbound, patterns_to_search_lowest_level, patterns_no_need_searching,
                   whole_data_frame, attributes, children_of_p_related_to_new_tuple,
                   patterns_top_k, whole_cardinality, data_size, alpha, k, patterns_size_whole, pc_whole_data,
                   new_tuple):
    # print("In update_k_value")
    find, ancestor_st = Find_closest_ancestor(patterns_to_search_lowest_level, st, num_att)
    if find:  # this pattern either exists in stop set itself or has a ancestor in stop set (it is below the stop set)
        if st in nodes_dict.keys():
            nodes_dict[st].smallest_valid_k = smallest_valid_k
            if ancestor_st == st: # this pattern is in stop set
                parent_str = findParentForStr(st)
                find2, ancestor_st2 = Find_closest_ancestor(nodes_dict.keys(), parent_str, num_att)
                if find2: # should be able to find in most cases, if not found, it means parent_str = root_str
                    if nodes_dict[ancestor_st2].smallest_valid_k == smallest_valid_k:
                        nodes_dict[st].smallest_ancestor = ancestor_st2
                    elif nodes_dict[ancestor_st2].smallest_valid_k < smallest_valid_k:
                        nodes_dict.pop(st) # there is a smaller ancestor, and this node has a larger k, so it is removed
                else: # if not found, it means parent_str = root_str
                    # the current pattern only has one deterministic attribute, and it is in the stop set
                    # so we only need to change the k value of this node, which is done several lines above
                    # and then return, without searching deeper
                    return
            else: # this pattern has an ancestor in stop set, which means this pattern is below the stop set
                return # in this case, we don't need to do anything for this pattern
        else:
            if ancestor_st == st: # this pattern is in stop set
                parent_str = findParentForStr(st)
                find2, ancestor_st2 = Find_closest_ancestor(nodes_dict.keys(), parent_str, num_att)
                if find2: # should be able to find in most cases, if not found, it means parent_str = root_str
                    if nodes_dict[ancestor_st2].smallest_valid_k == smallest_valid_k:
                        nodes_dict[st] = Node(p, st, smallest_valid_k, ancestor_st2, True)
                    elif nodes_dict[ancestor_st2].smallest_valid_k > smallest_valid_k:
                        nodes_dict[st] = Node(p, st, smallest_valid_k, "", True)
                else: # if not found, it means parent_str = root_str
                    return
    else: # this pattern is above the stop set, and it is possible not tracked by nodes_dict
        if st in nodes_dict.keys():
            nodes_dict[st].smallest_valid_k = smallest_valid_k
        else:
            find2, ancestor_st2 = Find_closest_ancestor(nodes_dict.keys(), st, num_att)
            if not find2:
                raise Exception("This is impossible!")
            if smallest_valid_k == nodes_dict[ancestor_st2].smallest_valid_k:
                nodes_dict[st] = Node(p, st, smallest_valid_k, ancestor_st2, True)
            elif smallest_valid_k < nodes_dict[ancestor_st2].smallest_valid_k:
                nodes_dict[st] = Node(p, st, smallest_valid_k, "", True)
            else:
                smallest_valid_k = nodes_dict[ancestor_st2].smallest_valid_k
        children = GenerateChildren(p, whole_data_frame, attributes)
        # print("children:", children)
        for c in children:
            if c in pattern_treated_unfairly_lowerbound:
                continue
            c_str = num2string(c)
            if c_str in patterns_no_need_searching:
                continue
            # if c == [0, -1, -1, 1, -1, -1]:
            #     print("c={}".format(c))
            #     print("\n")
            if c in children_of_p_related_to_new_tuple:
                # if c violates the conditions

                if c_str in patterns_size_whole:
                    whole_cardinality = patterns_size_whole[c_str]
                else:
                    whole_cardinality = pc_whole_data.pattern_count(c_str)
                if whole_cardinality < Thc: # this condition is impossible to meet
                    raise Exception("whole_cardinality_of_child < Thc")
                num_top_k_of_c = patterns_top_k.pattern_count(c_str)
                children_of_c_related_to_new_tuple = GenerateChildrenRelatedToTuple(c, new_tuple)
                For_node_related_to_new_tuple(nodes_dict, smallest_valid_k, p, st, c, c_str, num_top_k_of_c,
                                              patterns_to_search_lowest_level, patterns_no_need_searching,
                                              pattern_treated_unfairly_lowerbound,
                                              data_size, alpha, patterns_top_k, whole_cardinality, whole_data_frame,
                                              attributes, children_of_c_related_to_new_tuple, k, new_tuple,
                                              patterns_size_whole, pc_whole_data)
            else:
                For_node_not_related_to_new_tuple(nodes_dict, smallest_valid_k, p, st, c, c_str,
                                              patterns_to_search_lowest_level, patterns_no_need_searching,
                                              data_size, alpha, patterns_top_k, whole_data_frame, attributes, k,
                                                  patterns_size_whole, pc_whole_data)


# p and st is the pattern so far with smallest k value
# c, c_str is the pattern to check, and it is sure that they deserve checking
def For_node_related_to_new_tuple(nodes_dict, smallest_valid_k, p, st, c, c_str, num_top_k_of_c,
                                  patterns_to_search_lowest_level, patterns_no_need_searching,
                                  pattern_treated_unfairly_lowerbound,
                                  data_size, alpha, patterns_top_k, whole_cardinality_of_c, whole_data_frame, attributes,
                                  children_of_c_related_to_new_tuple, k, new_tuple, patterns_size_whole, pc_whole_data):
    # check whether pattern c violates the conditions
    lowerbound = (whole_cardinality_of_c / data_size - alpha) * k
    upperbound = (whole_cardinality_of_c / data_size + alpha) * k
    if num_top_k_of_c < lowerbound or num_top_k_of_c > upperbound:
        CheckDominationAndAddForStopSetStr(c_str, patterns_no_need_searching)
        Remove_descendants_str(c_str, patterns_to_search_lowest_level)
        CheckDominationAndAddForLowerbound(c, pattern_treated_unfairly_lowerbound)
        if c_str in nodes_dict.keys():  # TODO: c_str is already in result set, I think I pop here.
            nodes_dict.pop(c_str)
        return
    # update k value for c, and continue checking for its children
    if whole_cardinality_of_c / data_size - alpha <= 0:
        smallest_valid_k_of_c = k_max + 1
    else:
        smallest_valid_k_of_c = math.floor(num_top_k_of_c / ((whole_cardinality_of_c / data_size) - alpha))
    if smallest_valid_k_of_c < k: # TODO: What does this mean? this means this pattern doesn't satisfy conditions now???
        raise Exception("smallest_valid_k_of_c < k")
    if c_str in nodes_dict.keys():
        nodes_dict[c_str].smallest_valid_k = smallest_valid_k_of_c
        if smallest_valid_k_of_c < smallest_valid_k:
            nodes_dict[c_str].smallest_ancestor = ""
            nodes_dict[c_str].self_smallest_k = True
            if c_str in patterns_to_search_lowest_level:
                return
            Continue_check_k_for_children(smallest_valid_k_of_c, c, c_str, c, c_str, patterns_to_search_lowest_level,
                             patterns_no_need_searching,
                             whole_data_frame,
                             attributes, children_of_c_related_to_new_tuple, nodes_dict, data_size,
                             patterns_top_k, k, new_tuple, patterns_size_whole, pc_whole_data,
                             pattern_treated_unfairly_lowerbound)
        elif smallest_valid_k_of_c == smallest_valid_k:
            nodes_dict[c_str].smallest_ancestor = st
            nodes_dict[c_str].self_smallest_k = True
            if c_str in patterns_to_search_lowest_level:
                return
            Continue_check_k_for_children(smallest_valid_k_of_c, c, c_str, c, c_str, patterns_to_search_lowest_level,
                             patterns_no_need_searching,
                             whole_data_frame,
                             attributes, children_of_c_related_to_new_tuple, nodes_dict, data_size,
                             patterns_top_k, k, new_tuple, patterns_size_whole, pc_whole_data,
                             pattern_treated_unfairly_lowerbound)
        else:
            if c_str in patterns_to_search_lowest_level:
                return
            Continue_check_k_for_children(smallest_valid_k, p, st, c, c_str, patterns_to_search_lowest_level,
                             patterns_no_need_searching,
                             whole_data_frame,
                             attributes, children_of_c_related_to_new_tuple, nodes_dict, data_size,
                             patterns_top_k, k, new_tuple, patterns_size_whole, pc_whole_data,
                             pattern_treated_unfairly_lowerbound)
    else:
        if smallest_valid_k_of_c == smallest_valid_k:
            nodes_dict[c_str] = Node(c, c_str, smallest_valid_k_of_c, st, True)
            if c_str in patterns_to_search_lowest_level:
                return
            Continue_check_k_for_children(smallest_valid_k_of_c, c, c_str, c, c_str, patterns_to_search_lowest_level,
                             patterns_no_need_searching,
                             whole_data_frame,
                             attributes, children_of_c_related_to_new_tuple, nodes_dict, data_size,
                             patterns_top_k, k, new_tuple, patterns_size_whole, pc_whole_data,
                             pattern_treated_unfairly_lowerbound)
        elif smallest_valid_k_of_c < smallest_valid_k:
            nodes_dict[c_str] = Node(c, c_str, smallest_valid_k_of_c, "", True)
            if c_str in patterns_to_search_lowest_level:
                return
            Continue_check_k_for_children(smallest_valid_k_of_c, c, c_str, c, c_str, patterns_to_search_lowest_level,
                             patterns_no_need_searching,
                             whole_data_frame,
                             attributes, children_of_c_related_to_new_tuple, nodes_dict, data_size,
                             patterns_top_k, k, new_tuple, patterns_size_whole, pc_whole_data,
                             pattern_treated_unfairly_lowerbound)
        else:
            if c_str in patterns_to_search_lowest_level:
                return
            Continue_check_k_for_children(smallest_valid_k, p, st, c, c_str, patterns_to_search_lowest_level,
                             patterns_no_need_searching,
                             whole_data_frame,
                             attributes, children_of_c_related_to_new_tuple, nodes_dict, data_size,
                             patterns_top_k, k, new_tuple, patterns_size_whole, pc_whole_data,
                             pattern_treated_unfairly_lowerbound)



# generate children of c
def Continue_check_k_for_children(smallest_valid_k, p, st, c, c_str, patterns_to_search_lowest_level,
                     patterns_no_need_searching, whole_data_frame,
                     attributes, children_of_c_related_to_new_tuple, nodes_dict, data_size,
                     patterns_top_k, k, new_tuple, patterns_size_whole, pc_whole_data,
                     pattern_treated_unfairly_lowerbound):
    children = GenerateChildren(c, whole_data_frame, attributes)
    for child in children:
        if child in pattern_treated_unfairly_lowerbound:
            continue
        child_str = num2string(child)
        if child_str in patterns_no_need_searching:
            continue
        num_top_k_of_child = patterns_top_k.pattern_count(child_str)
        if child_str in patterns_size_whole:
            whole_cardinality_of_child = patterns_size_whole[child_str]
        else:
            whole_cardinality_of_child = pc_whole_data.pattern_count(child_str)
        if whole_cardinality_of_child < Thc:
            raise Exception("whole_cardinality_of_child < Thc")
        if child in children_of_c_related_to_new_tuple:
            children_of_child_related_to_new_tuple = GenerateChildrenRelatedToTuple(child, new_tuple)
            For_node_related_to_new_tuple(nodes_dict, smallest_valid_k, p, st, child, child_str, num_top_k_of_child,
                                          patterns_to_search_lowest_level, patterns_no_need_searching,
                                          pattern_treated_unfairly_lowerbound,
                                          data_size, alpha, patterns_top_k, whole_cardinality_of_child,
                                          whole_data_frame, attributes, children_of_child_related_to_new_tuple, k,
                                          new_tuple, patterns_size_whole, pc_whole_data)
        else:
            For_node_not_related_to_new_tuple(nodes_dict, smallest_valid_k, p, st, child, child_str,
                                              patterns_to_search_lowest_level, patterns_no_need_searching,
                                              data_size, alpha, patterns_top_k,
                                              whole_data_frame, attributes, k, patterns_size_whole, pc_whole_data)


# note isn't related to new tuple, so k doesn't change, but the smallest k value among this route may change
# This function is to deal with the situations that smallest k value changes
# c is a pattern which is either above, or in the stop set
# p and st is the current pattern with smallest k value in the ancestors
def For_node_not_related_to_new_tuple(nodes_dict, smallest_valid_k, p, st, c, c_str, patterns_to_search_lowest_level,
                                      patterns_no_need_searching,
                                  data_size, alpha, patterns_top_k, whole_data_frame, attributes, k,
                                      patterns_size_whole, pc_whole_data):
    if c_str in nodes_dict.keys():
        if nodes_dict[c_str].smallest_valid_k < smallest_valid_k: # no need to go deeper
            nodes_dict[c_str].smallest_ancestor = ""
            nodes_dict[c_str].self_smallest_k = True
        elif nodes_dict[c_str].smallest_valid_k == smallest_valid_k:
            nodes_dict[c_str].smallest_ancestor = st
            nodes_dict[c_str].self_smallest_k = True
        else:
            # smallest_valid_k is the current smallest value of k, so it may be smaller than k values of c's descendants
            nodes_dict[c_str].smallest_ancestor = st
            nodes_dict[c_str].self_smallest_k = False
            if c_str not in patterns_to_search_lowest_level:
                children = GenerateChildren(c, whole_data_frame, attributes)
                for child in children:
                    # if child in pattern_treated_unfairly_lowerbound: # TODO: this checking is redundant??
                    #     continue
                    child_str = num2string(child)
                    if child_str in patterns_no_need_searching:
                        continue
                    For_node_not_related_to_new_tuple(nodes_dict, smallest_valid_k, p, st, child, child_str,
                                                      patterns_to_search_lowest_level, patterns_no_need_searching,
                                                      data_size, alpha, patterns_top_k, whole_data_frame,
                                                      attributes, k, patterns_size_whole, pc_whole_data)
    else:
        num_top_k_of_c = patterns_top_k.pattern_count(c_str)
        if c_str in patterns_size_whole:
            whole_cardinality_of_c = patterns_size_whole[c_str]
        else:
            whole_cardinality_of_c = pc_whole_data.pattern_count(c_str)
        if whole_cardinality_of_c / data_size - alpha <= 0:
            smallest_valid_k_of_c = k_max + 1
        else:
            smallest_valid_k_of_c = math.floor(num_top_k_of_c / ((whole_cardinality_of_c / data_size) - alpha))
        if smallest_valid_k_of_c < k:
            raise Exception("smallest_valid_k_of_c < k")
        if smallest_valid_k_of_c == smallest_valid_k:
            nodes_dict[c_str] = Node(c, c_str, smallest_valid_k_of_c, st, True)
            if c_str not in patterns_to_search_lowest_level:
                children = GenerateChildren(c, whole_data_frame, attributes)
                for child in children:
                    child_str = num2string(child)
                    if child_str in patterns_no_need_searching:
                        continue
                    For_node_not_related_to_new_tuple(nodes_dict, smallest_valid_k_of_c, c, c_str, child, child_str,
                                                      patterns_to_search_lowest_level, patterns_no_need_searching,
                                                      data_size, alpha, patterns_top_k,
                                                      whole_data_frame, attributes, k, patterns_size_whole, pc_whole_data)
        elif smallest_valid_k_of_c < smallest_valid_k:
            nodes_dict[c_str] = Node(c, c_str, smallest_valid_k_of_c, "", True)

        else:
            if c_str not in patterns_to_search_lowest_level:
                children = GenerateChildren(c, whole_data_frame, attributes)
                for child in children:
                    child_str = num2string(child)
                    if child_str in patterns_no_need_searching:
                        continue
                    For_node_not_related_to_new_tuple(nodes_dict, smallest_valid_k, p, st, child, child_str,
                                                      patterns_to_search_lowest_level, patterns_no_need_searching,
                                                      data_size, alpha, patterns_top_k,
                                                      whole_data_frame, attributes, k, patterns_size_whole, pc_whole_data)
    return




# Stop set: doesn't allow ancestor and children, but allow dominance between others.
# We can only put a node itself into the stop set, whether it is in result set or not.
# Nodes in stop set:
# 1. size is too small
# 2. already in result set
# 3. others.
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
    patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k_min], encoded=False)
    patterns_top_kmin.parse_data()
    patterns_size_whole = dict()
    k = k_min
    # stop set, but the patterns need further checking with larger k values
    # allow dominance, but not between ancestors and descendants
    patterns_to_search_lowest_level = []
    # stop set, but its patterns don't need any further checking. They either have too small sizes, or are in the result set.
    # allow dominance, but not between ancestors and descendants
    patterns_no_need_searching = []
    # this dict stores all patterns, indexed by num2string(p)
    nodes_dict = dict()

    # DFS
    # this part is the main time consumption
    while len(S) > 0:
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        P = S.pop(0)
        st = num2string(P)
        # print("GraphTraverse, st = {}".format(st))
        if st == "0||||":
            print("GraphTraverse, st={}".format(st))
            print("\n")
        num_patterns_visited += 1
        whole_cardinality = pc_whole_data.pattern_count(st)
        patterns_size_whole[st] = whole_cardinality
        parent = findParent(P, num_att)
        parent_str = num2string(parent)
        if whole_cardinality < Thc:
            # We can only add itself into stop set,
            # since stop set doesn't allow dominance between ancestors and descendant
            CheckDominationAndAddForStopSetStr(st, patterns_no_need_searching)
            continue
        num_top_k = patterns_top_kmin.pattern_count(st)
        lowerbound = (whole_cardinality / data_size - alpha) * k
        upperbound = (whole_cardinality / data_size + alpha) * k
        if num_top_k < lowerbound or num_top_k > upperbound:
            CheckDominationAndAddForStopSetStr(st, patterns_no_need_searching)
            CheckDominationAndAddForLowerbound(P, pattern_treated_unfairly_lowerbound)
            Remove_descendants_str(st, patterns_to_search_lowest_level)
        else:
            children = GenerateChildren(P, whole_data_frame, attributes)
            if len(children) == 0:
                CheckDominationAndAddForStopSetStr(st, patterns_to_search_lowest_level)
                # if st not in set(patterns_searched_lowest_level_lowerbound):
                #     patterns_searched_lowest_level_lowerbound.append(st)
                # patterns_searched_lowest_level_lowerbound.add(st)
                # print("patterns_searched_lowest_level_lowerbound added {}".format(st))
                # print(patterns_searched_lowest_level_lowerbound)
            S = children + S
            # maintain sets for k values only for a node not in result set.
            # so now we add the this node to nodes_dict
            # smallest k before which lower bound is ok
            if whole_cardinality / data_size - alpha <= 0:
                smallest_valid_k = k_max + 1
            else:
                smallest_valid_k = math.floor(num_top_k / ((whole_cardinality / data_size) - alpha))
            if smallest_valid_k < k:
                raise Exception("smallest_valid_k_of_c < k")
            if st not in nodes_dict.keys():
                Add_node_to_set(nodes_dict, smallest_valid_k, P, st, parent, parent_str, root, root_str, num_att)
    print("finish kmin")
    # print("pattern_treated_unfairly_lowerkbound:", pattern_treated_unfairly_lowerbound)
    # for p in patterns_searched_lowest_level_lowerbound:
    #     print(p)
    for k in range(k_min + 1, k_max):
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        if PatternInSet([0, -1, 0, 0, -1], pattern_treated_unfairly_lowerbound):
            print("in, k={}".format(k))
        st = "0||0||"
        if st in nodes_dict.keys():
            print("before AddNewTuple, {} in nodes_dict".format(st))
            print("smallest_valid_k = {}".format(nodes_dict[st].smallest_valid_k))
        # if st in patterns_to_search_lowest_level:
        #     print("{} in stop set".format(st))
        patterns_top_k = pattern_count.PatternCounter(ranked_data[:k], encoded=False)
        patterns_top_k.parse_data()
        new_tuple = ranked_data.iloc[[k - 1]].values.flatten().tolist()
        print("k={}, new tuple={}".format(k, new_tuple))
        # print("k={}, new tuple = {}".format(k, new_tuple))
        # top down for related patterns, using similar methods as k_min, add to result set if needed
        # ancestors are patterns checked in AddNewTuple() function, to avoid checking them again
        ancestors, num_patterns_visited = AddNewTuple(new_tuple, Thc, pattern_treated_unfairly_lowerbound,
                                                      patterns_to_search_lowest_level, patterns_no_need_searching,
                                                      whole_data_frame, patterns_top_k, k, k_min, pc_whole_data,
                                                      num_patterns_visited,
                                                      patterns_size_whole, alpha, num_att,
                                                      data_size, nodes_dict, attributes)
        if st in nodes_dict.keys():
            print("after AddNewTuple, {} in nodes_dict, smallest_valid_k = {}".format(st, nodes_dict[st].smallest_valid_k))
        # print("patterns_searched_lowest_level_lowerbound:")
        # for p in patterns_searched_lowest_level_lowerbound:
        #     print(p)
        if PatternInSet([0, -1, 0, -1, -1], pattern_treated_unfairly_lowerbound):
            print("after AddNewTuple, in result set, k={}".format(k))
        num_patterns_visited, patterns_to_search_lowest_level, patterns_no_need_searching \
            = CheckCandidatesForKValues(nodes_dict, ancestors, patterns_to_search_lowest_level,
                                        patterns_no_need_searching,
                                        root, root_str,
                                        pattern_treated_unfairly_lowerbound, k,
                                        k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                                        alpha, num_att, whole_data_frame,
                                        attributes, num_patterns_visited, Thc, data_size)
        if PatternInSet([0, -1, 0, -1, -1], pattern_treated_unfairly_lowerbound):
            print("after CheckCandidatesForKValues, in result set, k={}".format(k))
    time1 = time.time()
    return pattern_treated_unfairly_lowerbound, num_patterns_visited, time1 - time0



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
def AddNewTuple(new_tuple, Thc, pattern_treated_unfairly_lowerbound,
                patterns_to_search_lowest_level, patterns_no_need_searching,
                whole_data_frame, patterns_top_k, k, k_min, pc_whole_data, num_patterns_visited,
                patterns_size_whole, alpha, num_att, data_size, nodes_dict, attributes):
    ancestors = []
    root = [-1] * num_att
    root_str = '|' * (num_att - 1)
    children = GenerateChildrenRelatedToTuple(root, new_tuple)  # pattern with one deterministic attribute
    S = children
    # if the k values increases, go to function () without generating children
    # otherwise, generating children and add children to queue
    while len(S) > 0:
        P = S.pop(0)
        st = num2string(P)
        # if st == "0||||" and k == 15:
        #     print("st={}".format(st))
        #     print("\n")
        # print("in addnewtuple, st={}".format(st))
        if P in pattern_treated_unfairly_lowerbound or st in patterns_no_need_searching:
            continue
        parent = findParent(P, num_att)
        parent_str = num2string(parent)
        num_patterns_visited += 1
        children = GenerateChildrenRelatedToTuple(P, new_tuple)
        if st in patterns_size_whole:
            whole_cardinality = patterns_size_whole[st]
        else:
            whole_cardinality = pc_whole_data.pattern_count(st)
        if whole_cardinality < Thc:
            raise Exception("whole_cardinality_of_child < Thc")
        else:
            # print("lower bound")
            # special case: this pattern itself is in the result set
            num_top_k = patterns_top_k.pattern_count(st)
            upperbound = (whole_cardinality / data_size + alpha) * k
            if num_top_k > upperbound:
                CheckDominationAndAddForLowerbound(P, pattern_treated_unfairly_lowerbound)
                CheckDominationAndAddForStopSetStr(st, patterns_no_need_searching)
                Remove_descendants_str(st, patterns_to_search_lowest_level)
                continue
            # smallest k before which lower bound is ok
            if whole_cardinality / data_size - alpha <= 0:
                smallest_valid_k = k_max + 1
            else:
                smallest_valid_k = math.floor(num_top_k / ((whole_cardinality / data_size) - alpha))
            if smallest_valid_k < k:
                raise Exception("smallest_valid_k_of_c < k")
            old_k = math.floor((num_top_k - 1) / ((whole_cardinality / data_size) - alpha))
            if old_k <= 0:
                old_k = k_max + 1
            # we need to check k values for this node
            if smallest_valid_k > old_k:
                # print("In addnewtuple, going to update_k_value")
                Update_k_value(nodes_dict, smallest_valid_k, P, st, parent, parent_str, root,
                               root_str, num_att, pattern_treated_unfairly_lowerbound,
                               patterns_to_search_lowest_level, patterns_no_need_searching,
                               whole_data_frame, attributes,
                               children, patterns_top_k, whole_cardinality, data_size, alpha, k,
                               patterns_size_whole, pc_whole_data, new_tuple)
            else:
                S = children + S
                ancestors = ancestors + children
    return ancestors, num_patterns_visited



def CheckCandidatesForKValues_patterns_to_search_lowest_level(nodes_dict, ancestors, patterns_to_search_lowest_level,
                                                              patterns_no_need_searching,
                             root, root_str,
                             pattern_treated_unfairly_lowerbound, k,
                             k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                             alpha, num_att, whole_data_frame,
                             attributes, num_patterns_visited, Thc, data_size):
    checked_patterns = set()
    for st in patterns_to_search_lowest_level:  # st is a string
        if st in checked_patterns:
            continue
        checked_patterns.add(st)
        # print("CheckCandidatesForKValues, st={}".format(st))
        # if st == "0|0|||" and k == 15:
        #     print("CheckCandidatesForKValues_patterns_to_search_lowest_level, st = {}".format(st))
        num_patterns_visited += 1
        p = string2num(st)
        if p in ancestors or p in pattern_treated_unfairly_lowerbound:  # already checked
            continue
        if st in patterns_size_whole:
            whole_cardinality = patterns_size_whole[st]
        else:
            whole_cardinality = pc_whole_data.pattern_count(st)
        if whole_cardinality < Thc: # TODO: this is impossible??
            raise Exception("whole_cardinality_of_child < Thc")
            # num_patterns_visited += 1
            # # should go up to ancestors
            # st = findParentForStr(st)
            # if st == root_str:
            #     continue
        find, nearest_ancestor_str = Find_closest_ancestor(nodes_dict.keys(), st, num_att)
        if find:
            nearest_ancestor_node = nodes_dict[nearest_ancestor_str]
        else:
            raise Exception('ancestors {} in nodes_dict not found'.format(st))
        checked_patterns.add(nearest_ancestor_str)
        if nearest_ancestor_node.smallest_valid_k >= k:
            continue
        # nearest_ancestor.smallest_valid_k < k, we need to find whether it has ancestor with the same k
        # and then that should be added to result set
        while nearest_ancestor_node.smallest_ancestor != "":
            nearest_ancestor_node = nodes_dict[nearest_ancestor_node.smallest_ancestor]
            nodes_dict.pop(nearest_ancestor_str)
            nearest_ancestor_str = nearest_ancestor_node.st
            checked_patterns.add(nearest_ancestor_str)
        child_str = nearest_ancestor_str
        child = nearest_ancestor_node
        CheckDominationAndAddForLowerbound(child.pattern, pattern_treated_unfairly_lowerbound)
        CheckDominationAndAddForStopSetStr(child_str, patterns_no_need_searching)
        Remove_descendants_str(child_str, patterns_to_search_lowest_level)
        if child_str == "0||0||":
            print("{} added to result set".format(child_str))
    return checked_patterns



def CheckCandidatesForKValues_patterns_no_need_searching(nodes_dict, ancestors, patterns_to_search_lowest_level,
                                                         patterns_no_need_searching,
                             root, root_str,
                             pattern_treated_unfairly_lowerbound, k,
                             k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                             alpha, num_att, whole_data_frame,
                             attributes, num_patterns_visited, Thc, data_size, checked_patterns):
    for st in patterns_no_need_searching:  # st is a string
        if st in checked_patterns:
            continue
        parent_str = findParentForStr(st)
        if parent_str in checked_patterns or parent_str == root_str:
            continue
        checked_patterns.add(parent_str)
        parent = string2num(parent_str)
        if parent in ancestors:
            continue
        find, nearest_ancestor_str = Find_closest_ancestor(nodes_dict.keys(), parent_str, num_att)
        if find:
            nearest_ancestor_node = nodes_dict[nearest_ancestor_str]
        else:
            raise Exception('ancestors {} in nodes_dict not found'.format(st))
        checked_patterns.add(nearest_ancestor_str)
        if nearest_ancestor_node.smallest_valid_k >= k:
            continue
        # to_remove.add(st) # TODO: if each time we check dominance for patterns_no_need_searching, we don't need to_remove in this function!
        # nearest_ancestor.smallest_valid_k < k, we need to find whether it has ancestor with the same k
        # and then that should be added to result set
        while nearest_ancestor_node.smallest_ancestor != "":
            nearest_ancestor_node = nodes_dict[nearest_ancestor_node.smallest_ancestor]
            nodes_dict.pop(nearest_ancestor_str)
            nearest_ancestor_str = nearest_ancestor_node.st
            checked_patterns.add(nearest_ancestor_str)

        child_str = nearest_ancestor_str
        child = nearest_ancestor_node
        CheckDominationAndAddForLowerbound(child.pattern, pattern_treated_unfairly_lowerbound)
        CheckDominationAndAddForStopSetStr(child_str, patterns_no_need_searching)
        Remove_descendants_str(child_str, patterns_to_search_lowest_level)


# check whether k values exceeds the smallest k for a pattern
# we check all patterns in the stop set, and go up
def CheckCandidatesForKValues(nodes_dict, ancestors, patterns_to_search_lowest_level, patterns_no_need_searching,
                             root, root_str,
                             pattern_treated_unfairly_lowerbound, k,
                             k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                             alpha, num_att, whole_data_frame,
                             attributes, num_patterns_visited, Thc, data_size):
    checked_patterns = CheckCandidatesForKValues_patterns_to_search_lowest_level(nodes_dict, ancestors,
                                                                                 patterns_to_search_lowest_level,
                                                                                    patterns_no_need_searching,
                             root, root_str,
                             pattern_treated_unfairly_lowerbound, k,
                             k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                             alpha, num_att, whole_data_frame,
                             attributes, num_patterns_visited, Thc, data_size)

    CheckCandidatesForKValues_patterns_no_need_searching(nodes_dict, ancestors, patterns_to_search_lowest_level,
                                                         patterns_no_need_searching,
                                                         root, root_str,
                                                         pattern_treated_unfairly_lowerbound, k,
                                                         k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                                                         alpha, num_att, whole_data_frame,
                                                         attributes, num_patterns_visited, Thc, data_size,
                                                         checked_patterns)


    return num_patterns_visited, patterns_to_search_lowest_level, patterns_no_need_searching


all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
                  'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C',
                  'failures_C', 'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C', 'higher_C',
                  'internet_C', 'romantic_C', 'famrel_C', 'freetime_C', 'goout_C', 'Dalc_C', 'Walc_C',
                  'health_C', 'absences_C', 'G1_C', 'G2_C', 'G3_C']

selected_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
                       'Fedu_C', 'Mjob_C', 'Fjob_C']

"""
with the above 19 att,
naive: 98s num_patterns_visited = 2335488
optimized: 124s num_patterns_visited = 299559
num of pattern_treated_unfairly_lowerbound = 85, num of pattern_treated_unfairly_upperbound = 18
"""

original_data_file = r"../../InputData/StudentDataset/ForRanking_1/student-mat_cat_ranked.csv"


ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]


time_limit = 5 * 60
k_min = 50
k_max = 150
Thc = 30

List_k = list(range(k_min, k_max))

alpha = 0.1


print("start the new alg")

pattern_treated_unfairly_lowerbound, num_patterns_visited, running_time = \
    GraphTraverse(ranked_data, selected_attributes, Thc,
                 alpha,
                 k_min, k_max, time_limit)

print("num_patterns_visited = {}".format(num_patterns_visited))
print("time = {} s, num of pattern_treated_unfairly_lowerbound = {}".format(running_time,
        len(pattern_treated_unfairly_lowerbound)), "\n", "patterns:\n",
      pattern_treated_unfairly_lowerbound)

print("dominated by pattern_treated_unfairly_lowerbound:")
for p in pattern_treated_unfairly_lowerbound:
    if PDominatedByM(p, pattern_treated_unfairly_lowerbound)[0]:
        print(p)





print("start the naive alg")

pattern_treated_unfairly_lowerbound2, num_patterns_visited2, running_time2 = \
    naiveranking.NaiveAlg(ranked_data, selected_attributes, Thc,
                                                             alpha,
                                                             k_min, k_max, time_limit)


print("num_patterns_visited = {}".format(num_patterns_visited2))
print("time = {} s, num of pattern_treated_unfairly_lowerbound = {}".format(running_time2,
        len(pattern_treated_unfairly_lowerbound2)), "\n", "patterns:\n",
      pattern_treated_unfairly_lowerbound2)



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




