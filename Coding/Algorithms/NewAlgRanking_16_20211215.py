"""
New algorithm for group detection in ranking
fairness definition: the number of a group members in top-k is bounded by U_k, L_k, k_min <= k
bounds for different k can be different, but all patterns have the same bounds
This is the final algorithm for this fairness definition in ranking


Go top-down, find two result sets: for lower bound
For lower bound: most general pattern

This script is different from NewAlgRanking_10_20210702.py:
1. we only do lower bound
2. we check stop set and patterns related to new tuple
3. stop set is where we stopped not its parents




"""

import pandas as pd

from itertools import combinations
from Algorithms import pattern_count
import time
from Algorithms import Predict_0_20210127 as predict
from Algorithms import NaiveAlgRanking_4_20211213 as naiveranking


def TSatisfiesP(t, p, num_att):
    for i in range(num_att):
        if p[i] == -1:
            continue
        else:
            if p[i] != t[i]:
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



def CheckDominationAndAddForLowerbound_topdown_search(pattern, pattern_treated_unfairly):
    to_remove = []
    for p in pattern_treated_unfairly:
        # if PatternEqual(p, pattern):
        #     return
        if P1DominatedByP2(pattern, p):
            return False
    pattern_treated_unfairly.append(pattern)
    return True




def AddToBackup(pattern, dominated_by_lowerbound_result, second_backup):
    if pattern in dominated_by_lowerbound_result or pattern in second_backup:
        return
    move_to_second = []
    for p in dominated_by_lowerbound_result:
        if P1DominatedByP2(pattern, p):
            second_backup.append(pattern)
            return
        elif P1DominatedByP2(p, pattern):
            move_to_second.append(p)
    for p in move_to_second:
        dominated_by_lowerbound_result.remove(p)
        if p not in second_backup:
            second_backup.append(p)
    dominated_by_lowerbound_result.append(pattern)


def RemoveFromBackup(pattern, dominated_by_lowerbound_result, second_backup):
    if pattern in second_backup:
        second_backup.remove(pattern)
        return True
    elif pattern in dominated_by_lowerbound_result:
        dominated_by_lowerbound_result.remove(pattern)
        remove_from_second_backup = []
        for s in second_backup:
            if P1DominatedByP2(s, pattern):
                if PDominatedByM(s, dominated_by_lowerbound_result)[0] is False:
                    level_up = True
                    remove_back = []
                    for r in remove_from_second_backup:
                        if P1DominatedByP2(s, r):
                            level_up = False
                            break
                        elif P1DominatedByP2(r, s):
                            remove_back.append(r)
                    if level_up:
                        dominated_by_lowerbound_result.append(s)
                        remove_from_second_backup.append(s)
                        for t in remove_back:
                            remove_from_second_backup.remove(t)
                            dominated_by_lowerbound_result.remove(t)
        for t in remove_from_second_backup:
            second_backup.remove(t)
        return True
    return False


# only need to check the lower bound of parents
# need to check patterns_should_be_in_result_set, and patterns_no_children
def CheckCandidatesForBounds(ancestors, patterns_should_be_in_result_set,
                             patterns_small_size, patterns_no_children,
                             root, root_str,
                             result_set_lowerbound, k,
                             k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                             Lowerbounds, num_att, whole_data_frame,
                             attributes, num_patterns_visited, Thc):
    to_remove_from_patterns_should_be_in_result_set = []
    to_append_to_to_remove_from_patterns_should_be_in_result_set = []
    checked_patterns = set()
    # st = "|0||0|"
    # if st in patterns_searched_lowest_level_lowerbound:
    #     print("in CheckCandidatesForBounds, {} in stop set".format(st))
    for st in patterns_should_be_in_result_set:  # st is a string
        if st in checked_patterns:
            continue
        checked_patterns.add(st)
        p = string2num(st)
        # if PatternEqual(p, [-1, 0, -1, 0, -1]):
        #     print("CheckCandidatesForBounds, p = {}".format(p))
        if p in ancestors:  # already checked
            continue
        num_patterns_visited += 1
        pattern_size_in_topk = patterns_top_k.pattern_count(st)
        if pattern_size_in_topk >= Lowerbounds[k - k_min]:
            if st in result_set_lowerbound:
                result_set_lowerbound.remove(st)
            to_remove_from_patterns_should_be_in_result_set.append(st)
            if p[num_att - 1] == -1:
                raise Exception("why is this pattern {} in stop set ???".format(p))
            continue
        # if pattern_size_in_topk < Lowerbounds[k - k_min], remove child and add this parent
        # why do you only care about the parent generating this child, but not other parents?
        # because other parents will be handled by the child it generates by itself
        child_str = st
        parent_str = findParentForStr(child_str)
        child = p
        if parent_str == root_str:
            CheckDominationAndAddForLowerbound(child, result_set_lowerbound)
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
                CheckDominationAndAddForLowerbound(child, result_set_lowerbound)
                if st != child_str:
                    to_remove_from_patterns_should_be_in_result_set.append(st)
                    to_append_to_to_remove_from_patterns_should_be_in_result_set.append(child_str)
                break
        if parent_str == root_str:
            CheckDominationAndAddForLowerbound(child, result_set_lowerbound)
            to_append_to_to_remove_from_patterns_should_be_in_result_set.append(child_str)
            continue
    for p_str in to_remove_from_patterns_should_be_in_result_set:
        patterns_should_be_in_result_set.remove(p_str)
    patterns_should_be_in_result_set = patterns_should_be_in_result_set + \
                                       to_append_to_to_remove_from_patterns_should_be_in_result_set
    to_remove_from_patterns_no_children = []
    to_append_to_to_remove_from_patterns_no_children = []
    for st in patterns_no_children:
        if st in checked_patterns:
            continue
        checked_patterns.add(st)
        p = string2num(st)
        if p in ancestors:  # already checked
            continue
        num_patterns_visited += 1
        pattern_size_in_topk = patterns_top_k.pattern_count(st)
        if pattern_size_in_topk >= Lowerbounds[k - k_min]:
            continue
        child_str = st
        parent_str = findParentForStr(child_str)
        child = p
        if parent_str == root_str:
            to_remove_from_patterns_no_children.append(st)
            CheckDominationAndAddForLowerbound(child, result_set_lowerbound)
            patterns_should_be_in_result_set.append(st)
            continue
        checked_patterns.add(parent_str)
        while parent_str != root_str:
            num_patterns_visited += 1
            pattern_size_in_topk = patterns_top_k.pattern_count(parent_str)
            if pattern_size_in_topk < Lowerbounds[k - k_min]:
                child_str = parent_str
                parent_str = findParentForStr(child_str)
                checked_patterns.add(parent_str)
            else:
                CheckDominationAndAddForLowerbound(child, result_set_lowerbound)
                if child_str not in patterns_should_be_in_result_set:
                    patterns_should_be_in_result_set.append(child_str)
                to_remove_from_patterns_no_children.append(st)
                break
        if parent_str == root_str:
            CheckDominationAndAddForLowerbound(child, result_set_lowerbound)
            if child_str not in patterns_should_be_in_result_set:
                patterns_should_be_in_result_set.append(child_str)
            to_remove_from_patterns_no_children.append(st)
            continue
    for st in to_remove_from_patterns_no_children:
        patterns_no_children.remove(st)
    return num_patterns_visited


# when bound doesn't change, check patterns_should_be_in_result_set only, don't go up
def CheckStopSet(ancestors, patterns_should_be_in_result_set,
                 patterns_small_size, patterns_no_children,
                 root, root_str,
                 result_set_lowerbound, k,
                 k_min, pc_whole_data, patterns_top_k, patterns_size_whole,
                 Lowerbounds, num_att, whole_data_frame,
                 attributes, num_patterns_visited, Thc):
    to_remove = set()
    to_append = []
    # st = "|0||0|"
    # if st in patterns_searched_lowest_level_lowerbound:
    #     print("in CheckCandidatesForBounds, {} in stop set".format(st))
    for st in patterns_should_be_in_result_set:  # st is a string
        p = string2num(st)
        if p in ancestors or p in result_set_lowerbound:  # already checked
            continue
        num_patterns_visited += 1
        pattern_size_in_topk = patterns_top_k.pattern_count(st)
        if pattern_size_in_topk >= Lowerbounds[k - k_min]:
            if p[num_att - 1] == -1:
                raise Exception("why is this pattern {} in stop set ???".format(p))
            continue
        # this node is below lower bound
        CheckDominationAndAddForLowerbound(p, result_set_lowerbound)
    return num_patterns_visited


def GraphTraverse(ranked_data, attributes, Thc, Lowerbounds, k_min, k_max, time_limit):
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
    patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k_min], encoded=False)
    patterns_top_kmin.parse_data()
    patterns_size_whole = dict()
    k = k_min
    result_set_lowerbound = []
    # stop reasons:
    # 1. add to result set
    # 2. if size too small, add its parent
    # 3. no children
    patterns_dominated_by_result = []  # doesn't include patterns in result set
    patterns_children_small_size = []
    patterns_no_children = []

    # DFS
    # this part is the main time consumption
    while len(S) > 0:
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        P = S.pop(0)
        if PatternEqual(P, [0, 0, -1, 0]):
            print("k={}, pattern equal = {}".format(k, P))
            print("\n")
        st = num2string(P)
        num_patterns_visited += 1
        whole_cardinality = pc_whole_data.pattern_count(st)
        patterns_size_whole[st] = whole_cardinality
        if whole_cardinality < Thc:
            parent_str = findParentForStr(st)
            if parent_str != root_str:
                patterns_children_small_size.append(parent_str)
            continue
        num_top_k = patterns_top_kmin.pattern_count(st)
        if num_top_k < Lowerbounds[k - k_min]:
            if not CheckDominationAndAddForLowerbound_topdown_search(P, result_set_lowerbound):
                patterns_dominated_by_result.append(st)
        else:
            if P[num_att - 1] != -1:  # no children
                patterns_no_children.append(st)
            else:
                children = GenerateChildren(P, whole_data_frame, attributes)
                S = S + children

    pattern_treated_unfairly_lowerbound.append(result_set_lowerbound)
    # print("time for k_min = {}".format(time.time() - time0))
    # print("num pattern visited in k_min = {}".format(num_patterns_visited))
    for k in range(k_min + 1, k_max):
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        result_set_lowerbound = result_set_lowerbound.copy()
        patterns_top_k = pattern_count.PatternCounter(ranked_data[:k], encoded=False)
        patterns_top_k.parse_data()
        new_tuple = ranked_data.iloc[[k - 1]].values.flatten().tolist()
        print("k={}, new tuple = {}".format(k, new_tuple))

        if Lowerbounds[k - k_min] > Lowerbounds[k - 1 - k_min]:
            time3 = time.time()
            # TODO: what if the bound changes
            checked_patterns = []

            to_remove = set()

            for p in result_set_lowerbound:
                if p in checked_patterns:
                    continue
                else:
                    checked_patterns.append(p)
                num_patterns_visited += 1
                st = num2string(p)
                if TSatisfiesP(new_tuple, p, num_att):
                    child = p
                    parent = findParent(child, num_att)
                    up = False
                    while PatternEqual(parent, root) is False:
                        checked_patterns.append(parent)
                        num_top_k_parent = patterns_top_k.pattern_count(num2string(parent))
                        if num_top_k_parent < Lowerbounds[k - k_min]:
                            up = True
                            patterns_dominated_by_result.append(num2string(child))
                            child = parent
                            parent = findParent(child, num_att)
                        else:
                            if up:
                                to_remove.add(p)
                                for q in result_set_lowerbound:
                                    if P1DominatedByP2(child, q):
                                        patterns_dominated_by_result.append(num2string(child))
                                        break
                                    elif P1DominatedByP2(q, child):
                                        to_remove.add(q)
                            break
            for s in to_remove:
                result_set_lowerbound.remove(s)
                patterns_dominated_by_result.append(num2string(s))
            to_remove = set()
            to_remove_from_resultset = []
            for st in patterns_no_children:
                p = string2num(st)
                if p in checked_patterns:
                    continue
                else:
                    checked_patterns.append(p)
                num_top_k = patterns_top_k.pattern_count(st)
                old_str = st
                new_str = st
                lower = False
                while num_top_k < Lowerbounds[k - k_min]:
                    lower = True
                    old_str = new_str
                    new_str = findParentForStr(new_str)
                    new_p = string2num(new_str)
                    if new_p in checked_patterns:
                        break
                    else:
                        checked_patterns.append(new_p)
                    num_top_k = patterns_top_k.pattern_count(new_str)
                    if new_str == root_str:
                        break
                if lower:
                    to_remove.add(st)
                    new_p = string2num(old_str)
                    to_remove_from_resultset = []
                    can_add = True
                    for q in result_set_lowerbound:
                        if P1DominatedByP2(new_p, q):
                            patterns_dominated_by_result.append(num2string(new_p))
                            can_add = False
                            break
                        elif P1DominatedByP2(q, new_p):
                            to_remove_from_resultset.append(q)
                    if can_add:
                        result_set_lowerbound.append(new_p)
                        for pa in to_remove_from_resultset:
                            result_set_lowerbound.remove(pa)
                            patterns_dominated_by_result.append(num2string(pa))
            for s in to_remove:
                patterns_no_children.remove(s)

            to_remove = set()
            to_remove_from_resultset = []
            for st in patterns_children_small_size:
                num_top_k = patterns_top_k.pattern_count(st)
                old_str = st
                new_str = st
                new_p = string2num(new_str)
                if new_p in checked_patterns:
                    continue
                else:
                    checked_patterns.append(new_p)
                lower = False
                while num_top_k < Lowerbounds[k - k_min]:
                    lower = True
                    old_str = new_str
                    new_str = findParentForStr(new_str)
                    new_p = string2num(new_str)
                    if new_p in checked_patterns:
                        break
                    else:
                        checked_patterns.append(new_p)
                    num_top_k = patterns_top_k.pattern_count(new_str)
                    if new_str == root_str:
                        break
                if lower:
                    to_remove.add(st)
                    new_p = string2num(old_str)
                    to_remove_from_resultset = []
                    can_add = True
                    for q in result_set_lowerbound:
                        if P1DominatedByP2(new_p, q):
                            patterns_dominated_by_result.append(num2string(new_p))
                            can_add = False
                            break
                        elif P1DominatedByP2(q, new_p):
                            to_remove_from_resultset.append(q)
                    if can_add:
                        result_set_lowerbound.append(new_p)
                        for pa in to_remove_from_resultset:
                            result_set_lowerbound.remove(pa)
                            patterns_dominated_by_result.append(num2string(pa))
            for s in to_remove:
                patterns_children_small_size.remove(s)
            for q in to_remove_from_resultset:
                result_set_lowerbound.remove(q)

            time4 = time.time()
            # print("time for CheckCandidatesForBounds = {}".format(time4 - time3))
        else:
            # the bound doesn't change
            # check resul set
            to_remove = []
            to_append = []
            to_append_patterns_dominated_by_result = set()
            to_append_patterns_no_children = set()
            for p in result_set_lowerbound:
                num_patterns_visited += 1
                # if PatternEqual(p, [0, 0, 0, -1, -1, 0, -1, -1, -1, -1]):
                #     print("checking result_set_lowerbound, p={}".format(p))
                st = num2string(p)
                if TSatisfiesP(new_tuple, p, num_att):
                    num_top_k = patterns_top_k.pattern_count(st)
                    if num_top_k >= Lowerbounds[k - k_min]:
                        to_remove.append(p)
                        if p[num_att - 1] != -1:
                            to_append_patterns_no_children.add(st)
                        else:
                        # go down from here
                            num_patterns_visited1 = GoDownForResultSet(p, st, k, patterns_top_k, result_set_lowerbound,
                                                       patterns_size_whole, pc_whole_data, whole_data_frame,
                                                       attributes, Thc, Lowerbounds[k - k_min], num_att,
                                                       to_append_patterns_dominated_by_result, patterns_children_small_size,
                                                       to_append_patterns_no_children, 0, to_append, to_remove)
                            num_patterns_visited += num_patterns_visited1
            for p in to_remove:
                result_set_lowerbound.remove(p)
            for p in to_append:
                result_set_lowerbound.append(p)

            # don't need to check patterns_children_small_size or patterns_no_children
            # check patterns_dominated_by_result
            to_remove = []
            to_append = []
            for st in patterns_dominated_by_result:
                num_patterns_visited += 1
                num_top_k = patterns_top_k.pattern_count(st)
                p = string2num(st)
                # if PatternEqual(p, [0, -1, -1, -1, -1, 0]):
                #     print("ssss")
                # if st == '0||0|||0||||0':
                #     print("checking result_set_lowerbound, st={}".format(st))
                if num_top_k >= Lowerbounds[k - k_min]:
                    to_remove.append(st)
                    if p[num_att - 1] != -1:
                        to_append_patterns_no_children.add(st)
                    else:
                        num_patterns_visited2 = GoDownForDominatedByResult(p, st, k, patterns_top_k, result_set_lowerbound,
                                                       patterns_size_whole, pc_whole_data, whole_data_frame,
                                                       attributes, Thc, Lowerbounds[k - k_min], num_att,
                                                       patterns_dominated_by_result, patterns_children_small_size,
                                                       to_append_patterns_no_children, 0, to_append)
                        num_patterns_visited += num_patterns_visited2
                else:
                    to_remove_from_resultset = []
                    can_add = True
                    for r in result_set_lowerbound:
                        # if PatternEqual(p, pattern):
                        if P1DominatedByP2(p, r):
                            can_add = False
                            break
                        elif P1DominatedByP2(r, p):
                            to_remove_from_resultset.append(r)
                    if can_add:
                        result_set_lowerbound.append(p)
                        to_remove.append(st)
                        for s in to_remove_from_resultset:
                            result_set_lowerbound.remove(s)
                            to_append.append(num2string(s))
            for st in to_remove:
                patterns_dominated_by_result.remove(st)
            for st in to_append:
                patterns_dominated_by_result.append(st)
            for st in to_append_patterns_dominated_by_result:
                patterns_dominated_by_result.append(st)
            for st in to_append_patterns_no_children:
                patterns_no_children.append(st)

        # print("k = {}, number of pattern visited: {}, {}".format(k, num_patterns_visited_addnewtuple,
        #                                                  num_patterns_visited_checkbound))

        pattern_treated_unfairly_lowerbound.append(result_set_lowerbound)
        # if p in patterns_searched_lowest_level_lowerbound:
        #     print("after CheckCandidatesForBounds {} in stop set".format(p))
        # if p in result_set_lowerbound:
        #     print("after CheckCandidatesForBounds {} in result set".format(p))
    time1 = time.time()
    return pattern_treated_unfairly_lowerbound, num_patterns_visited, time1 - time0


# search top-down to go over all patterns related to new_tuple
# using similar checking methods as k_min
# add to result set if they are outliers
# def AddNewTuple(new_tuple, Thc, result_set_lowerbound,
#                 whole_data_frame, patterns_top_k, k, k_min, pc_whole_data,
#                 num_patterns_visited,
#                 patterns_size_whole, Lowerbounds, num_att,
#                 attributes,
#                 patterns_should_be_in_result_set,
#                 patterns_small_size, patterns_no_children):
#     root = [-1] * num_att
#     children = GenerateChildrenRelatedToTuple(root, new_tuple)  # pattern with one deternimistic attribute
#     S = children
#     ancestors = S
#     new_tuple_only = [True] * len(S)
#     while len(S) > 0:
#         time1 = time.time()
#         P = S.pop(0)
#         nto = new_tuple_only.pop()
#         st = num2string(P)
#         # if PatternEqual(P, [-1, -1, 0, -1, -1, -1]):
#         #     print("AddNewTuple, P={}".format(P))
#         #     print("\n")
#         num_patterns_visited += 1
#         children = []
#         if st in patterns_size_whole:
#             whole_cardinality = patterns_size_whole[st]
#         else:
#             whole_cardinality = pc_whole_data.pattern_count(st)
#         if whole_cardinality < Thc:
#             continue
#         else:
#             num_top_k = patterns_top_k.pattern_count(st)
#             time2 = time.time()
#             # print("time2-1 = {}".format(time2 - time1))
#             if num_top_k < Lowerbounds[k - k_min]:
#                 if P not in result_set_lowerbound:
#                     CheckDominationAndAddForLowerbound(P, result_set_lowerbound)
#                 if st not in patterns_should_be_in_result_set:
#                     patterns_should_be_in_result_set.append(st)
#                 if st in patterns_no_children:
#                     patterns_no_children.remove(st)
#                 continue
#             else:
#                 how_to_generate = []
#                 if P in result_set_lowerbound:
#                     result_set_lowerbound.remove(P)
#                     patterns_should_be_in_result_set.remove(st)
#                     if P[num_att - 1] == -1:  # has children
#                         children = GenerateChildren(P, whole_data_frame, attributes)
#                         how_to_generate = [False] * len(children)
#                     else:
#                         patterns_no_children.append(st)
#                 else:
#                     if P[num_att - 1] != -1:  # no children
#                         if st not in patterns_no_children:
#                             patterns_no_children.append(st)
#                     else:
#                         if st in patterns_should_be_in_result_set:  # < lower bound last time
#                             children = GenerateChildren(P, whole_data_frame, attributes)
#                             how_to_generate = [False] * len(children)
#                             patterns_should_be_in_result_set.remove(st)
#                         elif nto:
#                             children = GenerateChildrenRelatedToTuple(P, new_tuple)
#                             how_to_generate = [True] * len(children)
#                         else:
#                             children = GenerateChildren(P, whole_data_frame, attributes)
#                             how_to_generate = [False] * len(children)
#                 if len(children) != 0:
#                     S = S + children
#                     new_tuple_only = new_tuple_only + how_to_generate
#                     ancestors = ancestors + children
#                 time3 = time.time()
#                 # print("time2-3 = {}".format(time3 - time2))
#     return ancestors, num_patterns_visited
#

# go down to p's all descendants
def GoDownForResultSet(p, st, k, patterns_top_k, result_set_lowerbound, patterns_size_whole, pc_whole_data, whole_data_frame,
           attributes, Thc, lowerbound, num_att,
           to_append_patterns_dominated_by_result, patterns_children_small_size, to_append_patterns_no_children,
           num_patterns_visited, to_append, to_remove):
    S = GenerateChildren(p, whole_data_frame, attributes)
    while len(S) > 0:
        P = S.pop(0)
        # if PatternEqual(P, [0, -1, 0, -1]):
        #     print("xxx")
        st = num2string(P)
        num_patterns_visited += 1
        whole_cardinality = pc_whole_data.pattern_count(st)
        patterns_size_whole[st] = whole_cardinality
        if whole_cardinality < Thc:
            patterns_children_small_size.append(findParentForStr(st))
            continue
        num_top_k = patterns_top_k.pattern_count(st)
        if num_top_k < lowerbound:
            dominated = False
            for r in result_set_lowerbound:
                if r in to_remove:
                    continue
                if P1DominatedByP2(P, r):
                    dominated = True
                    to_append_patterns_dominated_by_result.add(st)
                    break
            if not dominated:
                for s in to_append:
                    if P1DominatedByP2(P, s):
                        dominated = True
                        to_append_patterns_dominated_by_result.add(st)
                        break
                if not dominated:
                    to_append.append(P)
        else:
            if P[num_att - 1] != -1:  # no children
                to_append_patterns_no_children.add(st)
            else:
                children = GenerateChildren(P, whole_data_frame, attributes)
                S = S + children
    return num_patterns_visited


# go down to p's all descendants
def GoDownForDominatedByResult(p, st, k, patterns_top_k, result_set_lowerbound, patterns_size_whole, pc_whole_data, whole_data_frame,
           attributes, Thc, lowerbound, num_att,
           patterns_dominated_by_result, patterns_children_small_size, to_append_patterns_no_children,
           num_patterns_visited, to_append):
    S = GenerateChildren(p, whole_data_frame, attributes)
    while len(S) > 0:
        P = S.pop(0)
        st = num2string(P)
        num_patterns_visited += 1
        whole_cardinality = pc_whole_data.pattern_count(st)
        patterns_size_whole[st] = whole_cardinality
        if whole_cardinality < Thc:
            patterns_children_small_size.append(findParentForStr(st))
            continue
        num_top_k = patterns_top_k.pattern_count(st)
        if num_top_k < lowerbound:
            if CheckDominationAndAddForLowerbound_topdown_search(P, result_set_lowerbound) is False:
                to_append.append(st)
        else:
            if P[num_att - 1] != -1:  # no children
                to_append_patterns_no_children.add(st)
            else:
                children = GenerateChildren(P, whole_data_frame, attributes)
                S = S + children
    return num_patterns_visited






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
# original_data_file = r"../../InputData/StudentDataset/ForRanking_1/student-mat_cat_ranked.csv"


all_attributes = ["age_binary", "sex_binary", "race_C", "MarriageStatus_C", "juv_fel_count_C",
                  "decile_score_C", "juv_misd_count_C", "juv_other_count_C", "priors_count_C",
                  "days_b_screening_arrest_C",
                  "c_days_from_compas_C", "c_charge_degree_C", "v_decile_score_C", "start_C", "end_C",
                  "event_C"]

original_data_file = r"../../InputData/CompasData/general/compas_data_cat_necessary_att_ranked.csv"
# original_data_file = "../../InputData/CompasData/ForRanking/LargeDatasets/8000.csv"

# with 14 attributes, naive alg over time, new alg needs 117 s
selected_attributes = all_attributes[:4]

ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]

time_limit = 10 * 60
k_min = 5
k_max = 20
Thc = 50

List_k = list(range(k_min, k_max))


def lowerbound(x):
    return int((x+7)/2)


Lowerbounds = [lowerbound(x) for x in List_k]

print(Lowerbounds)

print("start the new alg")

pattern_treated_unfairly_lowerbound, num_patterns_visited, running_time = \
    GraphTraverse(ranked_data, selected_attributes, Thc,
                  Lowerbounds,
                  k_min, k_max, time_limit)

print("num_patterns_visited = {}".format(num_patterns_visited))
print("time = {} s".format(running_time))
# for k in range(0, k_max - k_min):
#     print("k = {}, num = {}, patterns =".format(k + k_min, len(pattern_treated_unfairly_lowerbound[k])),
#           pattern_treated_unfairly_lowerbound[k])

print("start the naive alg")

pattern_treated_unfairly_lowerbound2, \
num_patterns_visited2, running_time2 = naiveranking.NaiveAlg(ranked_data, selected_attributes, Thc,
                                                             Lowerbounds,
                                                             k_min, k_max, time_limit)

print("num_patterns_visited = {}".format(num_patterns_visited2))
print("time = {} s".format(running_time2))
# for k in range(0, k_max - k_min):
#     print("k = {}, num = {}, patterns =".format(k + k_min, len(pattern_treated_unfairly_lowerbound2[k])),
#           pattern_treated_unfairly_lowerbound2[k])


k_printed = False
print("p in pattern_treated_unfairly_lowerbound but not in pattern_treated_unfairly_lowerbound2:")
for k in range(0, k_max - k_min):
    for p in pattern_treated_unfairly_lowerbound[k]:
        if p not in pattern_treated_unfairly_lowerbound2[k]:
            if k_printed is False:
                print("k=", k + k_min)
                k_printed = True
            print(p)

k_printed = False
print("p in pattern_treated_unfairly_lowerbound2 but not in pattern_treated_unfairly_lowerbound:")
for k in range(0, k_max - k_min):
    for p in pattern_treated_unfairly_lowerbound2[k]:
        if p not in pattern_treated_unfairly_lowerbound[k]:
            if k_printed is False:
                print("k=", k + k_min)
                k_printed = True
            print(p)
