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

"""

import pandas as pd

from itertools import combinations
from Algorithms import pattern_count
import time
from Algorithms import Predict_0_20210127 as predict
from Algorithms import NaiveAlgRanking_1_20210611 as naiveranking


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
Tha: threshold of accuracy 
Thc: threshold of cardinality
"""

def findParent(child, length):
    parent = child.copy()
    for i in range(length-1, -1, -1):
        if parent[i] != -1:
            parent[i] = -1
            break
    return parent

def GraphTraverse(ranked_data, attributes, Thc, Lowerbounds, Upperbounds, k_min, k_max, time_limit):
    # print("attributes:", attributes)
    time1 = time.time()

    pc_whole_data = pattern_count.PatternCounter(ranked_data, encoded=False)
    pc_whole_data.parse_data()

    whole_data_frame = ranked_data.describe(include='all')

    num_patterns_visited = 0
    num_att = len(attributes)
    root = [-1] * num_att
    children = GenerateChildren(root, whole_data_frame, attributes)
    S = children
    pattern_treated_unfairly_lowerbound = []
    pattern_treated_unfairly_upperbound = []
    patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k_min], encoded=False)
    patterns_top_kmin.parse_data()
    patterns_size_topk = dict()
    patterns_size_whole = dict()
    k = k_min
    patterns_searched_lowest_level_lowerbound = []
    patterns_searched_lowest_level_upperbound = []

    parent_candidate_for_upperbound = []
    # DFS
    while len(S) > 0:
        if time.time() - time1 > time_limit:
            print("newalg overtime")
            break
        P = S.pop()
        # if PatternEqual(P, [-1, -1, 2, 1]):
        #     print("pattern equal ".format(P))
        st = num2string(P)
        num_patterns_visited += 1

        whole_cardinality = pc_whole_data.pattern_count(st)
        patterns_size_whole[st] = whole_cardinality
        if whole_cardinality < Thc:
            parent = findParent(P, num_att)
            # patterns in patterns_searched_lowest_level all have valid whole cardinality
            # and are not in pattern_treated_unfairly
            if PatternEqual(parent, root) is False:
                patterns_searched_lowest_level_lowerbound.append(parent)
                patterns_searched_lowest_level_upperbound.append(parent)
                # TODO : upper bound ??
            continue

        num_top_k = patterns_top_kmin.pattern_count(st)
        patterns_size_topk[st] = num_top_k
        if num_top_k < Lowerbounds[k - k_min]:
            parent = findParent(P, num_att)
            if PatternEqual(parent, root) is False:
                patterns_searched_lowest_level_lowerbound.append(parent)
            pattern_treated_unfairly_lowerbound.append(P)
        elif num_top_k > Upperbounds[k - k_min]:
            parent_candidate_for_upperbound = P
            children = GenerateChildren(P, whole_data_frame, attributes)
            S = children + S
        else: # generate children for lower bound
            children = GenerateChildren(P, whole_data_frame, attributes)
            if len(children) == 0:
                patterns_searched_lowest_level_lowerbound.append(P)
                # patterns_searched_lowest_level_upperbound.append(P)
            else:
                if len(parent_candidate_for_upperbound) > 0:
                    pattern_treated_unfairly_upperbound.append(parent_candidate_for_upperbound)
                    parent_candidate_for_upperbound = []
                S = children + S

    for k in range(k_min + 1, k_max):
        if time.time() - time1 > time_limit:
            print("newalg overtime")
            break
        new_tuple = ranked_data.iloc[[k - 1]].values.flatten().tolist()
        # top down for related patterns
        ancestors = AddNewTuple(new_tuple, Thc, pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound,
                                whole_data_frame, patterns_top_kmin, k, k_min, pc_whole_data,
                    patterns_size_topk, patterns_size_whole, Lowerbounds, Upperbounds, num_att, attributes)
        # suppose Lowerbounds and Upperbounds monotonically increases
        if Lowerbounds[k-k_min] > Lowerbounds[k-1-k_min] or Upperbounds[k-k_min] > Upperbounds[k-1-k_min]:
            num_patterns_visited = CheckCandidatesForBounds(ancestors, patterns_searched_lowest_level_lowerbound,
                                                            patterns_searched_lowest_level_upperbound, root,
                                                            pattern_treated_unfairly_lowerbound,
                                                            pattern_treated_unfairly_upperbound, patterns_top_kmin, k,
                                                            k_min, pc_whole_data, patterns_size_topk, patterns_size_whole,
                                                            Lowerbounds, Upperbounds, num_att, whole_data_frame,
                                                            attributes, num_patterns_visited)
    # RemoveDominatation(pattern_treated_unfairly)
    time2 = time.time()
    # print(duration1, duration2, duration3, duration4, duration5, duration6)
    return pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound, num_patterns_visited, time2 - time1


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
                                patterns_searched_lowest_level_upperbound, root,
                                pattern_treated_unfairly_lowerbound,
                                pattern_treated_unfairly_upperbound, patterns_top_kmin, k,
                                k_min, pc_whole_data, patterns_size_topk, patterns_size_whole,
                                Lowerbounds, Upperbounds, num_att, whole_data_frame,
                                attributes, num_patterns_visited):
    for p in patterns_searched_lowest_level_lowerbound:
        num_patterns_visited += 1
        if p in ancestors or p in pattern_treated_unfairly_lowerbound:
            continue
        st = num2string(p)
        if st in patterns_size_topk:
            pattern_size_in_topk = patterns_size_topk[st]
        else:
            pattern_size_in_topk = patterns_top_kmin.pattern_count(st)
            patterns_size_topk[st] = pattern_size_in_topk
        if pattern_size_in_topk >= Lowerbounds[k - k_min]:
            continue

        child = p
        parent = findParent(p, num_att)
        if PatternEqual(parent, root):
            CheckDominationAndAddForLowerbound(child, pattern_treated_unfairly_lowerbound)
            patterns_searched_lowest_level_lowerbound.remove(child)

        while PatternEqual(parent, root) is False:
            num_patterns_visited += 1
            st = num2string(parent)
            if st in patterns_size_topk:
                pattern_size_in_topk = patterns_size_topk[st]
            else:
                pattern_size_in_topk = patterns_top_kmin.pattern_count(st)
                patterns_size_topk[st] = pattern_size_in_topk
            if pattern_size_in_topk < Lowerbounds[k - k_min]:
                child = parent
                parent = findParent(child, num_att)
            else:
                CheckDominationAndAddForLowerbound(child, pattern_treated_unfairly_lowerbound)
                patterns_searched_lowest_level_lowerbound.remove(p)
                patterns_searched_lowest_level_lowerbound.append(parent)
                break

    for p in patterns_searched_lowest_level_upperbound:
        # if PatternEqual(p, [0, 0, 0, 0]):
        #     print("pattern equal ".format(p))
        num_patterns_visited += 1
        if p in ancestors or p in pattern_treated_unfairly_upperbound:
            continue
        st = num2string(p)
        if st in patterns_size_topk:
            pattern_size_in_topk = patterns_size_topk[st]
        else:
            pattern_size_in_topk = patterns_top_kmin.pattern_count(st)
            patterns_size_topk[st] = pattern_size_in_topk
        if pattern_size_in_topk <= Upperbounds[k - k_min]:
            continue

        parent = p
        patterns_searched_lowest_level_upperbound.remove(parent)
        children = GenerateChildren(p, whole_data_frame, attributes)
        if len(children) == 0:
            CheckDominationAndAddForUpperbound(parent, pattern_treated_unfairly_upperbound)
        while len(children) != 0:
            child = children.pop()
            num_patterns_visited += 1
            st = num2string(child)
            if st in patterns_size_topk:
                pattern_size_in_topk = patterns_size_topk[st]
            else:
                pattern_size_in_topk = patterns_top_kmin.pattern_count(st)
                patterns_size_topk[st] = pattern_size_in_topk
            if pattern_size_in_topk > Upperbounds[k - k_min]:
                parent = child
                children_new = GenerateChildren(parent, whole_data_frame, attributes)
                children = children_new + children
                if len(children_new) == 0:
                    CheckDominationAndAddForUpperbound(parent, pattern_treated_unfairly_upperbound)
                    parent = []
            else:
                if len(parent) > 0:
                    CheckDominationAndAddForUpperbound(parent, pattern_treated_unfairly_upperbound)
                    patterns_searched_lowest_level_upperbound.append(child)
                    parent = []
    return num_patterns_visited


# search top-down
def AddNewTuple(new_tuple, Thc, pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound,
                whole_data_frame, patterns_top_kmin, k, k_min, pc_whole_data,
                patterns_size_topk, patterns_size_whole, Lowerbounds, Upperbounds, num_att, attributes):
    ancestors = []
    root = [-1] * num_att
    children = GenerateChildrenRelatedToTuple(root, new_tuple)
    S = children
    parent_candidate_for_upperbound = []
    while len(S) > 0:
        P = S.pop()
        st = num2string(P)
        if st in patterns_size_whole:
            whole_cardinality = patterns_size_whole[st]
        else:
            whole_cardinality = pc_whole_data.pattern_count(st)
        if whole_cardinality >= Thc:
            if st in patterns_size_topk:
                patterns_size_topk[st] += 1
            else:
                patterns_size_topk[st] = patterns_top_kmin.pattern_count(st) + 1
            if patterns_size_topk[st] < Lowerbounds[k - k_min]:
                CheckDominationAndAddForLowerbound(new_tuple, pattern_treated_unfairly_lowerbound)
            elif patterns_size_topk[st] > Upperbounds[k - k_min]:
                parent_candidate_for_upperbound = P
                children = GenerateChildrenRelatedToTuple(P, new_tuple)
                S = children + S
                if len(children) == 0:
                    CheckDominationAndAddForUpperbound(P, pattern_treated_unfairly_upperbound)
                    parent_candidate_for_upperbound = []
            else:
                children = GenerateChildrenRelatedToTuple(P, new_tuple)
                if len(children) > 0:
                    if len(parent_candidate_for_upperbound) > 0:
                        CheckDominationAndAddForUpperbound(parent_candidate_for_upperbound, pattern_treated_unfairly_upperbound)
                        parent_candidate_for_upperbound = []
                    S = children + S
                    ancestors = ancestors + children
    return ancestors


selected_attributes = ["sex_binary", "age_binary", "race_C", "age_bucketized"]

original_file = r"../../InputData/CompasData/ForRanking/SmallDataset/CompasData_ranked_5att_100.csv"
ranked_data = pd.read_csv(original_file)
ranked_data = ranked_data.drop('rank', axis=1)

# def GraphTraverse(ranked_data, Thc, Lowerbounds, Upperbounds, k_min, k_max, time_limit):


time_limit = 20 * 60
k_min = 40
k_max = 50
Thc = 8
Lowerbounds = [1, 1, 2, 2, 2, 3, 3, 3, 3, 4]
Upperbounds = [8,8,8,9,9, 10,10,11,11, 12, 12]

# print(ranked_data[:k_max])

pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound, num_patterns_visited, running_time = GraphTraverse(ranked_data, selected_attributes, Thc,
                                                                     Lowerbounds, Upperbounds,
                                                                     k_min, k_max, time_limit)

print("num_patterns_visited = {}".format(num_patterns_visited))
print("time = {} s, num of pattern_treated_unfairly_lowerbound = {}, num of pattern_treated_unfairly_upperbound = {} ".format(running_time,
        len(pattern_treated_unfairly_lowerbound), len(pattern_treated_unfairly_upperbound)), "\n", "patterns:\n",
      pattern_treated_unfairly_lowerbound, "\n", pattern_treated_unfairly_upperbound)

print("dominated by pattern_treated_unfairly_lowerbound:")
for p in pattern_treated_unfairly_lowerbound:
    if PDominatedByM(p, pattern_treated_unfairly_lowerbound)[0]:
        print(p)



pattern_treated_unfairly_lowerbound2, pattern_treated_unfairly_upperbound2, \
num_patterns_visited2, running_time2 = naiveranking.NaiveAlg(ranked_data, selected_attributes, Thc,
                                                                     Lowerbounds, Upperbounds,
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


print("p in pattern_treated_unfairly but not in pattern_treated_unfairly2:")
for p in pattern_treated_unfairly_upperbound:
    if p not in pattern_treated_unfairly_upperbound2:
        print(p)


print("\n\n\n")

print("p in pattern_treated_unfairly2 but not in pattern_treated_unfairly:")
for p in pattern_treated_unfairly_upperbound2:
    if p not in pattern_treated_unfairly_upperbound:
        print(p)
