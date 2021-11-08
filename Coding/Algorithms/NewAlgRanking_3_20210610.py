"""
New algorithm for group detection in ranking
fairness definition: the number of a group members in top-k is bounded by U_k, L_k, k_min <= k <= k_max
bounds for different k can be different, but all patterns have the same bounds
this alg has some problem, not the final used version
"""


"""
1. For related nodes, go top down
2. For unrelated, start from the last search and go up
    a. child's whole size is too small, so stop at the parent. lower bound <= parent in top-k <= upper bound
        with new k, need to check the lower bound of the parent
    b. child size in top-k < lower bound, stop at the parent. 
        with new k, need to check the lower bound of the parent
    c. child size in top-k > upper bound, it's parent's size also > upper bound. 
        So the result set is [root]. Search stops.

"""

import pandas as pd

from itertools import combinations
from Algorithms import pattern_count
import time
from Algorithms import Predict_0_20210127 as predict
from Algorithms import NaiveAlgRanking_0_20210516 as naiveranking


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
    for i in range(length):
        if m[i] != P[i]:
            return False
    return True


def GenerateChildrenRelatedToTuple(P, whole_data_frame, attributes, new_tuple):
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
    pattern_treated_unfairly = []
    pattern_treated_unfairly_with_k = []
    patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k_min], encoded=False)
    patterns_top_kmin.parse_data()
    patterns_size_topk = dict()
    patterns_size_whole = dict()
    k = k_min
    patterns_searched_lowest_level = []


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
                patterns_searched_lowest_level.append(parent)
            continue

        num_top_k = patterns_top_kmin.pattern_count(st)
        patterns_size_topk[st] = num_top_k
        if num_top_k < Lowerbounds[k - k_min] or num_top_k > Upperbounds[k - k_min]:
            parent = findParent(P, num_att)
            if PatternEqual(parent, root) is False:
                patterns_searched_lowest_level.append(parent)
            pattern_treated_unfairly.append(P)
            pattern_treated_unfairly_with_k.append((P, k))
        else:
            children = GenerateChildren(P, whole_data_frame, attributes)
            if len(children) == 0:
                patterns_searched_lowest_level.append(P)
            else:
                S = S + children
            continue
    for k in range(k_min + 1, k_max):
        if time.time() - time1 > time_limit:
            print("newalg overtime")
            break
        new_tuple = ranked_data.iloc[[k - 1]].values.flatten().tolist()
        # top down for related patterns
        ancestors = AddNewTuple(new_tuple, Thc, pattern_treated_unfairly, whole_data_frame, patterns_top_kmin, k, k_min, pc_whole_data,
                    patterns_size_topk, patterns_size_whole, Lowerbounds, Upperbounds, num_att, attributes)
        # suppose Lowerbounds and Upperbounds monotonically increases
        if Lowerbounds[k-k_min] > Lowerbounds[k-1-k_min] or Upperbounds[k-k_min] > Upperbounds[k-1-k_min]:
            num_patterns_visited = CheckCandidatesForBounds(ancestors, patterns_searched_lowest_level, root, pattern_treated_unfairly, patterns_top_kmin, k, k_min, pc_whole_data,
                    patterns_size_topk, patterns_size_whole, Lowerbounds, Upperbounds, num_att, whole_data_frame, attributes, num_patterns_visited)
    # RemoveDominatation(pattern_treated_unfairly)
    time2 = time.time()
    # print(duration1, duration2, duration3, duration4, duration5, duration6)
    return pattern_treated_unfairly, num_patterns_visited, time2 - time1


def CheckDominationAndAdd(pattern, pattern_treated_unfairly):
    for p in pattern_treated_unfairly:
        # if PatternEqual(p, pattern):
        #     return
        if P1DominatedByP2(pattern, p):
            return
        elif P1DominatedByP2(p, pattern):
            pattern_treated_unfairly.remove(p)
    pattern_treated_unfairly.append(pattern)


# only need to check the lower bound of parents
def CheckCandidatesForBounds(ancestors, patterns_searched_lowest_level, root, pattern_treated_unfairly, patterns_top_kmin, k, k_min, pc_whole_data,
                patterns_size_topk, patterns_size_whole, Lowerbounds, Upperbounds, num_att, whole_data_frame, attributes, num_patterns_visited):

    for p in patterns_searched_lowest_level:
        num_patterns_visited += 1
        if p in ancestors or p in pattern_treated_unfairly:
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
            CheckDominationAndAdd(child, pattern_treated_unfairly)
            patterns_searched_lowest_level.remove(child)

        while PatternEqual(parent, root) is False:
            num_patterns_visited += 1
            st = num2string(parent)
            if st in patterns_size_topk:
                pattern_size_in_topk = patterns_size_topk[st]
            else:
                pattern_size_in_topk = patterns_top_kmin.pattern_count(st)
                patterns_size_topk[st] = pattern_size_in_topk
            if pattern_size_in_topk < Lowerbounds[k - k_min]:
                child_treated_unfairly = True
                child = parent
                parent = findParent(child, num_att)
            else:
                CheckDominationAndAdd(child, pattern_treated_unfairly)
                patterns_searched_lowest_level.remove(p)
                patterns_searched_lowest_level.append(parent)
                break

    return num_patterns_visited


# search top-down
def AddNewTuple(new_tuple, Thc, pattern_treated_unfairly, whole_data_frame, patterns_top_kmin, k, k_min, pc_whole_data,
                patterns_size_topk, patterns_size_whole, Lowerbounds, Upperbounds, num_att, attributes):
    ancestors = []
    root = [-1] * num_att
    children = GenerateChildrenRelatedToTuple(root, whole_data_frame, attributes, new_tuple)
    S = children
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
            if patterns_size_topk[st] < Lowerbounds[k - k_min] or patterns_size_topk[st] > Upperbounds[k - k_min]:
                CheckDominationAndAdd(new_tuple, pattern_treated_unfairly)
            else:
                children = GenerateChildrenRelatedToTuple(P, whole_data_frame, attributes, new_tuple)
                S = S + children
                ancestors = ancestors + children
    return ancestors


selected_attributes = ["sex_binary", "age_binary", "race_C", "age_bucketized"]

original_file = r"../../InputData/CompasData/ForRanking/SmallDataset/CompasData_ranked_5att_100.csv"
ranked_data = pd.read_csv(original_file)
ranked_data = ranked_data.drop('rank', axis=1)

# def GraphTraverse(ranked_data, Thc, Lowerbounds, Upperbounds, k_min, k_max, time_limit):


time_limit = 20 * 60
k_min = 10
k_max = 20
Thc = 5
Lowerbounds = [1, 1, 2, 2, 2, 3, 3, 3, 3, 4]
Upperbounds = [3, 3, 4, 4, 4, 5, 5, 5, 5, 6]

print(ranked_data[:k_max])

pattern_treated_unfairly, num_patterns_visited, running_time = GraphTraverse(ranked_data, selected_attributes, Thc,
                                                                     Lowerbounds, Upperbounds,
                                                                     k_min, k_max, time_limit)

print("num_patterns_visited = {}".format(num_patterns_visited))
print("time = {} s, num of patterns = {} ".format(running_time, len(pattern_treated_unfairly)), "\n", "patterns\n",
      pattern_treated_unfairly)

print("dominated by pattern_treated_unfairly:")
for p in pattern_treated_unfairly:
    if PDominatedByM(p, pattern_treated_unfairly)[0]:
        print(p)



pattern_treated_unfairly2, num_patterns_visited2, running_time2 = naiveranking.NaiveAlg(ranked_data, selected_attributes, Thc,
                                                                     Lowerbounds, Upperbounds,
                                                                     k_min, k_max, time_limit)
print("num_patterns_visited2 = {}".format(num_patterns_visited2))
print("time = {} s, num of patterns = {} ".format(running_time2, len(pattern_treated_unfairly2)), "\n", "patterns\n",
      pattern_treated_unfairly2)

print("dominated by pattern_treated_unfairly2:")
for p in pattern_treated_unfairly2:
    t, m = PDominatedByM(p, pattern_treated_unfairly2)
    if t:
        print("{} dominated by {}".format(p, m))


print("p in pattern_treated_unfairly but not in pattern_treated_unfairly2:")
for p in pattern_treated_unfairly:
    if p not in pattern_treated_unfairly2:
        print(p)


print("\n\n\n")

print("p in pattern_treated_unfairly2 but not in pattern_treated_unfairly:")
for p in pattern_treated_unfairly2:
    if p not in pattern_treated_unfairly:
        print(p)
