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

from Algorithms import pattern_count
import time
from Algorithms.DevelopingHistory import NaiveAlgRanking_2_20210701 as naiveranking


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
    pattern_treated_unfairly_lowerbound = [] # looking for the most general patterns
    pattern_treated_unfairly_upperbound = [] # looking for the most specific patterns
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
            if len(parent_candidate_for_upperbound) > 0: # there is a parent which is above upper bound
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
            parent_candidate_for_upperbound = P # we need to store this so that if child is below upper bound, we put this into result set
            children = GenerateChildren(P, whole_data_frame, attributes)
            S = children + S
            if len(children) == 0: # P is in result set
                CheckDominationAndAddForUpperbound(P, pattern_treated_unfairly_upperbound)
                parent_candidate_for_upperbound = []
        else:
            if len(parent_candidate_for_upperbound) > 0: # P is not above upperbound, so its parent should be added to the result set
                CheckDominationAndAddForUpperbound(parent_candidate_for_upperbound, pattern_treated_unfairly_upperbound)
                parent_candidate_for_upperbound = []

    for k in range(k_min + 1, k_max):
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        patterns_top_k = pattern_count.PatternCounter(ranked_data[:k], encoded=False)
        patterns_top_k.parse_data()
        new_tuple = ranked_data.iloc[[k - 1]].values.flatten().tolist()
        # top down for related patterns, using similar methods as k_min, add to result set if needed
        # ancestors are patterns checked in AddNewTuple() function, to avoid checking them again
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
    checked_patterns = set()
    for st in patterns_searched_lowest_level_lowerbound: # st is a string
        if st in checked_patterns:
            continue
        checked_patterns.add(st)
        num_patterns_visited += 1
        p = string2num(st)
        if p in ancestors or p in pattern_treated_unfairly_lowerbound: # already checked
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
            CheckDominationAndAddForLowerbound(child, pattern_treated_unfairly_lowerbound)
            to_remove.add(child_str) # child need removing
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
def AddNewTuple(new_tuple, Thc, pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound,
                                whole_data_frame, patterns_top_k, k, k_min, pc_whole_data, num_patterns_visited,
                    patterns_size_whole, Lowerbounds, Upperbounds, num_att, attributes):

    ancestors = []
    root = [-1] * num_att
    children = GenerateChildrenRelatedToTuple(root, new_tuple) # pattern with one deternimistic attribute
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



all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
                  'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C',
                  'failures_C', 'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C', 'higher_C',
                  'internet_C', 'romantic_C', 'famrel_C', 'freetime_C', 'goout_C', 'Dalc_C', 'Walc_C',
                  'health_C', 'absences_C', 'G1_C', 'G2_C', 'G3_C']

selected_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C',
                       'Pstatus_C', 'Medu_C', 'Fedu_C', 'Mjob_C', 'Fjob_C',
                       'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C', 'failures_C',
                       'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C']

"""
with the above 19 att,
naive: 98s num_patterns_visited = 2335488
optimized: 124s num_patterns_visited = 299559
num of pattern_treated_unfairly_lowerbound = 85, num of pattern_treated_unfairly_upperbound = 18
"""

original_data_file = r"../../../InputData/StudentDataset/ForRanking_1/student-mat_cat_ranked.csv"


ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]


time_limit = 5 * 60
k_min = 10
k_max = 100
Thc = 50

List_k = list(range(k_min, k_max))

def lowerbound(x):
    return 5 # int((x-3)/4)

def upperbound(x):
    return 25 # int(3+(x-k_min+1)/3)

Lowerbounds = [lowerbound(x) for x in List_k]
Upperbounds = [upperbound(x) for x in List_k]

print(Lowerbounds, "\n", Upperbounds)




print("start the new alg")

pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound, num_patterns_visited, running_time = \
    GraphTraverse(ranked_data, selected_attributes, Thc,
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




print("start the naive alg")

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


