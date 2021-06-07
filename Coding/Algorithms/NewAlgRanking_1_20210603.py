"""
New algorithm for minority group detection in general case
Search the graph top-down, generate children using the method in coverage paper to avoid redundancy.
Stop point 1: when finding a pattern satisfying the requirements
Stop point 2: when the cardinality is too small
"""
import pandas as pd

from Algorithms import pattern_count
import time
from Algorithms import Predict_0_20210127 as predict


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
    length = len(m)
    if len(P) != length:
        return False
    for i in range(length):
        if m[i] != P[i]:
            return False
    return True


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


def GraphTraverse(ranked_data, attributes, Thc, Lowerbounds, Upperbounds, k_min, k_max, time_limit):
    print("attributes:", attributes)
    time1 = time.time()

    pc_whole_data = pattern_count.PatternCounter(ranked_data, encoded=False)
    pc_whole_data.parse_data()

    whole_data_frame = ranked_data.describe(include = 'all')

    num_patterns = 0
    root = [-1] * (len(attributes))
    S = []
    children = GenerateChildren(root, whole_data_frame, attributes)
    S = S + children
    pattern_treated_unfairly = []
    patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k_min], encoded=False)
    patterns_top_kmin.parse_data()
    patterns_size_topk = dict()
    patterns_size_whole = dict()
    k = k_min

    while len(S) > 0:
        if time.time() - time1 > time_limit:
            print("newalg overtime")
            break
        P = S.pop()
        st = num2string(P)
        # print("pattern:", P, st)
        num_patterns += 1

        whole_cardinality = pc_whole_data.pattern_count(st)
        patterns_size_whole[st] = whole_cardinality
        if whole_cardinality < Thc:
            continue

        num_top_k = patterns_top_kmin.pattern_count(st)
        patterns_size_topk[st] = num_top_k
        if num_top_k < Lowerbounds[k - k_min] or num_top_k > Upperbounds[k - k_min]:
            if PDominatedByM(P, pattern_treated_unfairly)[0] is False:
                pattern_treated_unfairly.append((P, k))
        else:
            children = GenerateChildren(P, whole_data_frame, attributes)
            S = S + children
            continue

    for k in range(k_min + 1, k_max):
        if time.time() - time1 > time_limit:
            print("newalg overtime")
            break
        new_tuple = ranked_data.iloc[[k - 1]].values.flatten().tolist()
        AddNewTuple(new_tuple, Thc, pattern_treated_unfairly, patterns_top_kmin, k, k_min, pc_whole_data,
                    patterns_size_topk, patterns_size_whole, Lowerbounds, Upperbounds, len(attributes))

    time2 = time.time()
    # print(duration1, duration2, duration3, duration4, duration5, duration6)
    return pattern_treated_unfairly, num_patterns, time2 - time1


def AddNewTuple(new_tuple, Thc, pattern_treated_unfairly, patterns_top_kmin, k, k_min, pc_whole_data,
                patterns_size_topk, patterns_size_whole, Lowerbounds, Upperbounds, num_att):
    tuple = new_tuple.copy()
    st = num2string(tuple)
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
            if PDominatedByM(tuple, pattern_treated_unfairly)[0] is False:
                pattern_treated_unfairly.append((tuple, k))
        elif st in pattern_treated_unfairly:
            pattern_treated_unfairly.remove(tuple)
    elif st in pattern_treated_unfairly:
        pattern_treated_unfairly.remove(tuple)

    for i in range(num_att - 1, 0, -1):
        tuple[i] = -1
        st = num2string(tuple)
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
                if PDominatedByM(tuple, pattern_treated_unfairly)[0] is False:
                    pattern_treated_unfairly.append((tuple, k))
            elif st in pattern_treated_unfairly:
                pattern_treated_unfairly.remove(tuple)
        elif st in pattern_treated_unfairly:
            pattern_treated_unfairly.remove(tuple)



selected_attributes = ["sex_binary","age_binary","race_C","age_bucketized"]

original_file = r"../../InputData/CompasData/ForRanking/SmallDataset/CompasData_ranked_5att_100.csv"
ranked_data = pd.read_csv(original_file)
ranked_data = ranked_data.drop('rank', axis=1)


# def GraphTraverse(ranked_data, Thc, Lowerbounds, Upperbounds, k_min, k_max, time_limit):


time_limit = 20*60
k_min = 10
k_max = 20
Thc = 5
Lowerbounds = [1,1,2,2,2,  3,3,3,3,4]
Upperbounds = [3,3,4,4,4,  5,5,5,5,6]

print(ranked_data[:k_max])

pattern_treated_unfairly, num_patterns, running_time = GraphTraverse(ranked_data, selected_attributes, Thc,
                                                                     Lowerbounds, Upperbounds,
                                                                     k_min, k_max, time_limit)



print(num_patterns)
print("time = {} s, num of patterns = {} ".format(running_time, len(pattern_treated_unfairly)), "\n", pattern_treated_unfairly)

for p in pattern_treated_unfairly:
    if PDominatedByM(p, pattern_treated_unfairly)[0]:
        print(p)
