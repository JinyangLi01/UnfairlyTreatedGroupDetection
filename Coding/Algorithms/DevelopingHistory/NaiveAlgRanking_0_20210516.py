"""
naive alg for ranking

"""

from Algorithms import pattern_count
import time


def DFSattributes(cur, last, comb, pattern, all_p, mcdes, attributes):
    # print("DFS", attributes)
    if cur == last:
        # print("comb[{}] = {}".format(cur, comb[cur]))
        # print("{} {}".format(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max'])))
        for a in range(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max']) + 1):
            s = pattern.copy()
            s[comb[cur]] = a
            all_p.append(s)
        return
    else:
        # print("comb[{}] = {}".format(cur, comb[cur]))
        # print("{} {}".format(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max'])))
        for a in range(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max']) + 1):
            s = pattern.copy()
            s[comb[cur]] = a
            DFSattributes(cur + 1, last, comb, s, all_p, mcdes, attributes)


def AllPatternsInComb(comb, NumAttribute, mcdes, attributes):  # comb = [1,4]
    # print("All", attributes)
    all_p = []
    pattern = [-1] * NumAttribute
    DFSattributes(0, len(comb) - 1, comb, pattern, all_p, mcdes, attributes)
    return all_p


def num2string(pattern):
    st = ''
    for i in pattern:
        if i != -1:
            st += str(i)
        st += '|'
    st = st[:-1]
    return st


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


# coverage of P among dataset D
def cov(P, D):
    cnt = 0
    for d in D:
        if P1DominatedByP2(d, P):
            cnt += 1
    return cnt


# whether a pattern P is dominated by MUP M
# except from P itself
def PDominatedByM(P, M):
    for m in M:
        if PatternEqual(m, P):
            continue
        if P1DominatedByP2(P, m):
            # print(P, "domintated by", m)
            return True, m
    return False, None


def GenerateChildren(P, whole_data_frame, attributes):
    children = []
    length = len(P)
    i = 0
    for i in range(length-1, -1, -1):
        if P[i] != -1:
            break
    if P[i] == -1:
        i -= 1
    for j in range(i+1, length, 1):
        for a in range(int(whole_data_frame[attributes[j]]['min']), int(whole_data_frame[attributes[j]]['max'])+1):
            s = P.copy()
            s[j] = a
            children.append(s)
    return children





"""
whole_data: the original data file 
mis_class_data: file containing mis-classified tuples
Tha: delta fairness value 
Thc: size threshold
"""

def NaiveAlg(ranked_data, attributes, Thc, Lowerbounds, Upperbounds, k_min, k_max, time_limit):
    time1 = time.time()

    pc_whole_data = pattern_count.PatternCounter(ranked_data, encoded=False)
    pc_whole_data.parse_data()

    whole_data_frame = ranked_data.describe(include='all')
    num_patterns_visited = 0
    pattern_treated_unfairly = []
    pattern_treated_unfairly_with_k = []

    for k in range(k_min, k_max):
        root = [-1] * (len(attributes))
        S = GenerateChildren(root, whole_data_frame, attributes)
        patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k], encoded=False)
        patterns_top_kmin.parse_data()

        while len(S) > 0:
            if time.time() - time1 > time_limit:
                print("newalg overtime")
                break
            P = S.pop()
            # if PatternEqual(P, [-1, -1, 1, 1]):
            #     print("pattern equal ".format(P))
            st = num2string(P)

            num_patterns_visited += 1

            whole_cardinality = pc_whole_data.pattern_count(st)
            if whole_cardinality < Thc:
                continue
            num_top_k = patterns_top_kmin.pattern_count(st)
            if num_top_k < Lowerbounds[k - k_min] or num_top_k > Upperbounds[k - k_min]:
                if PDominatedByM(P, pattern_treated_unfairly)[0] is False:
                    if P not in pattern_treated_unfairly:
                        pattern_treated_unfairly_with_k.append((P, k))
                        pattern_treated_unfairly.append(P)
            else:
                children = GenerateChildren(P, whole_data_frame, attributes)
                S = S + children
                continue
    for p in pattern_treated_unfairly:
        if PDominatedByM(p, pattern_treated_unfairly)[0]:
            pattern_treated_unfairly.remove(p)
    time2 = time.time()
    return pattern_treated_unfairly, num_patterns_visited, time2 - time1



#
# selected_attributes = ["sex_binary", "age_binary", "race_C", "age_bucketized"]
#
# original_file = r"../../InputData/CompasData/ForRanking/SmallDataset/CompasData_ranked_5att_100.csv"
# ranked_data = pd.read_csv(original_file)
# ranked_data = ranked_data.drop('rank', axis=1)
#
# # def GraphTraverse(ranked_data, Thc, Lowerbounds, Upperbounds, k_min, k_max, time_limit):
#
#
# time_limit = 20 * 60
# k_min = 10
# k_max = 20
# Thc = 5
# Lowerbounds = [1, 1, 2, 2, 2, 3, 3, 3, 3, 4]
# Upperbounds = [3, 3, 4, 4, 4, 5, 5, 5, 5, 6]
#
# print(ranked_data[:k_max])
#
# pattern_treated_unfairly, num_patterns, running_time = NaiveAlg(ranked_data, selected_attributes, Thc,
#                                                                      Lowerbounds, Upperbounds,
#                                                                      k_min, k_max, time_limit)
#
# print(num_patterns)
# print("time = {} s, num of patterns = {} ".format(running_time, len(pattern_treated_unfairly)), "\n", "patterns:\n",
#       pattern_treated_unfairly)
#
# # print("dominated by pattern_treated_unfairly:")
# # for p in pattern_treated_unfairly:
# #     if PDominatedByM(p, pattern_treated_unfairly)[0]:
# #         print(p)
# #
