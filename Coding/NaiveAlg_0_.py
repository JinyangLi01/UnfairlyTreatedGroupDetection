from itertools import combinations
import pandas as pd
import numpy as np
import pattern_count
import time



def Prepatation(filename):
    mc = pd.read_csv(filename)
    mcdes = mc.describe()
    attributes = mcdes.columns.values
    return mc, mcdes, attributes


def DFSattributes(cur, last, comb, pattern, all_p, mcdes, attributes):
    #print("DFS", attributes)
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
    #print("All", attributes)
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


# coverage of P among dataset D
def cov(P, D):
    cnt = 0
    for d in D:
        if P1DominatedByP2(d, P):
            cnt += 1
    return cnt


# whether a pattern P is dominated by MUP M
def PDominatedByM(P, M):
    for m in M:
        if P1DominatedByP2(P, m):
            return True
    return False


def equalPattern(s, t):
    if len(s) != len(t):
        return False
    lens = len(s)
    for i in range(0, lens):
        if s[i] != t[i]:
            return False
    return True


def NaiveAlg(whole_data_file, mis_class_data_file, Tha, Thc):
    time1 = time.time()

    mc, mcdes, attributes = Prepatation(mis_class_data_file)
    NumAttribute = len(attributes)
    index_list = list(range(0, NumAttribute))  # list[1, 2, ...13]

    column_list_mc = np.array(mc.columns).tolist()
    pc_mc = pattern_count.PatternCounter(mis_class_data_file, column_list_mc, encoded=False)
    pc_mc.parse_data()

    data = pd.read_csv(whole_data_file)
    column_list_adult = np.array(data.columns).tolist()
    pc_adult = pattern_count.PatternCounter(whole_data_file, column_list_adult, encoded=False)
    pc_adult.parse_data()

    num_pattern_checked = 0
    pattern_with_low_accuracy = []
    for num_att in range(1, NumAttribute + 1):
        comb_num_att = list(
            combinations(index_list, num_att))  # list of combinations of attribute index, length num_att
        for comb in comb_num_att:
            patterns = AllPatternsInComb(comb, NumAttribute, mcdes, attributes)
            for p in patterns:

                num_pattern_checked += 1
                p_ = num2string(p)
                cardinality = pc_adult.pattern_count(p_)

                if cardinality < Thc:
                    continue
                mc = pc_mc.pattern_count(p_)
                acc = 1 - mc / cardinality

                if acc < Tha:
                    if not PDominatedByM(p, pattern_with_low_accuracy):
                        pattern_with_low_accuracy.append(p)
                        #print("pattern_with_low_accuracy number = {}".format(len(pattern_with_low_accuracy)))

    time2 = time.time()
    execution_time = time2 - time1
    print("execution time = %s seconds" % execution_time)
    print(len(pattern_with_low_accuracy))
    print("num_pattern_checked = ", num_pattern_checked)
    return pattern_with_low_accuracy, num_pattern_checked, execution_time

