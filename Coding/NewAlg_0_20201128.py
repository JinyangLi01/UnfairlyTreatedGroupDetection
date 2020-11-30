
import pandas as pd
import pattern_count
import time


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

def GraphTraverse(whole_data_file, mis_class_data_file, Tha, Thc):
    time1 = time.time()

    pc_mis_class = pattern_count.PatternCounter(mis_class_data_file, encoded=False)
    pc_mis_class.parse_data()

    pc_whole_data = pattern_count.PatternCounter(whole_data_file, encoded=False)
    pc_whole_data.parse_data()

    whole_data = pd.read_csv(whole_data_file)
    whole_data_frame = whole_data.describe()
    attributes = whole_data_frame.columns.values.tolist()

    num_patterns_checked = 0
    root = [-1] * (len(attributes))
    S = [root]
    pattern_with_low_accuracy = []
    while len(S) > 0:
        P = S.pop()
        st = num2string(P)
        num_patterns_checked += 1
        mis_class_cardinality = pc_mis_class.pattern_count(st)
        if mis_class_cardinality < Tha * Thc:
            continue
        whole_cardinality = pc_whole_data.pattern_count(st)
        if whole_cardinality < Thc:
            continue
        accuracy = (whole_cardinality - mis_class_cardinality) / whole_cardinality
        if accuracy >= Tha:
            children = GenerateChildren(P, whole_data_frame, attributes)
            S = S + children
            continue
        if PDominatedByM(P, pattern_with_low_accuracy)[0] is False:
            pattern_with_low_accuracy.append(P)
    time2 = time.time()
    return pattern_with_low_accuracy, num_patterns_checked, time2-time1

