"""
New algorithm for minority group detection
Search the graph top-down, generate children using the method in coverage paper to avoid redundancy.
Stop point 1: when finding a pattern satisfying the requirements
Stop point 2: when the cardinality is too small
"""

from Algorithms import pattern_count
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


"""
whole_data: the original data file 
mis_class_data: file containing mis-classified tuples
Tha: threshold of accuracy 
Thc: threshold of cardinality
"""
def GraphTraverse(whole_data, mis_class_data, Tha, Thc, time_limit):
    print("(1-Tha) * Thc = {}".format((1-Tha) * Thc))
    time1 = time.time()

    pc_mis_class = pattern_count.PatternCounter(mis_class_data, encoded=False)
    pc_mis_class.parse_data()
    pc_whole_data = pattern_count.PatternCounter(whole_data, encoded=False)
    pc_whole_data.parse_data()

    whole_data_frame = whole_data.describe()
    attributes = whole_data_frame.columns.values.tolist()

    num_calculation = 0
    root = [-1] * (len(attributes))
    S = [root]
    pattern_with_low_accuracy = []


    while len(S) > 0:
        if time.time() - time1 > time_limit:
            print("newalg overtime")
            break
        P = S.pop()
        st = num2string(P)

        num_calculation += 1
        # card_mis_cal += 1
        # time consuming!!
        mis_class_cardinality = pc_mis_class.pattern_count(st)

        if mis_class_cardinality < (1 - Tha) * Thc:
            # pattern_skipped_mis_c.append(P)
            continue

        num_calculation += 1
        # card_whole_cal += 1
        # time consuming!!
        whole_cardinality = pc_whole_data.pattern_count(st)

        if whole_cardinality < Thc:
            # pattern_skipped_whole_c.append(P)
            continue

        accuracy = (whole_cardinality - mis_class_cardinality) / whole_cardinality

        if accuracy >= Tha:
            children = GenerateChildren(P, whole_data_frame, attributes)
            S = S + children
            continue
        # why this line?
        if PDominatedByM(P, pattern_with_low_accuracy)[0] is False:
            pattern_with_low_accuracy.append(P)
    time2 = time.time()
    # print(duration1, duration2, duration3, duration4, duration5, duration6)
    return pattern_with_low_accuracy, num_calculation, time2-time1



