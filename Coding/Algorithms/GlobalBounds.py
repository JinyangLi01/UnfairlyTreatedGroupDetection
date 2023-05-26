"""
full copy from NewAlgRanking_20_20220510.py

"""

import pandas as pd

from Algorithms import pattern_count
import time

def string2list(st):
    p = list()
    idx = 0
    item = ''
    i = ''
    for i in st:
        if i == '|':
            if item == '':
                p.append(-1)
            else:
                if item.isnumeric():
                    p.append(int(item))
                else:
                    p.append(item)
                item = ''
            idx += 1
        else:
            item += i
    if i != '|':
        if item.isnumeric():
            p.append(int(item))
        else:
            p.append(item)
    else:
        p.append(-1)
    return p


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


def GenerateChildren(P, whole_data_frame, ranked_data, attributes):
    children = []
    length = len(P)
    i = 0
    for i in range(length - 1, -1, -1):
        if P[i] != -1:
            break
    if P[i] == -1:
        i -= 1
    for j in range(i + 1, length, 1):
        all_values = ranked_data[attributes[j]].unique()
        for a in all_values:
            s = P.copy()
            s[j] = a
            children.append(s)
        # for a in range(int(whole_data_frame[attributes[j]]['min']), int(whole_data_frame[attributes[j]]['max']) + 1):
        #     s = P.copy()
        #     s[j] = a
        #     children.append(s)
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


def PDominatedByMForStr(st, M, num_att):
    for m in M:
        if st == m:
            continue
        if P1DominatedByP2ForStr(st, m, num_att):
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


def CheckDominationAndAddForLowerBound(pattern_st, pattern_treated_unfairly, num_att):
    to_remove = set()
    for st in pattern_treated_unfairly:
        if st == pattern_st:
            return
        if P1DominatedByP2ForStr(pattern_st, st, num_att):
            return
        elif P1DominatedByP2ForStr(st, pattern_st, num_att):
            to_remove.add(st)
    for st in to_remove:
        pattern_treated_unfairly.remove(st)
    pattern_treated_unfairly.add(pattern_st)


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
        if s1 != s2 and s2 != '':
            return False
        start_pos1 = p1 + 1
        start_pos2 = p2 + 1
    s1 = str1[start_pos1:]
    s2 = str2[start_pos2:]
    if s1 != s2 and s2 != '':
        return False
    return True


def CheckDominationAndAddForLowerbound_topdown_search(pattern_st, pattern_treated_unfairly, num_att):
    to_remove = []
    for st in pattern_treated_unfairly:
        # if PatternEqual(p, pattern):
        #     return
        if P1DominatedByP2ForStr(pattern_st, st, num_att):
            return False
    pattern_treated_unfairly.add(pattern_st)
    return True


# whether a is an ancestor of b, a and b are string
def A_is_ancestor_of_B_string(a, b):
    if len(a) >= len(b):
        return False
    length = len(a)  # len(b) should >= len(a)
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


# whether a is an ancestor of b, a and b are string
def A_is_ancestor_of_B_list(a, b, num_att):
    nondeterministic = False
    for i in range(num_att - 1, 0, -1):
        if nondeterministic:
            if a[i] != -1:
                return False
        else:
            if a[i] == -1 and b[i] != -1:
                nondeterministic = True
            elif a[i] != -1:
                return False
    return True


def CheckDominationAndAddForLowerbound_with_backup(pattern, pattern_treated_unfairly,
                                                   patterns_dominated_by_result, num_att):
    to_remove = []
    for p in pattern_treated_unfairly:
        if P1DominatedByP2(pattern, p):
            if pattern not in patterns_dominated_by_result:
                patterns_dominated_by_result.append(pattern)
            return False
        elif P1DominatedByP2(p, pattern):
            if A_is_ancestor_of_B_list(pattern, p, num_att):
                to_remove.append(p)
            else:
                patterns_dominated_by_result.append(p)
    pattern_treated_unfairly.append(pattern)
    for s in to_remove:
        pattern_treated_unfairly.remove(s)
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




def GraphTraverse(ranked_data, attributes, Thc, Lowerbounds, k_min, k_max, time_limit):
    time0 = time.time()

    pc_whole_data = pattern_count.PatternCounter(ranked_data, encoded=False)
    pc_whole_data.parse_data()

    whole_data_frame = ranked_data.describe(include='all')

    num_patterns_visited = 0
    num_att = len(attributes)
    root = [-1] * num_att
    root_str = '|' * (num_att - 1)
    S = GenerateChildren(root, whole_data_frame, ranked_data, attributes)
    store_children = {root_str: S}
    S = store_children[root_str].copy()
    pattern_treated_unfairly_lowerbound = []  # looking for the most general patterns
    patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k_min], encoded=False)
    patterns_top_kmin.parse_data()
    patterns_size_whole = dict()
    k = k_min
    result_set_lowerbound = set()
    # stop reasons:
    # 1. add to result set
    # 2. if size too small, add its parent
    # 3. no children
    patterns_dominated_by_result = set()  # doesn't include patterns in result set
    # patterns_children_small_size = []
    """
       to get pattern in top k for the purpose of demo:
       """
    patterns_size_topk = dict()
    patterns_size_topk[k_min] = patterns_top_kmin

    # DFS
    # this part is the main time consumption
    while len(S) > 0:
        if time.time() - time0 > time_limit:
            print("newalg overtime")
            break
        P = S.pop(0)
        st = num2string(P)
        num_patterns_visited += 1
        whole_cardinality = pc_whole_data.pattern_count(st)
        patterns_size_whole[st] = whole_cardinality
        if whole_cardinality < Thc:
            # parent_str = findParentForStr(st)
            # if parent_str != root_str:
            #     patterns_children_small_size.append(parent_str)
            continue
        num_top_k = patterns_top_kmin.pattern_count(st)
        if num_top_k < Lowerbounds[k - k_min]:
            if not CheckDominationAndAddForLowerbound_topdown_search(st, result_set_lowerbound, num_att):
                patterns_dominated_by_result.add(st)
        else:
            if P[num_att - 1] == -1:  # no children
                if st in store_children:
                    children = store_children[st]
                else:
                    children = GenerateChildren(P, whole_data_frame, ranked_data, attributes)
                    store_children[st] = children
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
        patterns_size_topk[k] = patterns_top_k
        new_tuple = ranked_data.iloc[[k - 1]].values.flatten().tolist()
        # print("k={}, new tuple = {}".format(k, new_tuple))
        if Lowerbounds[k - k_min] > Lowerbounds[k - 1 - k_min]:
            result_set_lowerbound = set()
            patterns_dominated_by_result = set()  # doesn't include patterns in result set
            S = store_children[root_str].copy()
            while len(S) > 0:
                if time.time() - time0 > time_limit:
                    print("newalg overtime")
                    break
                P = S.pop(0)
                st = num2string(P)
                num_patterns_visited += 1
                whole_cardinality = pc_whole_data.pattern_count(st)
                patterns_size_whole[st] = whole_cardinality
                if whole_cardinality < Thc:
                    # parent_str = findParentForStr(st)
                    # if parent_str != root_str:
                    #     patterns_children_small_size.append(parent_str)
                    continue
                num_top_k = patterns_top_k.pattern_count(st)
                if num_top_k < Lowerbounds[k - k_min]:
                    if not CheckDominationAndAddForLowerbound_topdown_search(st, result_set_lowerbound, num_att):
                        patterns_dominated_by_result.add(st)
                else:
                    if P[num_att - 1] == -1:
                        if st in store_children:
                            children = store_children[st]
                        else:
                            children = GenerateChildren(P, whole_data_frame, ranked_data, attributes)
                            store_children[st] = children
                        S = S + children
            pattern_treated_unfairly_lowerbound.append(result_set_lowerbound)
        else:
            # the bound doesn't change
            # check resul set
            patterns_resultset_now_removed = set()
            to_append_patterns_dominated_by_result = set()
            to_search_down = []
            for st in result_set_lowerbound:
                num_patterns_visited += 1
                p = string2list(st)
                if TSatisfiesP(new_tuple, p, num_att):
                    num_top_k = patterns_top_k.pattern_count(st)
                    if num_top_k >= Lowerbounds[k - k_min]:
                        patterns_resultset_now_removed.add(st)
                        if p[num_att - 1] == -1:
                            # go down from here
                            if st in store_children:
                                children = store_children[st]
                            else:
                                children = GenerateChildren(p, whole_data_frame, ranked_data, attributes)
                                store_children[st] = children
                            to_search_down = to_search_down + children
            for st in patterns_resultset_now_removed:
                result_set_lowerbound.remove(st)
            to_append_to_result_set = set()
            num_patterns_visited, to_append_patterns_dominated_by_result = \
                GoDownForResultSet(k, patterns_top_k, result_set_lowerbound, patterns_size_whole, pc_whole_data,
                                   whole_data_frame, to_search_down,
                                   attributes, Thc, Lowerbounds[k - k_min], num_att,
                                   to_append_patterns_dominated_by_result,
                                   num_patterns_visited, to_append_to_result_set, store_children, ranked_data)
            for st in to_append_to_result_set:
                result_set_lowerbound.add(st)
            # don't need to check patterns_children_small_size or patterns_no_children
            # check patterns_dominated_by_result
            to_remove_from_dominted_set = set()
            to_append = set()
            for st in patterns_dominated_by_result:
                num_patterns_visited += 1
                num_top_k = patterns_top_k.pattern_count(st)
                p = string2list(st)
                if num_top_k >= Lowerbounds[k - k_min]:
                    to_remove_from_dominted_set.add(st)
                    if p[num_att - 1] == -1:
                        num_patterns_visited2 = GoDownForDominatedByResult(p, st, k, patterns_top_k,
                                                                           result_set_lowerbound,
                                                                           patterns_size_whole, pc_whole_data,
                                                                           whole_data_frame,
                                                                           attributes, Thc, Lowerbounds[k - k_min],
                                                                           num_att,
                                                                           patterns_dominated_by_result,
                                                                           0, to_append, store_children, ranked_data)
                        num_patterns_visited += num_patterns_visited2
                elif PDominatedByMForStr(st, patterns_resultset_now_removed, num_att):
                    to_remove_from_resultset = set()
                    can_add = True
                    for r in result_set_lowerbound:
                        if P1DominatedByP2ForStr(st, r, num_att):
                            can_add = False
                            break
                        elif P1DominatedByP2ForStr(r, st, num_att):
                            to_remove_from_resultset.add(r)
                    if can_add:
                        result_set_lowerbound.add(st)
                        to_remove_from_dominted_set.add(st)
                        for s in to_remove_from_resultset:
                            result_set_lowerbound.remove(s)
                            to_append.add(s)
            for st in to_remove_from_dominted_set:
                patterns_dominated_by_result.remove(st)
            for st in to_append:
                patterns_dominated_by_result.add(st)
            for st in to_append_patterns_dominated_by_result:
                patterns_dominated_by_result.add(st)
            pattern_treated_unfairly_lowerbound.append(result_set_lowerbound)
    time1 = time.time()
    """
        to get pattern in top k for the purpose of demo:
        when the string format of a pattern is st, then size of st in top k is 
        size = patterns_size_topk[k].pattern_count(st)
        """
    return pattern_treated_unfairly_lowerbound, num_patterns_visited, time1 - time0, pc_whole_data, patterns_size_topk


# go down to p's all descendants
def GoDownForResultSet(k, patterns_top_k, result_set_lowerbound, patterns_size_whole, pc_whole_data,
                       whole_data_frame, to_search_down,
                       attributes, Thc, lowerbound, num_att,
                       to_append_patterns_dominated_by_result,
                       num_patterns_visited, to_append_to_result_set, store_children, ranked_data):
    S = to_search_down
    while len(S) > 0:
        P = S.pop(0)
        st = num2string(P)
        num_patterns_visited += 1
        whole_cardinality = pc_whole_data.pattern_count(st)
        patterns_size_whole[st] = whole_cardinality
        if whole_cardinality < Thc:
            # patterns_children_small_size.append(findParentForStr(st))
            continue
        num_top_k = patterns_top_k.pattern_count(st)
        if num_top_k < lowerbound:
            dominated = False
            for r in result_set_lowerbound:
                # r is impossible to be dominated by P
                if P1DominatedByP2ForStr(st, r, num_att):
                    dominated = True
                    to_append_patterns_dominated_by_result.add(st)  # in case r is removed later
                    break
            if not dominated:
                to_delete_from_to_append = set()
                for s in to_append_to_result_set:
                    if P1DominatedByP2ForStr(st, s, num_att):
                        dominated = True
                        to_append_patterns_dominated_by_result.add(st)
                        break
                    elif P1DominatedByP2ForStr(s, st, num_att):
                        to_delete_from_to_append.add(s)
                if not dominated:
                    to_append_to_result_set.add(st)
                for r in to_delete_from_to_append:
                    to_append_to_result_set.remove(r)
        else:
            if P[num_att - 1] == -1:  # no children
                if st in store_children:
                    children = store_children[st]
                else:
                    children = GenerateChildren(P, whole_data_frame, ranked_data, attributes)
                    store_children[st] = children
                S = S + children
    return num_patterns_visited, to_append_patterns_dominated_by_result


# go down to p's all descendants
def GoDownForDominatedByResult(p, st, k, patterns_top_k, result_set_lowerbound, patterns_size_whole, pc_whole_data,
                               whole_data_frame,
                               attributes, Thc, lowerbound, num_att,
                               patterns_dominated_by_result,
                               num_patterns_visited, to_append, store_children, ranked_data):
    if st in store_children:
        S = store_children[st].copy()
    else:
        S = GenerateChildren(p, whole_data_frame, ranked_data, attributes)
        store_children[st] = S
        S = store_children[st].copy()
    while len(S) > 0:
        P = S.pop(0)
        st = num2string(P)
        num_patterns_visited += 1
        whole_cardinality = pc_whole_data.pattern_count(st)
        patterns_size_whole[st] = whole_cardinality
        if whole_cardinality < Thc:
            # parentstr = findParentForStr(st)
            # if parentstr not in patterns_children_small_size:
            #     patterns_children_small_size.append(parentstr)
            continue
        num_top_k = patterns_top_k.pattern_count(st)
        if num_top_k < lowerbound:
            if CheckDominationAndAddForLowerbound_topdown_search(st, result_set_lowerbound, num_att) is False:
                to_append.add(st)
        else:
            if P[num_att - 1] == -1:  # no children
                if st in store_children:
                    children = store_children[st]
                else:
                    children = GenerateChildren(P, whole_data_frame, ranked_data, attributes)
                    store_children[st] = children
                S = S + children
    return num_patterns_visited



############################################ Example ##################################################
#
# all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
#                       'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C',
#                       'failures_C', 'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C', 'higher_C',
#                       'internet_C', 'romantic_C', 'famrel_C', 'freetime_C', 'goout_C', 'Dalc_C', 'Walc_C',
#                       'health_C', 'absences_C', 'G1_C', 'G2_C', 'G3_C']
#
# all_attributes_original = ['student\'s school', 'student\'s sex', 'student\'s age',
#                   'student\'s home address type', 'family size', 'parent\'s cohabitation status',
#                   'mother\'s education', 'father\'s education', 'mother\'s job',
#                   'father\'s job', 'reason to choose this school', 'student\'s guardian',
#                   'home to school travel time', 'weekly study time', 'number of past class failures',
#                   'extra educational support', 'family educational support', 'extra paid classes within the course subject',
#                   'extra-curricular activities', 'attended nursery school', 'wants to take higher education',
#                   'Internet access at home', 'with a romantic relationship', 'quality of family relationships',
#                   'free time after school', 'going out with friends', 'workday alcohol consumption',
#                   'weekend alcohol consumption', 'current health status', 'number of school absences',
#                   'first period grade', 'second period grade', 'final grade']
#
#
# original_data_file = r"../../InputData/StudentDataset/ForRanking_1/student-mat_cat_ranked.csv"
#
# ranked_data = pd.read_csv(original_data_file, index_col=False)
#
# selected_attributes = all_attributes[:16]
#
# k = 49
# k_min = k
# k_max = k
# thc = 50
# Lowerbounds = [20]
# time_limit = 5*60
# result_global_bounds, num_patterns_visited1_, t1_ \
#     = GraphTraverse(
#     ranked_data[selected_attributes].copy(deep=True), selected_attributes, thc,
#     Lowerbounds,
#     k_min, k_max, time_limit)
# print(result_global_bounds)
# groups_global_bounds = result_global_bounds[0]
# print(len(groups_global_bounds))
# print([string2list(s) for s in groups_global_bounds])
#
