import copy

import pandas as pd
import sys
import math

sys.path.append('../Coding')
from itertools import combinations
from Algorithms import pattern_count
import time
from Algorithms import NewAlgRanking_19_20211216 as newalg
from Algorithms import NaiveAlgRanking_4_20211213 as naivealg

all_attributes = ["age_binary", "sex_binary", "race_C", "MarriageStatus_C", "juv_fel_count_C",
                  "decile_score_C", "juv_misd_count_C", "juv_other_count_C", "priors_count_C",
                  "days_b_screening_arrest_C",
                  "c_days_from_compas_C", "c_charge_degree_C", "v_decile_score_C", "start_C", "end_C",
                  "event_C"]

original_data_file = r"../../../InputData/CompasData/general/compas_data_cat_necessary_att_ranked.csv"

ranked_data = pd.read_csv(original_data_file, index_col=False)

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


def all_patterns_in_comb(comb, NumAttribute, mcdes, attributes):  # comb = [1,4]
    # print("All", attributes)
    all_p = []
    pattern = [-1] * NumAttribute
    DFSattributes(0, len(comb) - 1, comb, pattern, all_p, mcdes, attributes)
    return all_p


def attribute_in_pattern(pattern, all_attributes):
    num_att = len(all_attributes)
    att_in_p = []
    idx_of_att_in_p = []
    print("num_att = {}".format(num_att))
    for i in range(0, num_att):
        if pattern[i] != -1:
            att_in_p.append(all_attributes[i])
            idx_of_att_in_p.append(i)
    return att_in_p, idx_of_att_in_p


def num_attribute_in_pattern(p):
    return len(p) - p.count(-1)


def get_all_coalitions(all_attributes, data_frame_describe):
    all_patterns = []
    num_attributes = len(all_attributes)
    index_list = list(range(0, num_attributes))
    for num_att in range(1, num_attributes + 1):
        # print("----------------------------------------------------  num_att = ", num_att)
        comb_num_att = list(
            combinations(index_list, num_att))  # list of combinations of attribute index, length, num_att
        for comb in comb_num_att:
            patterns = all_patterns_in_comb(comb, num_attributes, data_frame_describe, all_attributes)
            all_patterns += patterns
    # all_patterns = [[-1] * num_attributes] + all_patterns
    return all_patterns


def num2string(pattern):
    st = ''
    for i in pattern:
        if i != -1:
            st += str(i)
        st += '|'
    st = st[:-1]
    return st


# patterns are presented as lists rather than string
def shapley_value_option1(data_topk, topk_frame_describe, pc_topk, patt, attribute, att_value, all_attributes):
    att_of_patt, idx_of_att_of_patt = attribute_in_pattern(patt, all_attributes)
    num_att_in_patt = len(att_of_patt)
    # print("num_att_in_patt = {}".format(num_att_in_patt))
    idx_of_att = all_attributes.index(attribute)
    # patt_wo_attribute = patt.copy()
    # patt_wo_attribute[idx_of_att] = -1  # black
    other_attributes = [x for x in all_attributes if x not in att_of_patt]
    if attribute in other_attributes:
        other_attributes.remove(attribute)
    all_coalitions = get_all_coalitions(other_attributes, topk_frame_describe)
    print(all_coalitions)
    def compute_demoninator(all_coalitions):
        denominator = 0
        for coa in all_coalitions:
            cont = 1
            for idx in range(0, len(other_attributes)):
                if coa[idx] == -1:
                    att = other_attributes[idx]
                    cont *= int(topk_frame_describe[att]['max']) + 1 - int(topk_frame_describe[att]['min'])
            denominator += cont
        return denominator

    denominator = compute_demoninator(all_coalitions)
    print("denomiator = {}".format(denominator))
    for coa in all_coalitions:
        for j in range(num_att_in_patt):
            coa.insert(idx_of_att_of_patt[j], patt[idx_of_att_of_patt[j]])
    if len(all_coalitions[0]) < len(patt):
        for coa in all_coalitions:
            coa.insert(idx_of_att, att_value)
    print(all_coalitions)
    # compute the denominator
    # denominator = math.factorial(len(all_attributes) - num_att_in_patt)
    # for att in other_attributes:
    #     denominator *= int(topk_frame_describe[att]['max']) + 1 - int(topk_frame_describe[att]['min'])
    # print("denomiator = {}".format(denominator))
    contribution = 0
    sum_coefficient = 0
    for coa in all_coalitions:  # coa is p
        coa_wo_patt = coa.copy()
        coa_wo_patt[idx_of_att] = -1  # p'
        size_topk_coa = pc_topk.pattern_count(num2string(coa))
        # if size_topk_coa == 0:
        #     continue
        size_topk_coa_wo_patt = pc_topk.pattern_count(num2string(coa_wo_patt))
        num_att_in_coa_wo_patt = num_attribute_in_pattern(coa_wo_patt)
        print(coa, coa_wo_patt, size_topk_coa, size_topk_coa_wo_patt)
        print(num_att_in_coa_wo_patt - 1, len(all_attributes) - num_att_in_patt - (num_att_in_coa_wo_patt - 1),
              len(all_attributes), num_att_in_patt, (num_att_in_coa_wo_patt - 1))
        numerator = math.factorial(num_att_in_coa_wo_patt - 1) * \
                    math.factorial(len(all_attributes) - num_att_in_patt - (num_att_in_coa_wo_patt - 1) - 1)
        att_after = []
        for i in range(0, len(all_attributes)):
            if coa_wo_patt[i] == -1:
                att_after.append(all_attributes[i])
        att_after.remove(attribute)
        for att in att_after:
            numerator *= int(topk_frame_describe[att]['max']) + 1 - int(topk_frame_describe[att]['min'])
        print("numerator = {}".format(numerator))
        coefficient = numerator / denominator
        contribution += coefficient * (size_topk_coa - size_topk_coa_wo_patt)
        sum_coefficient += coefficient
    print("sum_coefficient = {}".format(sum_coefficient))
    return contribution


# selected_attributes = ["c_days_from_compas_C", "juv_other_count_C", "days_b_screening_arrest_C", "start_C", "end_C",
#                        "age_binary", "priors_count_C"]
selected_attributes = all_attributes[:4]
ranked_data = ranked_data[selected_attributes]


k = 50
k_min = k
k_max = k
thc = 100
Lowerbounds = [5]
time_limit = 5*60
data_topk = ranked_data[:k]
result1, num_patterns_visited1_, t1_ \
    = newalg.GraphTraverse(
    ranked_data, selected_attributes, thc,
    Lowerbounds,
    k_min, k_max, time_limit)

groups = result1[0]
for g in groups:
    print(g)

# [-1, -1, 2, -1, -1, -1, -1]
# [-1, -1, -1, -1, 1, -1, -1]
# [-1, -1, -1, -1, 2, -1, -1]
# [-1, -1, -1, -1, -1, -1, 1]
# [-1, -1, -1, -1, -1, -1, 2]
# [-1, -1, 1, -1, -1, 1, -1]

# ranking with att:
# c days from compas, juv other count, days b screening arrest, start, end, age, and priors count




data_topk = ranked_data[:k]
# print(data_topk)
topk_frame_describe = data_topk.describe()
pc_topk = pattern_count.PatternCounter(data_topk, encoded=False)
pc_topk.parse_data()

print("selected att : {}".format(selected_attributes))

for g in groups:
    for att in selected_attributes:
        for att_value in range(int(topk_frame_describe[att]['min']), int(topk_frame_describe[att]['max']) + 1):
            print("group {}, att {}={}".format(g, att, att_value))
            contribution = shapley_value_option1(data_topk, topk_frame_describe, pc_topk, g, att, att_value,
                                                 selected_attributes)
            print("group {}, att {}={}, contribution={}".format(g, att, att_value, contribution))


# print("option 1")
# patt = [-1, 1, 1, -1]
# att_value = 1
# attribute = "sex_binary"
# contribution = shapley_value_option1(data_topk, topk_frame_describe, pc_topk, patt, attribute, att_value,
#                                      all_attributes)
# print("contribution({}, {}={}) = {}".format(patt, attribute, att_value, contribution))

# patt = [1, 1, -1, -1]
# att_value = 1
# attribute = "sex_binary"
# contribution = shapley_value_option1(data_topk, topk_frame_describe, pc_topk, patt, attribute, att_value,
#                                      all_attributes)
# print("contribution({}, {}={}) = {}".format(patt, attribute, att_value, contribution))
#
# print("option2")
#
# patt = [1, 1, 0, -1]
# att_value = 1
# attribute = "age_binary"
# contribution = shapley_value_option1(data_topk, topk_frame_describe, pc_topk, patt, attribute, att_value,
#                                      all_attributes)
# print("contribution({}, {}={}) = {}".format(patt, attribute, att_value, contribution))
#
# patt = [1, 1, 0, -1]
# att_value = 1
# attribute = "sex_binary"
# contribution = shapley_value_option1(data_topk, topk_frame_describe, pc_topk, patt, attribute, att_value,
#                                      all_attributes)
# print("contribution({}, {}={}) = {}".format(patt, attribute, att_value, contribution))
#
# patt = [1, 1, 0, -1]
# attribute = "race_C"
# att_value = 0
# contribution = shapley_value_option1(data_topk, topk_frame_describe, pc_topk, patt, attribute, att_value,
#                                      all_attributes)
# print("contribution({}, {}={}) = {}".format(patt, attribute, att_value, contribution))
#
# patt = [1, 1, 1, -1]
# att_value = 1
# attribute = "age_binary"
# contribution = shapley_value_option1(data_topk, topk_frame_describe, pc_topk, patt, attribute, att_value,
#                                      all_attributes)
# print("contribution({}, {}={}) = {}".format(patt, attribute, att_value, contribution))
#
# patt = [1, 1, 1, -1]
# att_value = 1
# attribute = "sex_binary"
# contribution = shapley_value_option1(data_topk, topk_frame_describe, pc_topk, patt, attribute, att_value,
#                                      all_attributes)
# print("contribution({}, {}={}) = {}".format(patt, attribute, att_value, contribution))
#
# patt = [1, 1, 1, -1]
# attribute = "race_C"
# att_value = 1
# contribution = shapley_value_option1(data_topk, topk_frame_describe, pc_topk, patt, attribute, att_value,
#                                      all_attributes)
# print("contribution({}, {}={}) = {}".format(patt, attribute, att_value, contribution))
#
# patt = [1, 1, 2, -1]
# att_value = 1
# attribute = "age_binary"
# contribution = shapley_value_option1(data_topk, topk_frame_describe, pc_topk, patt, attribute, att_value,
#                                      all_attributes)
# print("contribution({}, {}={}) = {}".format(patt, attribute, att_value, contribution))
#
# patt = [1, 1, 2, -1]
# att_value = 1
# attribute = "sex_binary"
# contribution = shapley_value_option1(data_topk, topk_frame_describe, pc_topk, patt, attribute, att_value,
#                                      all_attributes)
# print("contribution({}, {}={}) = {}".format(patt, attribute, att_value, contribution))
#
# patt = [1, 1, 2, -1]
# attribute = "race_C"
# att_value = 2
# contribution = shapley_value_option1(data_topk, topk_frame_describe, pc_topk, patt, attribute, att_value,
#                                      all_attributes)
# print("contribution({}, {}={}) = {}".format(patt, attribute, att_value, contribution))

#
# patt = [1, -1, 0, -1]
# attribute = "age_binary"
# contribution = shapley_value_option1(data_topk, topk_frame_describe, pc_topk, patt, attribute, 1, all_attributes)
# print(contribution)
# patt = [1, -1, 0, -1]
# attribute = "race_C"
# contribution = shapley_value_option1(data_topk, topk_frame_describe, pc_topk, patt, attribute, 0, all_attributes)
# print(contribution)
