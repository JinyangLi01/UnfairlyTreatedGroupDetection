import copy

import pandas as pd
import sys
import math

sys.path.append('../Coding')
from itertools import combinations
from Algorithms import pattern_count
import time

all_attributes = ["age_binary", "sex_binary", "race_C", "MarriageStatus_C", "juv_fel_count_C",
                  "decile_score_C", "juv_misd_count_C", "juv_other_count_C", "priors_count_C",
                  "days_b_screening_arrest_C",
                  "c_days_from_compas_C", "c_charge_degree_C", "v_decile_score_C", "start_C", "end_C",
                  "event_C"]

original_data_file = r"../../../../InputData/CompasData/ForRanking/LargeDatasets/compas_data_cat_necessary_att_ranked.csv"

selected_attributes = all_attributes[:8]
ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]


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
    return all_patterns


def num2string(pattern):
    st = ''
    for i in pattern:
        if i != -2:
            st += str(i)
        st += '|'
    st = st[:-1]
    return st


# patterns are presented as lists rather than string
def shapley_value_option1(data_topk, topk_frame_describe, pc_topk, patt, attribute, att_value, all_attributes):
    att_of_patt, idx_of_att_of_patt = attribute_in_pattern(patt, all_attributes)
    num_att_in_patt = len(att_of_patt)
    idx_of_att = all_attributes.index(attribute)
    patt_wo_attribute = patt.copy()
    patt_wo_attribute[idx_of_att] = -1
    other_attributes = [x for x in all_attributes if x not in att_of_patt]
    all_coalitions = get_all_coalitions(other_attributes, topk_frame_describe)
    for coa in all_coalitions:
        for j in range(num_att_in_patt):
            coa.insert(idx_of_att_of_patt[j], att_of_patt[j])
    contribution = 0
    for coa in all_coalitions:
        coa_wo_patt = coa.copy()
        coa_wo_patt[idx_of_att] = -1
        size_topk_coa = pc_topk.pattern_count(num2string(coa))
        size_topk_coa_wo_patt = pc_topk.pattern_count(num2string(coa_wo_patt))
        num_att_in_coa_wo_patt = num_attribute_in_pattern(coa_wo_patt)
        contribution += math.factorial(num_att_in_coa_wo_patt) * \
                        math.factorial(num_att_in_patt - num_att_in_coa_wo_patt - 1) / math.factorial(num_att_in_patt) \
                        * (size_topk_coa - size_topk_coa_wo_patt)
    return contribution
