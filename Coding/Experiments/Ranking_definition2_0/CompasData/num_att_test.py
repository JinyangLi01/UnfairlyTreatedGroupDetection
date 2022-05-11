import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlgRanking_definition2_13_20220509 as newalg
from Algorithms import NaiveAlgRanking_definition2_5_20220506 as naivealg
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter



def ComparePatternSets(set1, set2):
    len1 = len(set1)
    len2 = len(set2)
    if len1 != len2:
        return False
    for p in set1:
        found = False
        for q in set2:
            if newalg.PatternEqual(p, q):
                found = True
                break
        if found is False:
            return False
    return True


def thousands_formatter(x, pos):
    return int(x / 1000)



all_attributes = ["age_binary","sex_binary","race_C","MarriageStatus_C","juv_fel_count_C",
                  "decile_score_C", "juv_misd_count_C","juv_other_count_C","priors_count_C","days_b_screening_arrest_C",
                  "c_days_from_compas_C","c_charge_degree_C","v_decile_score_C","start_C","end_C",
                  "event_C"]


thc = 50

original_data_file = r"../../../../InputData/CompasData/ForRanking/LargeDatasets/compas_data_cat_necessary_att_ranked.csv"

original_data = pd.read_csv(original_data_file)[all_attributes]


k_min = 10
k_max = 50

time_limit = 10 * 60

alpha = 0.1

# 14, both over time
# 13 att, new alg ok, naive over time
# 12 att, naive ok
number_attributes = 12

selected_attributes = all_attributes[:number_attributes]
print("{} attributes: {}".format(number_attributes, selected_attributes))

less_attribute_data = original_data[selected_attributes]
#
# pattern_treated_unfairly1, num_patterns_visited1_, t1_ \
#     = newalg.GraphTraverse(
#     less_attribute_data, selected_attributes, thc,
#     alpha,
#     k_min, k_max, time_limit)
#
# print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
# print("time = {} s".format(t1_))
# if t1_ > time_limit:
#     raise Exception("new alg exceeds time limit")

pattern_treated_unfairly2, \
num_patterns_visited2_, t2_ = naivealg.NaiveAlg(less_attribute_data, selected_attributes, thc,
                                                alpha,
                                                k_min, k_max, time_limit)

print("num_patterns_visited = {}".format(num_patterns_visited2_))
print("time = {} s".format(t2_))

if t2_ > time_limit:
    raise Exception("naive alg exceeds time limit")

# for k in range(k_min, k_max):
#     if ComparePatternSets(pattern_treated_unfairly1[k - k_min], pattern_treated_unfairly2[k - k_min]) is False:
#         raise Exception("k={}, sanity check fails!".format(k))
#



