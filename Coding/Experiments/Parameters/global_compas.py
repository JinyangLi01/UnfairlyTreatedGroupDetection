import pandas as pd
from Algorithms import GlobalBounds

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
            if GlobalBounds.PatternEqual(p, q):
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



selected_attributes = all_attributes


Thc_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
k_min = 10
k_max = 50

original_data_file = r"../../../InputData/CompasData/ForRanking/LargeDatasets/compas_data_cat_necessary_att_ranked.csv"


ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]

time_limit = 10*60


List_k = list(range(k_min, k_max))


Lowerbounds = [10] * 10 + [20] * 10 + [30] * 10 + [40] * 10

# 0.2: 1377, 14-28
# 0.25: 1722, 14 - 25
# 0.3: 12-26
# 0.4: 2755, 1 - 15
Thc = 0.4 * len(ranked_data)

print(len(ranked_data), Thc)


result1, num_patterns_visited1_, t1_, _ \
            = GlobalBounds.GraphTraverse(
            ranked_data, selected_attributes, Thc,
            Lowerbounds,
            k_min, k_max, time_limit)

print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
print("time = {} s, num of pattern_treated_unfairly_lowerbound = {} ".format(
    t1_, len(result1)))
for k in range(k_min, k_max):
    print("k={}, num={}".format(k, len(result1[k-k_min])))


