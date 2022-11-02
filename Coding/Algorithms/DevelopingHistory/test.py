
import pandas as pd
from Algorithms.DevelopingHistory import NaiveAlgRanking_definition2_5_20220506 as naiveranking, \
    NewAlgRanking_definition2_13_20220509 as newranking

all_attributes = ["age_binary", "sex_binary", "race_C", "MarriageStatus_C", "juv_fel_count_C",
                  "decile_score_C", "juv_misd_count_C", "juv_other_count_C", "priors_count_C", "days_b_screening_arrest_C",
                  "c_days_from_compas_C", "v_decile_score_C", "c_charge_degree_C", "start_C", "end_C",
                  "event_C"]

att_num = 15
# 13 att, 66s VS 138s
# 14 att, 182 VS 228s
# 15 ??
selected_attributes = all_attributes[:att_num]
print("{} attributes".format(att_num))
original_data_file = r"../../../InputData/CompasData/ForRanking/LargeDatasets/compas_data_cat_necessary_att_ranked.csv"

ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]

time_limit = 10 * 60
k_min = 10
k_max = 30
Thc = 50

List_k = list(range(k_min, k_max))

alpha = 0.8

# logger = logging.getLogger('MyLogger')


print("start the naive alg")

pattern_treated_unfairly2, num_patterns_visited2, running_time2 = \
    naiveranking.NaiveAlg(ranked_data, selected_attributes, Thc,
                          alpha,
                          k_min, k_max, time_limit)

print("num_patterns_visited = {}".format(num_patterns_visited2))
print("time = {} s".format(running_time2))
# for k in range(0, k_max - k_min):
#     print("k = {}, num = {}, patterns =".format(k + k_min, len(pattern_treated_unfairly2[k])),
#           pattern_treated_unfairly2[k])



print("start the new alg")

pattern_treated_unfairly, num_patterns_visited, running_time = \
    newranking.GraphTraverse(ranked_data, selected_attributes, Thc,
                  alpha, k_min, k_max, time_limit)

print("num_patterns_visited = {}".format(num_patterns_visited))
print("time = {} s".format(running_time))
# for k in range(0, k_max - k_min):
#     print("k = {}, num = {}, patterns =".format(k + k_min, len(pattern_treated_unfairly[k])),
#           pattern_treated_unfairly[k])


k_printed = False
print("p in pattern_treated_unfairly but not in pattern_treated_unfairly2:")
for k in range(0, k_max - k_min):
    for p in pattern_treated_unfairly[k]:
        if p not in pattern_treated_unfairly2[k]:
            if k_printed is False:
                print("k=", k + k_min)
                k_printed = True
            print(p)

print("\n\n\n")

k_printed = False
print("p in pattern_treated_unfairly2 but not in pattern_treated_unfairly:")
for k in range(0, k_max - k_min):
    for p in pattern_treated_unfairly2[k]:
        if p not in pattern_treated_unfairly[k]:
            if k_printed is False:
                print("k=", k + k_min)
                k_printed = True
            print(p)
