"""

result:
output_path = r'../../../../OutputData/CaseStudy/COMPAS/ranking/4att_3.txt'

attributes: ['sex', 'age_cat', 'race_factor']
k_min=10, k_max=15, thc=50
Lower bounds = [2, 2, 2, 2, 2]
Upper bounds = [8, 8, 8, 8, 8]
num of pattern_treated_unfairly_lowerbound = 6
[1, -1, -1][female]
[-1, 2, -1][>45]
[-1, -1, 1][Asian]
[-1, -1, 2][Caucasian]
[-1, -1, 3][Hispanic]
[-1, -1, 5][Other]
num of pattern_treated_unfairly_upperbound = 2
[0, 1, -1][male, 25-45]
[0, -1, 0][male, African-American]


"""


import pandas as pd

from itertools import combinations
from Algorithms import pattern_count
import time
from Algorithms import Predict_0_20210127 as predict
from Algorithms import NaiveAlgRanking_1_20210611 as naiveranking
from Algorithms import NewAlgRanking_5_20210624 as newranking



# all_attributes = ["age_binary", "age_bucketized", "sex_binary", "race_C", "MarriageStatus_C", "juv_fel_count_C",
#                   "decile_score_C",
#                     "juv_misd_count_C", "juv_other_count_C", "priors_count_C", "days_b_screening_arrest_C",
#                   "c_days_from_compas_C", "c_charge_degree_C", "v_decile_score_C", "start_C", "end_C",
#                   "event_C"]

# selected_attributes = ["age_binary", "age_bucketized", "sex_binary", "race_C", "MarriageStatus_C", "juv_fel_count_C",
#                   "decile_score_C"]

selected_attributes = ["sex", "age_cat", "race_factor"]


original_data_file = r"../../../../InputData/COMPAS_ProPublica/compas-analysis-master/categorize_cox_parsed_filtered/cox-parsed-filtered-cat-ranked.csv"


ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]
# ranked_data = ranked_data.drop('rank', axis=1)


time_limit = 5 * 60
k_min = 10
k_max = 15
Thc = 50

List_k = list(range(k_min, k_max))

def lowerbound(x):
    return 2 # int((x-2)/4)

def upperbound(x):
    return 8 # int(5+(x-k_min+5)/2)

Lowerbounds = [lowerbound(x) for x in List_k]
Upperbounds = [upperbound(x) for x in List_k]

print(Lowerbounds, "\n", Upperbounds)

output_path = r'../../../../OutputData/CaseStudy/COMPAS/ranking/4att_3.txt'
output_file = open(output_path, "w")

output_file.write("attributes: {}\n".format(selected_attributes))
output_file.write("k_min={}, k_max={}, thc={}\n".format(k_min, k_max, Thc))
output_file.write("Lower bounds = {}\n".format(Lowerbounds))
output_file.write("Upper bounds = {}\n".format(Upperbounds))



pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound, num_patterns_visited, running_time = newranking.GraphTraverse(ranked_data, selected_attributes, Thc,
                                                                     Lowerbounds, Upperbounds,
                                                                     k_min, k_max, time_limit)

output_file.write("num of pattern_treated_unfairly_lowerbound = {}\n".format(len(pattern_treated_unfairly_lowerbound)))
for p in pattern_treated_unfairly_lowerbound:
    output_file.write(str(p))
    output_file.write("\n")
output_file.write("num of pattern_treated_unfairly_upperbound = {}\n".format(len(pattern_treated_unfairly_upperbound)))
for p in pattern_treated_unfairly_upperbound:
    output_file.write(str(p))
    output_file.write("\n")

print("num_patterns_visited = {}".format(num_patterns_visited))
print("time = {} s, num of pattern_treated_unfairly_lowerbound = {}, num of pattern_treated_unfairly_upperbound = {} ".format(running_time,
        len(pattern_treated_unfairly_lowerbound), len(pattern_treated_unfairly_upperbound)), "\n", "patterns:\n",
      pattern_treated_unfairly_lowerbound, "\n", pattern_treated_unfairly_upperbound)

print("dominated by pattern_treated_unfairly_lowerbound:")
for p in pattern_treated_unfairly_lowerbound:
    if newranking.PDominatedByM(p, pattern_treated_unfairly_lowerbound)[0]:
        print(p)



pattern_treated_unfairly_lowerbound2, pattern_treated_unfairly_upperbound2, \
num_patterns_visited2, running_time2 = naiveranking.NaiveAlg(ranked_data, selected_attributes, Thc,
                                                                     Lowerbounds, Upperbounds,
                                                                     k_min, k_max, time_limit)


print("num_patterns_visited = {}".format(num_patterns_visited2))
print("time = {} s, num of pattern_treated_unfairly_lowerbound = {}, num of pattern_treated_unfairly_upperbound = {} ".format(running_time2,
        len(pattern_treated_unfairly_lowerbound2), len(pattern_treated_unfairly_upperbound2)), "\n", "patterns:\n",
      pattern_treated_unfairly_lowerbound2, "\n", pattern_treated_unfairly_upperbound2)


print("dominated by pattern_treated_unfairly2:")
for p in pattern_treated_unfairly_lowerbound2:
    t, m = newranking.PDominatedByM(p, pattern_treated_unfairly_lowerbound2)
    if t:
        print("{} dominated by {}".format(p, m))



print("p in pattern_treated_unfairly_lowerbound but not in pattern_treated_unfairly_lowerbound2:")
for p in pattern_treated_unfairly_lowerbound:
    if p not in pattern_treated_unfairly_lowerbound2:
        print(p)


print("\n\n\n")

print("p in pattern_treated_unfairly_lowerbound2 but not in pattern_treated_unfairly_lowerbound:")
for p in pattern_treated_unfairly_lowerbound2:
    if p not in pattern_treated_unfairly_lowerbound:
        print(p)


print("\n\n\n")

print("p in pattern_treated_unfairly_upperbound but not in pattern_treated_unfairly_upperbound2:")
for p in pattern_treated_unfairly_upperbound:
    if p not in pattern_treated_unfairly_upperbound2:
        print(p)


print("\n\n\n")

print("p in pattern_treated_unfairly_upperbound2 but not in pattern_treated_unfairly_upperbound:")
for p in pattern_treated_unfairly_upperbound2:
    if p not in pattern_treated_unfairly_upperbound:
        print(p)
