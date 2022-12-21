
import pandas as pd
from Algorithms.DevelopingHistory import NaiveAlgRanking_4_20211213 as naivealg, NewAlgRanking_19_20211216 as newalg


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


# 33 attributes
all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
                  'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C',
                  'failures_C', 'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C', 'higher_C',
                  'internet_C', 'romantic_C', 'famrel_C', 'freetime_C', 'goout_C', 'Dalc_C', 'Walc_C',
                  'health_C', 'absences_C', 'G1_C', 'G2_C', 'G3_C']

thc = 50

original_data_file = r"../../../../InputData/StudentDataset/ForRanking_1/student-mat_cat_ranked.csv"

original_data = pd.read_csv(original_data_file)[all_attributes]



time_limit = 10*60

k_min = 10
k_max = 50
List_k = list(range(k_min, k_max))
Lowerbounds = [10] * 10 + [20] * 10 + [30] * 10 + [40] * 10


number_attributes = 33

selected_attributes = all_attributes[:number_attributes]
print("{} attributes: {}".format(number_attributes, selected_attributes))

less_attribute_data = original_data[selected_attributes]

pattern_treated_unfairly1, num_patterns_visited1_, t1_ \
        = newalg.GraphTraverse(
        less_attribute_data, selected_attributes, thc,
        Lowerbounds,
        k_min, k_max, time_limit)

print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
print("time = {} s".format(t1_))
if t1_ > time_limit:
    raise Exception("new alg exceeds time limit")

pattern_treated_unfairly2, \
    num_patterns_visited2_, t2_ = naivealg.NaiveAlg(less_attribute_data, selected_attributes, thc,
                                                    Lowerbounds,
                                                    k_min, k_max, time_limit)


print("num_patterns_visited = {}".format(num_patterns_visited2_))
print("time = {} s".format(t2_))

if t2_ > time_limit:
    raise Exception("naive alg exceeds time limit")

for k in range(k_min, k_max):
    if ComparePatternSets(pattern_treated_unfairly1[k - k_min], pattern_treated_unfairly2[k - k_min]) is False:
        raise Exception("k={}, sanity check fails!".format(k))




