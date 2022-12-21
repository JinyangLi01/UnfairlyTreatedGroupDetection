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




all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C',
                  'Pstatus_C', 'Medu_C', 'Fedu_C', 'Mjob_C', 'Fjob_C',
                  'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C', 'failures_C',
                  'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C',
                  'higher_C', 'internet_C', 'romantic_C', 'famrel_C', 'freetime_C',
                  'goout_C', 'Dalc_C', 'Walc_C', 'health_C', 'absences_C',
                  'G1_C', 'G2_C', 'G3_C']

selected_attributes = all_attributes


Thc_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
k_min = 10
k_max = 50

ranked_data = pd.read_csv("../../../InputData/StudentDataset/original/student-mat_cat.csv")
ranked_data = ranked_data[selected_attributes]


time_limit = 10*60


List_k = list(range(k_min, k_max))


Lowerbounds = [10] * 10 + [20] * 10 + [30] * 10 + [40] * 10

# 0.6: 237, 3 - 17
Thc = 0.6 * len(ranked_data)

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


