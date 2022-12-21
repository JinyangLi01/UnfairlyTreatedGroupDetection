import pandas as pd
from Algorithms.DevelopingHistory import NaiveAlgRanking_definition2_5_20220506 as naivealg, \
    NewAlgRanking_definition2_15 as newalg

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


all_attributes = ['StatusExistingAcc', 'DurationMonth_C', 'CreditHistory', 'Purpose', 'CreditAmount_C',
                  'SavingsAccount', 'EmploymentLength', 'InstallmentRate', 'MarriedNSex', 'Debtors',
                  'ResidenceLength', 'Property', 'Age_C', 'InstallmentPlans', 'Housing',
                  'ExistingCredit', 'Job', 'NumPeopleLiable', 'Telephone', 'ForeignWorker']


selected_attributes = all_attributes


Thc_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
k_min = 10
k_max = 50

original_data_file = r"../../../InputData/GermanCredit/GermanCredit_ranked.csv"


ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]

time_limit = 10*60


List_k = list(range(k_min, k_max))

alpha = 0.8

# 0.4: 8 - 13
# 0.25: 14 - 23
Thc = len(ranked_data) * 0.25

print(len(ranked_data), Thc)

result1, num_patterns_visited1_, t1_ \
    = newalg.GraphTraverse(
    ranked_data, selected_attributes, Thc,
    alpha,
    k_min, k_max, time_limit)

print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
print("time = {} s, num of pattern_treated_unfairly_lowerbound = {} ".format(
    t1_, len(result1)))
for k in range(k_min, k_max):
    print("k={}, num={}".format(k, len(result1[k-k_min])))


