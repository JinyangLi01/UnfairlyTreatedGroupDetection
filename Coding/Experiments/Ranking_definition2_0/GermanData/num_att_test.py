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


all_attributes = ['StatusExistingAcc', 'DurationMonth_C', 'CreditHistory', 'Purpose', 'CreditAmount_C',
                  'SavingsAccount', 'EmploymentLength', 'InstallmentRate', 'MarriedNSex', 'Debtors',
                  'ResidenceLength', 'Property', 'Age_C', 'InstallmentPlans', 'Housing',
                  'ExistingCredit', 'Job', 'NumPeopleLiable', 'Telephone', 'ForeignWorker']


Thc_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
k_min = 10
k_max = 50
original_data_file = r"../../../../InputData/GermanCredit/GermanCredit_ranked.csv"

original_data = pd.read_csv(original_data_file)[all_attributes]


time_limit = 10 * 60

alpha = 0.8

thc = 50

# new alg: 17att, 162 s; 20 att, over time; 19 att, new over time
# 18 att, new 359s, naive over time
# new: <=18 att
# naive <= 17
number_attributes = 20

print("number_attributes = {}".format(number_attributes))

selected_attributes = all_attributes[:number_attributes]
print("{} attributes: {}".format(number_attributes, selected_attributes))

less_attribute_data = original_data[selected_attributes]

pattern_treated_unfairly1, num_patterns_visited1_, t1_ \
    = newalg.GraphTraverse(
    less_attribute_data, selected_attributes, thc,
    alpha,
    k_min, k_max, time_limit)

print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
print("time = {} s".format(t1_))
if t1_ > time_limit:
    raise Exception("new alg exceeds time limit")

pattern_treated_unfairly2, \
num_patterns_visited2_, t2_ = naivealg.NaiveAlg(less_attribute_data, selected_attributes, thc,
                                                alpha,
                                                k_min, k_max, time_limit)

print("num_patterns_visited = {}".format(num_patterns_visited2_))
print("time = {} s".format(t2_))

if t2_ > time_limit:
    raise Exception("naive alg exceeds time limit")

for k in range(k_min, k_max):
    if ComparePatternSets(pattern_treated_unfairly1[k - k_min], pattern_treated_unfairly2[k - k_min]) is False:
        raise Exception("k={}, sanity check fails!".format(k))

