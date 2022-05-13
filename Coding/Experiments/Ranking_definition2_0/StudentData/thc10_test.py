import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlgRanking_definition2_13_20220509 as newalg
from Algorithms import NaiveAlgRanking_definition2_5_20220506 as naivealg

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sns.set_palette("Paired")
# sns.set_palette("deep")
sns.set_context("poster", font_scale=2)
sns.set_style("whitegrid")
# sns.palplot(sns.color_palette("deep", 10))
# sns.palplot(sns.color_palette("Paired", 9))

line_style = ['o-', 's--', '^:', '-.p']
color = ['C0', 'C1', 'C2', 'C3', 'C4']
plt_title = ["BlueNile", "COMPAS", "Credit Card"]

label = ["PropBounds", "IterTD"]
line_width = 8
marker_size = 15
# f_size = (14, 10)

f_size = (14, 10)


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


all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
                  'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C',
                  'failures_C', 'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C', 'higher_C',
                  'internet_C', 'romantic_C', 'famrel_C', 'freetime_C', 'goout_C', 'Dalc_C', 'Walc_C',
                  'health_C', 'absences_C', 'G1_C', 'G2_C', 'G3_C']

# 11 att, ok, 36 VS 65
# 13 att, 141 VS 194
# 14 att, 365 VS 420
# 15 att, over time
selected_attributes = all_attributes[:14]

Thc_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
k_min = 10
k_max = 50

original_data_file = r"../../../../InputData/StudentDataset/ForRanking_1/student-mat_cat_ranked.csv"

ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]

time_limit = 10 * 60

alpha = 0.8

Thc = 10

print("num of attributes: {}".format(len(selected_attributes)))

result1, num_patterns_visited1_, t1_ \
    = newalg.GraphTraverse(
    ranked_data, selected_attributes, Thc,
    alpha,
    k_min, k_max, time_limit)

print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
print("time = {} s, num of pattern_treated_unfairly_lowerbound = {} ".format(
    t1_, len(result1)))

if t1_ > time_limit:
    raise Exception("new alg exceeds time limit")

result2, \
num_patterns_visited2_, t2_ = naivealg.NaiveAlg(ranked_data, selected_attributes, Thc,
                                                alpha,
                                                k_min, k_max, time_limit)

print("naive alg, num_patterns_visited = {}".format(num_patterns_visited2_))
print("time = {} s, num of pattern_treated_unfairly_lowerbound = {}".format(
    t2_, len(result2)))

if t2_ > time_limit:
    raise Exception("naive alg exceeds time limit")

for k in range(0, k_max - k_min):
    if ComparePatternSets(result1[k],
                          result2[k]) is False:
        raise Exception("sanity check fails! k = {}".format(k + k_min))

print(result1)
