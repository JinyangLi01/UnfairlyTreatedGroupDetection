import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlgRanking_definition2_8_20211228 as newalg
from Algorithms import NaiveAlgRanking_definition2_3_20211207 as naivealg
from Algorithms import Predict_0_20210127 as predict

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

label = ["UPR", "IterTD"]
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


all_attributes = ["age_binary", "sex_binary", "race_C", "MarriageStatus_C", "juv_fel_count_C",
                  "decile_score_C", "juv_misd_count_C", "juv_other_count_C", "priors_count_C",
                  "days_b_screening_arrest_C",
                  "c_days_from_compas_C", "c_charge_degree_C", "v_decile_score_C", "start_C", "end_C",
                  "event_C"]

# with 15 att, k=10-50, new alg over time
# with 10 att, 10-600, naive over time
# with 12 att, when k = 10-100, naive needs 540 s, when k is larger, naive over time
# with 11 att, when k = 50-300, naive over time
# with 9, ok
selected_attributes = all_attributes[:8]

print("num of att = {}".format(len(selected_attributes)))

thc = 50
k_min = 10
range_k_list = [40, 90, 190, 290, 390, 490, 590, 690, 790, 890, 990]

original_data_file = r"../../../../InputData/CompasData/ForRanking/LargeDatasets/compas_data_cat_necessary_att_ranked.csv"

ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]
time_limit = 10 * 60

execution_time1 = list()
execution_time2 = list()
num_patterns_visited1 = list()
num_patterns_visited2 = list()

num_pattern_skipped_mis_c1 = list()
num_pattern_skipped_mis_c2 = list()
num_pattern_skipped_whole_c1 = list()
num_pattern_skipped_whole_c2 = list()
num_patterns_found = list()
patterns_found = list()
num_loops = 1

alpha = 0.1

for range_k in range_k_list:
    k_max = k_min + range_k
    List_k = list(range(k_min, k_max))

    num_patterns_visited1_thc = 0
    num_patterns_visited2_thc = 0
    print("\nthc = {}, k={}-{}".format(thc, k_min, k_max))
    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        pattern_treated_unfairly1, num_patterns_visited1_, t1_ \
            = newalg.GraphTraverse(
            ranked_data, selected_attributes, thc,
            alpha,
            k_min, k_max, time_limit)

        print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
        print(
            "time = {} s".format(t1_), "\n",
            "patterns:\n",
            pattern_treated_unfairly1)
        if t1_ > time_limit:
            raise Exception("new alg exceeds time limit")

        t1 += t1_
        num_patterns_visited1_thc += num_patterns_visited1_

        pattern_treated_unfairly2, \
        num_patterns_visited2_, t2_ = naivealg.NaiveAlg(ranked_data, selected_attributes, thc,
                                                        alpha,
                                                        k_min, k_max, time_limit)

        print("num_patterns_visited = {}".format(num_patterns_visited2_))
        print(
            "time = {} s".format(t2_), "\n",
            "patterns:\n",
            pattern_treated_unfairly2)

        t2 += t2_
        num_patterns_visited2_thc += num_patterns_visited2_


        if t2_ > time_limit:
            raise Exception("naive alg exceeds time limit")

        for k in range(k_min, k_max):
            if ComparePatternSets(pattern_treated_unfairly1[k - k_min], pattern_treated_unfairly2[k - k_min]) is False:
                raise Exception("k={}, sanity check fails!".format(k))

        if l == 0:
            patterns_found.append(pattern_treated_unfairly1)
            num_patterns_found.append(len(pattern_treated_unfairly1))

    t1 /= num_loops
    t2 /= num_loops
    calculation1 /= num_loops
    calculation2 /= num_loops
    execution_time1.append(t1)
    num_patterns_visited1.append(num_patterns_visited1_thc)
    execution_time2.append(t2)
    num_patterns_visited2.append(num_patterns_visited2_thc)

output_path = r'../../../../OutputData/Ranking_definition2_1/CompasData/range_k.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("num of att = {}".format(len(selected_attributes)))

output_file.write("execution time\n")
for n in range(len(range_k_list)):
    output_file.write('{} {} {}\n'.format(range_k_list[n], execution_time1[n], execution_time2[n]))

output_file.write("\n\nnumber of patterns visited\n")
for n in range(len(range_k_list)):
    output_file.write('{} {} {}\n'.format(range_k_list[n], num_patterns_visited1[n], num_patterns_visited2[n]))

output_file.write("\n\nnumber of patterns found, lowerbound\n")
for n in range(len(range_k_list)):
    output_file.write('{} {} \n {}\n'.format(range_k_list[n], num_patterns_found[n], patterns_found[n]))

fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(range_k_list, execution_time1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
         markersize=marker_size)
plt.plot(range_k_list, execution_time2, line_style[1], color=color[1], label=label[1], linewidth=line_width,
         markersize=marker_size)
plt.xlabel('Range of k')
plt.ylabel('Execution time (s)')
plt.xticks([200, 400, 600, 800, 1000])
plt.legend(loc='best')
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/Ranking_definition2_1/CompasData/range_k_time.png",
            bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(range_k_list, num_patterns_visited1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
         markersize=marker_size)
plt.plot(range_k_list, num_patterns_visited2, line_style[1], color=color[1], label=label[1], linewidth=line_width,
         markersize=marker_size)
plt.xlabel('Range of k')
plt.ylabel('Number of patterns visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.xticks([200, 400, 600, 800, 1000])
plt.legend(loc='best')
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/Ranking_definition2_1/CompasData/range_k_calculations.png",
            bbox_inches='tight')
plt.show()
plt.close()

plt.clf()
