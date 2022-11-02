import pandas as pd
from Algorithms.DevelopingHistory import NaiveAlgRanking_4_20211213 as naivealg, NewAlgRanking_19_20211216 as newalg

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

label = ["GlobalBounds", "IterTD"]
line_width = 8
marker_size = 15
f_size = (14, 8)


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

selected_attributes = ["c_days_from_compas_C", "juv_other_count_C", "days_b_screening_arrest_C", "start_C", "end_C",
                       "age_binary", "priors_count_C"]

Thc = 50
k_min = 50
range_k_list = [40, 90, 190, 290, 390, 490, 590, 690, 790, 890, 990]


original_data_file = r"../../../../InputData/CompasData/general/compas_data_cat_necessary_att_ranked.csv"

ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]
# ranked_data = ranked_data.drop('rank', axis=1)

time_limit = 10 * 60

execution_time1 = list()
execution_time2 = list()
num_patterns_visited1 = list()
num_patterns_visited2 = list()

num_pattern_skipped_mis_c1 = list()
num_pattern_skipped_mis_c2 = list()
num_pattern_skipped_whole_c1 = list()
num_pattern_skipped_whole_c2 = list()
num_patterns_found_lowerbound = list()
patterns_found_lowerbound = list()
num_loops = 1
pattern_treated_unfairly_lowerbound = []


def generate_lowerbound(k_min, k_max):
    lb = []
    for i in range(k_min, k_max):
        if i % 10 == 0:
            lb += [i] * 10
    return lb


Lowerbounds = generate_lowerbound(10, 1000)

print(Lowerbounds)

for range_k in range_k_list:
    k_max = k_min + range_k
    List_k = list(range(k_min, k_max))

    num_patterns_visited1_thc = 0
    num_patterns_visited2_thc = 0
    print("\nthc = {}, k={}-{}".format(Thc, k_min, k_max))
    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    result1 = []
    result2 = []
    for l in range(num_loops):
        result1, num_patterns_visited1_, t1_ \
            = newalg.GraphTraverse(
            ranked_data, selected_attributes, Thc,
            Lowerbounds,
            k_min, k_max, time_limit)

        print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
        print("time = {} s, num of pattern_treated_unfairly_lowerbound = {} ".format(
            t1_, len(result1)))

        t1 += t1_
        num_patterns_visited1_thc += num_patterns_visited1_

        result2, \
        num_patterns_visited2_, t2_ = naivealg.NaiveAlg(ranked_data, selected_attributes, Thc,
                                                        Lowerbounds,
                                                        k_min, k_max, time_limit)

        print("naive alg, num_patterns_visited = {}".format(num_patterns_visited2_))
        print("time = {} s, num of pattern_treated_unfairly_lowerbound = {}".format(
            t2_, len(result2)))

        t2 += t2_
        num_patterns_visited2_thc += num_patterns_visited2_

        if t1_ > time_limit:
            raise Exception("new alg exceeds time limit")
        if t2_ > time_limit:
            raise Exception("naive alg exceeds time limit")

        for k in range(0, k_max - k_min):
            if ComparePatternSets(result1[k],
                                  result2[k]) is False:
                raise Exception("sanity check fails! k = {}".format(k + k_min))

        if l == 0:
            patterns_found_lowerbound.append(result1)
            num_patterns_found_lowerbound.append(len(result2))
    if range_k == 990:
        pattern_treated_unfairly_lowerbound = result2

    t1 /= num_loops
    t2 /= num_loops
    calculation1 /= num_loops
    calculation2 /= num_loops
    execution_time1.append(t1)
    num_patterns_visited1.append(num_patterns_visited1_thc)
    execution_time2.append(t2)
    num_patterns_visited2.append(num_patterns_visited2_thc)

output_path = r'../../../../OutputData/Ranking_definition1_1/CompasData/range_k.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("execution time\n")
for n in range(len(range_k_list)):
    output_file.write('k={} {} {}\n'.format(range_k_list[n], execution_time1[n], execution_time2[n]))

output_file.write("\n\nnumber of patterns visited\n")
for n in range(len(range_k_list)):
    output_file.write('k={} {} {}\n'.format(range_k_list[n], num_patterns_visited1[n], num_patterns_visited2[n]))

output_file.write("\n\nnumber of patterns found, lowerbound\n")
for n in range(len(range_k_list)):
    output_file.write(
        'k={} {} \n {}\n'.format(range_k_list[n], num_patterns_found_lowerbound[n], patterns_found_lowerbound[n]))

output_file.write("\n\npatterns below lowerbound\n")
for n in range(len(range_k_list)):
    output_file.write('k={} {} \n {}\n'.format(range_k_list[n], len(pattern_treated_unfairly_lowerbound[n]),
                                               pattern_treated_unfairly_lowerbound[n]))



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
plt.savefig("../../../../OutputData/Ranking_definition1_1/CompasData/range_k_time_urb_compas.png",
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
plt.savefig("../../../../OutputData/Ranking_definition1_1/CompasData/range_k_calculations_urb_compas.png",
            bbox_inches='tight')
plt.show()
plt.close()

plt.clf()
