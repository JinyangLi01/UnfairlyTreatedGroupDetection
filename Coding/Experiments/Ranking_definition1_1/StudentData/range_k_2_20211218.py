import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlgRanking_19_20211216 as newalg
from Algorithms import NaiveAlgRanking_4_20211213 as naivealg
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




all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
                  'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C',
                  'failures_C', 'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C', 'higher_C',
                  'internet_C', 'romantic_C', 'famrel_C', 'freetime_C', 'goout_C', 'Dalc_C', 'Walc_C',
                  'health_C', 'absences_C', 'G1_C', 'G2_C', 'G3_C']


# with 32 att, k = 10-50, naive alg needs 483 s, k = 10-100, naive over time
# with 31 att, over time
# with 30 att, ok
selected_attributes = all_attributes[:30]

Thc = 50
k_min = 10
range_k_list = [40, 90, 140, 190, 240, 290, 340]

original_data_file = r"../../../../InputData/StudentDataset/ForRanking_1/student-mat_cat_ranked.csv"


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
        
        if t1_ > time_limit:
            raise Exception("new alg exceeds time limit")

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

output_path = r'../../../../OutputData/Ranking_definition1_1/StudentData/range_k.txt'
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

# output_file.write("\n\npatterns below lowerbound\n")
# for n in range(len(range_k_list)):
#     output_file.write('k={} {} \n {}\n'.format(range_k_list[n], len(pattern_treated_unfairly_lowerbound[n]),
#                                                pattern_treated_unfairly_lowerbound[n]))
#


fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(range_k_list, execution_time1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
         markersize=marker_size)
plt.plot(range_k_list, execution_time2, line_style[1], color=color[1], label=label[1], linewidth=line_width,
         markersize=marker_size)
plt.xlabel('Range of k')
plt.ylabel('Execution time (s)')
plt.xticks([100, 200, 300, 350])
plt.legend(loc='best')
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/Ranking_definition1_1/StudentData/range_k_time.png",
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
plt.xticks([100, 200, 300, 350])
plt.legend(loc='best')
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/Ranking_definition1_1/StudentData/range_k_calculations.png",
            bbox_inches='tight')
plt.show()
plt.close()

plt.clf()
