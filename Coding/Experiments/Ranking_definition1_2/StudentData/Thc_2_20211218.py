

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
    return int(x/1000)



all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C',
                  'Pstatus_C', 'Medu_C', 'Fedu_C', 'Mjob_C', 'Fjob_C',
                  'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C', 'failures_C',
                  'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C',
                  'higher_C', 'internet_C', 'romantic_C', 'famrel_C', 'freetime_C',
                  'goout_C', 'Dalc_C', 'Walc_C', 'health_C', 'absences_C',
                  'G1_C', 'G2_C', 'G3_C']


# 33 att in total
# 32 ok
selected_attributes = all_attributes[:33]

Thc_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
k_min = 10
k_max = 50

original_data_file = r"../../../../InputData/StudentDataset/ForRanking_1/student-mat_cat_ranked.csv"


ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]
time_limit = 10*60


List_k = list(range(k_min, k_max))



Lowerbounds = [10] * 10 + [20] * 10 + [30] * 10 + [40] * 10



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



for Thc in Thc_list:
    num_patterns_visited1_thc = 0
    num_patterns_visited2_thc = 0
    print("\nthc = {}".format(Thc))
    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
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

    t1 /= num_loops
    t2 /= num_loops
    calculation1 /= num_loops
    calculation2 /= num_loops
    execution_time1.append(t1)
    num_patterns_visited1.append(num_patterns_visited1_thc)
    execution_time2.append(t2)
    num_patterns_visited2.append(num_patterns_visited2_thc)





output_path = r'../../../../OutputData/Ranking_definition1_1_thc50/StudentData/thc.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("execution time\n")
for n in range(len(Thc_list)):
    output_file.write('{} {} {}\n'.format(Thc_list[n], execution_time1[n], execution_time2[n]))


output_file.write("\n\nnumber of patterns visited\n")
for n in range(len(Thc_list)):
    output_file.write('{} {} {}\n'.format(Thc_list[n], num_patterns_visited1[n], num_patterns_visited2[n]))


output_file.write("\n\nnumber of patterns found, lowerbound\n")
for n in range(len(Thc_list)):
    output_file.write('{} {} \n {}\n'.format(Thc_list[n], num_patterns_found_lowerbound[n], patterns_found_lowerbound[n]))






fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(Thc_list, execution_time1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
plt.plot(Thc_list, execution_time2, line_style[1], color=color[1], label=label[1], linewidth=line_width,
             markersize=marker_size)
plt.xlabel('Size threshold')
plt.xticks(Thc_list)
plt.ylabel('Execution time (s)')
plt.legend(loc='best')
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/Ranking_definition1_1/StudentData/thc_time_global_student.png",
            bbox_inches='tight')
plt.show()
plt.close()





fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(Thc_list, num_patterns_visited1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
plt.plot(Thc_list, num_patterns_visited2, line_style[1], color=color[1], label=label[1], linewidth=line_width,
             markersize=marker_size)
plt.xlabel('Size threshold')
plt.xticks(Thc_list)
plt.ylabel('Number of patterns visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.legend(loc='best')
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/Ranking_definition1_1/StudentData/thc_calculations_global_student.png",
            bbox_inches='tight')
plt.show()
plt.close()


plt.clf()

