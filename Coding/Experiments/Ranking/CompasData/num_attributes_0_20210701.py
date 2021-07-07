"""
This script is to do experiment on the number of attribtues.

two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: the number of attributes, from 2 to 13.

other parameters:
CleanAdult2.csv
size threshold Thc = 30
threshold of minority group accuracy: overall acc - 20
"""


import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlgRanking_8_20210702 as newalg
from Algorithms import NaiveAlgRanking_2_20210701 as naivealg
from Algorithms import Predict_0_20210127 as predict
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20
plt.rc('figure', figsize=(7, 5.6))

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



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

def GridSearch(original_data, all_attributes, thc, Lowerbounds, Upperbounds, number_attributes, time_limit, only_new_alg=False):

    selected_attributes = all_attributes[:number_attributes]
    print("{} attributes: {}".format(number_attributes, selected_attributes))

    less_attribute_data = original_data[selected_attributes]


    if only_new_alg:
        pattern_treated_unfairly_lowerbound1, pattern_treated_unfairly_upperbound1, num_patterns_visited1_, t1_ \
            = newalg.GraphTraverse(
            less_attribute_data, selected_attributes, thc,
            Lowerbounds, Upperbounds,
            k_min, k_max, time_limit)

        print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
        print(
            "time = {} s, num of pattern_treated_unfairly_lowerbound = {}, num of pattern_treated_unfairly_upperbound = {} ".format(
                t1_,
                len(pattern_treated_unfairly_lowerbound1), len(pattern_treated_unfairly_upperbound1)), "\n",
            "patterns:\n",
            pattern_treated_unfairly_lowerbound1, "\n", pattern_treated_unfairly_upperbound1)

        return t1_, num_patterns_visited1_, 0, 0, pattern_treated_unfairly_lowerbound1, pattern_treated_unfairly_upperbound1


    pattern_treated_unfairly_lowerbound1, pattern_treated_unfairly_upperbound1, num_patterns_visited1_, t1_ \
        = newalg.GraphTraverse(
        less_attribute_data, selected_attributes, thc,
        Lowerbounds, Upperbounds,
        k_min, k_max, time_limit)

    print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
    print(
        "time = {} s, num of pattern_treated_unfairly_lowerbound = {}, num of pattern_treated_unfairly_upperbound = {} ".format(
            t1_,
            len(pattern_treated_unfairly_lowerbound1), len(pattern_treated_unfairly_upperbound1)), "\n",
        "patterns:\n",
        pattern_treated_unfairly_lowerbound1, "\n", pattern_treated_unfairly_upperbound1)


    pattern_treated_unfairly_lowerbound2, pattern_treated_unfairly_upperbound2, \
    num_patterns_visited2_, t2_ = naivealg.NaiveAlg(less_attribute_data, selected_attributes, thc,
                                                    Lowerbounds, Upperbounds,
                                                    k_min, k_max, time_limit)

    print("num_patterns_visited = {}".format(num_patterns_visited2_))
    print(
        "time = {} s, num of pattern_treated_unfairly_lowerbound = {}, num of pattern_treated_unfairly_upperbound = {} ".format(
            t2_,
            len(pattern_treated_unfairly_lowerbound2), len(pattern_treated_unfairly_upperbound2)), "\n",
        "patterns:\n",
        pattern_treated_unfairly_lowerbound2, "\n", pattern_treated_unfairly_upperbound2)



    if ComparePatternSets(pattern_treated_unfairly_lowerbound1, pattern_treated_unfairly_lowerbound2) is False:
        print("sanity check fails!")

    if ComparePatternSets(pattern_treated_unfairly_upperbound1, pattern_treated_unfairly_upperbound2) is False:
        print("sanity check fails!")


    if t1_ > time_limit:
        print("new alg exceeds time limit")
    if t2_ > time_limit:
        print("naive alg exceeds time limit")

    return t1_, num_patterns_visited1_, t2_, num_patterns_visited2_, \
           pattern_treated_unfairly_lowerbound2, pattern_treated_unfairly_upperbound2


all_attributes = ["age_binary","sex_binary","race_C","MarriageStatus_C","juv_fel_count_C",
                  "decile_score_C", "juv_misd_count_C","juv_other_count_C","priors_count_C","days_b_screening_arrest_C",
                  "c_days_from_compas_C","c_charge_degree_C","v_decile_score_C","start_C","end_C",
                  "event_C"]


thc = 50

original_data_file = r"../../../../InputData/CompasData/general/compas_data_cat_necessary_att_ranked.csv"

original_data = pd.read_csv(original_data_file)

ranked_data = pd.read_csv(original_data_file)


time_limit = 10*60



num_att_max_naive = 17 # if it's ??, naive out of time
num_att_min = 3
num_att_max = 17
execution_time1 = list()
execution_time2 = list()
num_calculation1 = list()
num_calculation2 = list()
num_pattern_skipped_mis_c1 = list()
num_pattern_skipped_mis_c2 = list()
num_pattern_skipped_whole_c1 = list()
num_pattern_skipped_whole_c2 = list()
num_patterns_found_lowerbound = list()
patterns_found_lowerbound = list()
num_patterns_found_upperbound = list()
patterns_found_upperbound = list()
num_loops = 1

def lowerbound(x):
    # return int((x-3)/4)
    return 5

def upperbound(x):
    # return int(3+(x-k_min+1)/3)
    return 25

k_min = 10
k_max = 50
List_k = list(range(k_min, k_max))
Lowerbounds = [lowerbound(x) for x in List_k]
Upperbounds = [upperbound(x) for x in List_k]

for number_attributes in range(num_att_min, num_att_max):
    print("\n\nnumber of attributes = {}".format(number_attributes))

    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, calculation1_,  t2_, calculation2_, pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound = \
            GridSearch(original_data, all_attributes, thc, Lowerbounds, Upperbounds, number_attributes, time_limit)
        t1 += t1_
        t2 += t2_
        calculation1 += calculation1_
        calculation2 += calculation2_
        if l == 0:
            patterns_found_lowerbound.append(pattern_treated_unfairly_lowerbound)
            num_patterns_found_lowerbound.append(len(pattern_treated_unfairly_lowerbound))
            patterns_found_upperbound.append(pattern_treated_unfairly_upperbound)
            num_patterns_found_upperbound.append(len(pattern_treated_unfairly_upperbound))
    t1 /= num_loops
    t2 /= num_loops
    calculation1 /= num_loops
    calculation2 /= num_loops

    execution_time1.append(t1)
    num_calculation1.append(calculation1)
    execution_time2.append(t2)
    num_calculation2.append(calculation2)



# for number_attributes in range(num_att_max_naive, num_att_max):
#     print("\n\nnumber of attributes = {}".format(number_attributes))
#     t1, calculation1 = 0, 0
#     result_cardinality = 0
#     for l in range(num_loops):
#         t1_, calculation1_, t2_, calculation2_, pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound = \
#             GridSearch(original_data, all_attributes, thc, Lowerbounds, Upperbounds, number_attributes, time_limit)
#         t1 += t1_
#         calculation1 += calculation1_
#         if l == 0:
#             patterns_found_lowerbound.append(pattern_treated_unfairly_lowerbound)
#             num_patterns_found_lowerbound.append(len(pattern_treated_unfairly_lowerbound))
#             patterns_found_upperbound.append(pattern_treated_unfairly_upperbound)
#             num_patterns_found_upperbound.append(len(pattern_treated_unfairly_upperbound))
#     t1 /= num_loops
#     calculation1 /= num_loops
#
#     execution_time1.append(t1)
#     num_calculation1.append(calculation1)
#



output_path = r'../../../../OutputData/Ranking2/CompasData/num_att.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)


output_file.write("execution time\n")
for n in range(num_att_min, num_att_max):
    output_file.write('{} {} {}\n'.format(n, execution_time1[n-num_att_min], execution_time2[n-num_att_min]))
# for n in range(num_att_max_naive, num_att_max):
#     output_file.write('{} {}\n'.format(n, execution_time1[n - num_att_max_naive]))


output_file.write("\n\nnumber of patterns checked\n")
for n in range(num_att_min, num_att_max):
    output_file.write('{} {} {}\n'.format(n, num_calculation1[n-num_att_min], num_calculation2[n-num_att_min]))
# for n in range(num_att_max_naive, num_att_max):
#     output_file.write('{} {}\n'.format(n, num_calculation1[n-num_att_max_naive]))



output_file.write("\n\nnumber of patterns found lowebound\n")
for n in range(num_att_min, num_att_max):
    output_file.write('{} {} \n {}\n'.format(n, num_patterns_found_lowerbound[n-num_att_min],
                                             patterns_found_lowerbound[n-num_att_min]))


output_file.write("\n\nnumber of patterns found upperbound\n")
for n in range(num_att_min, num_att_max):
    output_file.write('{} {} \n {}\n'.format(n, num_patterns_found_upperbound[n-num_att_min],
                                             patterns_found_upperbound[n-num_att_min]))



# when number of attributes = 8, naive algorithm running time > 10min
# so we only use x[:6]
x_new = list(range(num_att_min, num_att_max))
x_naive = list(range(num_att_min, num_att_max))


plt.plot(x_new, execution_time1, label="optimized algorithm", color='blue', linewidth = 3.4)
plt.plot(x_naive, execution_time2, label="naive algorithm", color='orange', linewidth = 3.4)

plt.xlabel('number of attributes')
plt.ylabel('execution time (s)')
plt.xticks(x_new)
plt.subplots_adjust(bottom=0.15, left=0.18)
plt.legend()
plt.savefig("../../../../OutputData/Ranking2/CompasData/num_att_time.png")
plt.show()


fig, ax = plt.subplots()
plt.plot(x_new, num_calculation1, label="optimized algorithm", color='blue', linewidth = 3.4)
plt.plot(x_naive, num_calculation2, label="naive algorithm", color='orange', linewidth = 3.4)
plt.xlabel('number of attributes')
plt.ylabel('number of patterns visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


plt.xticks(x_new)
plt.subplots_adjust(bottom=0.15, left=0.18)
plt.legend()
plt.savefig("../../../../OutputData/Ranking2/CompasData/num_att_calculations.png")
plt.show()

plt.close()
plt.clf()

