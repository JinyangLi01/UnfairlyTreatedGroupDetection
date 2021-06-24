"""
This script is to do experiment on the threshold of minority group sizes.

two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: Thc, from 1 to 1000

Other parameters:
CleanAdult2.csv
selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass', 'relationship']
threshold of minority group accuracy: overall acc - 20


"""


import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlgRanking_5_20210624 as newalg
from Algorithms import NaiveAlgRanking_1_20210611 as naivealg
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



selected_attributes = ["sex_binary", "age_binary", "race_C", "age_bucketized"]
Thc = 100
num_k = 50
k_min_list = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
original_data_file = r"../../../../InputData/CompasData/ForRanking/CompasData_ranked_5att.csv"
ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data.drop('rank', axis=1)
time_limit = 5*60


def lowerbound(x):
    return int((x-3)/4)

def upperbound(x):
    return int(3+(x-k_min+1)/3)


execution_time1 = list()
execution_time2 = list()
num_patterns_visited1 = list()
num_patterns_visited2 = list()

num_pattern_skipped_mis_c1 = list()
num_pattern_skipped_mis_c2 = list()
num_pattern_skipped_whole_c1 = list()
num_pattern_skipped_whole_c2 = list()
num_patterns_found_upperbound = list()
num_patterns_found_lowerbound = list()
patterns_found_upperbound = list()
patterns_found_lowerbound = list()
num_loops = 1



for k_min in k_min_list:
    print(k_min)
    k_max = k_min + num_k
    List_k = list(range(k_min, k_max))

    Lowerbounds = [lowerbound(x) for x in List_k]
    Upperbounds = [upperbound(x) for x in List_k]

    num_patterns_visited1_thc = 0
    num_patterns_visited2_thc = 0
    print("\nthc = {}, k={}-{}".format(Thc, k_min,  k_max))
    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        pattern_treated_unfairly_lowerbound1, pattern_treated_unfairly_upperbound1, num_patterns_visited1_, t1_ \
            = newalg.GraphTraverse(
            ranked_data, selected_attributes, Thc,
            Lowerbounds, Upperbounds,
            k_min, k_max, time_limit)

        print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
        print(
            "time = {} s, num of pattern_treated_unfairly_lowerbound = {}, num of pattern_treated_unfairly_upperbound = {} ".format(
                t1_,
                len(pattern_treated_unfairly_lowerbound1), len(pattern_treated_unfairly_upperbound1)), "\n",
            "patterns:\n",
            pattern_treated_unfairly_lowerbound1, "\n", pattern_treated_unfairly_upperbound1)

        t1 += t1_
        num_patterns_visited1_thc += num_patterns_visited1_

        pattern_treated_unfairly_lowerbound2, pattern_treated_unfairly_upperbound2, \
        num_patterns_visited2_, t2_ = naivealg.NaiveAlg(ranked_data, selected_attributes, Thc,
                                                                     Lowerbounds, Upperbounds,
                                                                     k_min, k_max, time_limit)

        print("num_patterns_visited = {}".format(num_patterns_visited2_))
        print("time = {} s, num of pattern_treated_unfairly_lowerbound = {}, num of pattern_treated_unfairly_upperbound = {} ".format(
                t2_,
                len(pattern_treated_unfairly_lowerbound2), len(pattern_treated_unfairly_upperbound2)), "\n",
            "patterns:\n",
            pattern_treated_unfairly_lowerbound2, "\n", pattern_treated_unfairly_upperbound2)

        t2 += t2_
        num_patterns_visited2_thc += num_patterns_visited2_

        if t1_ > time_limit:
            print("new alg exceeds time limit")
        if t2_ > time_limit:
            print("naive alg exceeds time limit")

        if ComparePatternSets(pattern_treated_unfairly_lowerbound1, pattern_treated_unfairly_lowerbound2) is False:
            print("sanity check fails!")
        if ComparePatternSets(pattern_treated_unfairly_upperbound1, pattern_treated_unfairly_upperbound2) is False:
            print("sanity check fails!")

        if l == 0:
            patterns_found_lowerbound.append(pattern_treated_unfairly_lowerbound1)
            num_patterns_found_lowerbound.append(len(pattern_treated_unfairly_lowerbound1))
            patterns_found_upperbound.append(pattern_treated_unfairly_upperbound1)
            num_patterns_found_upperbound.append(len(pattern_treated_unfairly_upperbound1))

    t1 /= num_loops
    t2 /= num_loops
    calculation1 /= num_loops
    calculation2 /= num_loops
    execution_time1.append(t1)
    num_patterns_visited1.append(num_patterns_visited1_thc)
    execution_time2.append(t2)
    num_patterns_visited2.append(num_patterns_visited2_thc)





output_path = r'../../../../OutputData/Ranking/CompasData/placement_k.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("execution time\n")
for n in range(len(k_min_list)):
    output_file.write('{} {} {}\n'.format(k_min_list[n], execution_time1[n], execution_time2[n]))


output_file.write("\n\nnumber of patterns visited\n")
for n in range(len(k_min_list)):
    output_file.write('{} {} {}\n'.format(k_min_list[n], num_patterns_visited1[n], num_patterns_visited2[n]))


output_file.write("\n\nnumber of patterns found, lowerbound\n")
for n in range(len(k_min_list)):
    output_file.write('{} {} \n {}\n'.format(k_min_list[n], num_patterns_found_lowerbound[n], patterns_found_lowerbound[n]))


output_file.write("\n\nnumber of patterns found, upperbound\n")
for n in range(len(k_min_list)):
    output_file.write('{} {} \n {}\n'.format(k_min_list[n], num_patterns_found_upperbound[n], patterns_found_upperbound[n]))



plt.plot(k_min_list, execution_time1, label="optimized algorithm", color='blue', linewidth = 3.4)
plt.plot(k_min_list, execution_time2, label="naive algorithm", color='orange', linewidth = 3.4)

plt.xlabel('k_min')
plt.ylabel('execution time (s)')
plt.xticks([100, 500, 1000, 1500, 2000])
plt.subplots_adjust(bottom=0.15, left=0.18)
plt.legend()
plt.savefig("../../../../OutputData/Ranking/CompasData/placement_k_time.png")
plt.show()


fig, ax = plt.subplots()
plt.plot(k_min_list, num_patterns_visited1, label="optimized algorithm", color='blue', linewidth = 3.4)
plt.plot(k_min_list, num_patterns_visited2, label="naive algorithm", color='orange', linewidth = 3.4)
plt.xlabel('k_min')
plt.ylabel('number of nodes visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


plt.xticks([100, 500, 1000, 1500, 2000])

plt.subplots_adjust(bottom=0.15, left=0.18)
plt.legend()
plt.savefig("../../../../OutputData/Ranking/CompasData/placement_k_calculation.png")
plt.show()

plt.close()
plt.clf()

