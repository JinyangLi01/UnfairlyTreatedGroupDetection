"""
This script is to do experiment on the threshold of minority group sizes.

two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: Thc, from 1 to 1000

Other parameters:
CleanAdult2.csv
selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass']
threshold of minority group accuracy: overall acc - 20


"""


import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlg_0_20201128 as newalg
from Algorithms import NaiveAlg_0_20201111 as naivealg
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def thousands_formatter(x, pos):
    return int(x/1000)

def GridSearch(original_data_file, selected_attributes, Thc, time_limit, att_to_predict):
    
    sanity_check, pattern_with_low_accuracy1, num_calculation1, execution_time1, \
    num_pattern_skipped_mis_c1, num_pattern_skipped_whole_c1, pattern_with_low_accuracy2, \
    num_calculation2, execution_time2, \
    overall_acc, Tha, mis_class_data = \
        wholeprocess.WholeProcessWithTwoAlgorithms(original_data_file, selected_attributes, Thc, time_limit,
                                                   att_to_predict)
    
    print("{} patterns with low accuracy: \n {}".format(len(pattern_with_low_accuracy1), pattern_with_low_accuracy1))

    if execution_time1 > time_limit:
        print("new alg exceeds time limit")
    if execution_time2 > time_limit:
        print("naive alg exceeds time limit")
    elif sanity_check is False:
        print("sanity check fails!")

    return execution_time1, num_calculation1, num_pattern_skipped_mis_c1, num_pattern_skipped_whole_c1, \
           execution_time2, num_calculation2, num_pattern_skipped_mis_c2, num_pattern_skipped_whole_c2, \
           pattern_with_low_accuracy1


selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass', 'relationship']
Thc_list = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]
original_data_file = "../../../InputData/AdultDataset/CleanAdult2.csv"
att_to_predict = 'income'
time_limit = 20*60
execution_time1 = list()
execution_time2 = list()
num_calculation1 = list()
num_calculation2 = list()
num_pattern_skipped_mis_c1 = list()
num_pattern_skipped_mis_c2 = list()
num_pattern_skipped_whole_c1 = list()
num_pattern_skipped_whole_c2 = list()
num_patterns_found = list()
patterns_found = list()
num_loops = 5

for thc in Thc_list:
    print("Thc = {}".format(thc))
    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, calculation1_, _, _, t2_, calculation2_, _, _, result = \
            GridSearch(original_data_file, selected_attributes, thc, time_limit, att_to_predict)
        t1 += t1_
        t2 += t2_
        calculation1 += calculation1_
        calculation2 += calculation2_
        if l == 0:
            result_cardinality = len(result)
            patterns_found.append(result)
            num_patterns_found.append(result_cardinality)
    t1 /= num_loops
    t2 /= num_loops
    calculation1 /= num_loops
    calculation2 /= num_loops

    execution_time1.append(t1)
    num_calculation1.append(calculation1)
    execution_time2.append(t2)
    num_calculation2.append(calculation2)





output_path = r'../../../OutputData/AdultDataset/thc.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("execution time\n")
for n in range(len(Thc_list)):
    output_file.write('{} {} {}\n'.format(Thc_list[n], execution_time1[n], execution_time2[n]))


output_file.write("\n\nnumber of calculations\n")
for n in range(len(Thc_list)):
    output_file.write('{} {} {}\n'.format(Thc_list[n], num_calculation1[n], num_calculation2[n]))


output_file.write("\n\nnumber of patterns found\n")
for n in range(len(Thc_list)):
    output_file.write('{} {} \n {}\n'.format(Thc_list[n], num_patterns_found[n], patterns_found[n]))




plt.plot(Thc_list, execution_time1, label="new algorithm", color='blue', linewidth = 3.4)
plt.plot(Thc_list, execution_time2, label="naive algorithm", color='orange', linewidth = 3.4)

plt.xlabel('threshold of cardinality')
plt.ylabel('execution time (s)')
plt.title('AdultDataset')
plt.xticks(Thc_list)
plt.xscale("log")
plt.legend()
plt.savefig("../../../OutputData/AdultDataset/thc_time.png")
plt.show()


fig, ax = plt.subplots()
plt.plot(Thc_list, num_calculation1, label="new algorithm", color='blue', linewidth = 3.4)
plt.plot(Thc_list, num_calculation2, label="naive algorithm", color='orange', linewidth = 3.4)
plt.xlabel('threshold of cardinality')
plt.ylabel('number of cardinality calculations (K)')
plt.title('AdultDataset')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


plt.xticks(Thc_list)
plt.xscale("log")
plt.legend()
plt.savefig("../../../OutputData/AdultDataset/thc_calculations.png")
plt.show()

plt.close()
plt.clf()

