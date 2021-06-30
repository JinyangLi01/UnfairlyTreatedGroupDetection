"""
This script is to do experiment on the data size.

RecidivismData_att_classified.txt: 6889 rows
data sizes: 100, 500, 1000, 2000, 3000, 4000, 5000, 6000
selected randomly, and generate files in InputData/RecidivismData/DifferentDataSizes/



two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: data sizes: 100, 500, 1000, 2000, 3000, 4000, 5000, 6000

Other parameters:
selected_attributes = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree']
size threshold Thc = 30
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

def GridSearch(original_data_file_pathpre, datasize, Thc, selected_attributes, att_to_predict):
    original_data_file = original_data_file_pathpre + str(datasize) + ".csv"

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

    return execution_time1, num_calculation1, execution_time2, num_calculation2, pattern_with_low_accuracy1



selected_attributes = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree', 'decile_score', 'c_days_from_compas_C']
data_sizes = [6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
Thc = 10
original_data_file_pathprefix = "../../../../InputData/CompasData/LargerDatasets/"
att_to_predict = 'is_recid'
time_limit = 20*60
# based on experiments with the above parameters, when number of attributes = 8, naive algorithm running time > 10min
# so for naive alg, we only do when number of attributes <= 7
execution_time1 = list()
execution_time2 = list()
num_patterns_checked1 = list()
num_patterns_checked2 = list()
num_patterns_found = list()
patterns_found = list()
num_loops = 1


for datasize in data_sizes:
    print('datasize = {}'.format(datasize))
    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, calculation1_, t2_, calculation2_, result = \
            GridSearch(original_data_file_pathprefix, datasize, Thc, selected_attributes, att_to_predict)
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
    num_patterns_checked1.append(calculation1)
    execution_time2.append(t2)
    num_patterns_checked2.append(calculation2)




output_path = r'../../../../OutputData/LowAccDetection/CompasDataset/data_size.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("execution time\n")
for n in range(len(data_sizes)):
    output_file.write('{} {} {}\n'.format(data_sizes[n], execution_time1[n], execution_time2[n]))


output_file.write("\n\nnumber of calculations\n")
for n in range(len(data_sizes)):
    output_file.write('{} {} {}\n'.format(data_sizes[n], num_patterns_checked1[n], num_patterns_checked2[n]))


fig, ax = plt.subplots()
plt.plot(data_sizes, execution_time1, label="new algorithm", color='blue', linewidth = 3.4)
plt.plot(data_sizes, execution_time2, label="naive algorithm", color='orange', linewidth = 3.4)
plt.xlabel('data size (K)')
plt.ylabel('execution time (s)')
plt.title('CompasDataset')
plt.xticks(data_sizes)
ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.legend()
plt.savefig("../../../OutputData/RecidivismDataset/datasize_time.png")
plt.show()


fig, ax = plt.subplots()
plt.plot(data_sizes, num_patterns_checked1, label="new algorithm", color='blue', linewidth=3.4)
plt.plot(data_sizes, num_patterns_checked2, label="naive algorithm", color='orange', linewidth=3.4)
plt.xlabel('data size (K)')
plt.ylabel('number of cardinality calculations (K)')
plt.title('CompasDataset')
plt.xticks(data_sizes)
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.legend()
plt.savefig("../../../OutputData/RecidivismDataset/datasize_calculations.png")
plt.show()


plt.close()
plt.clf()

