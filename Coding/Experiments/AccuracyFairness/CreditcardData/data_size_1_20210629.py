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
from Algorithms import NewAlg_1_20210529 as newalg
from Algorithms import NaiveAlg_1_20210528 as naivealg
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

def GridSearch(original_data_file_pathpre, datasize, thc, selected_attributes):
    original_data_file = original_data_file_pathpre + str(datasize) + "_testdata.csv"
    mis_data_file = original_data_file_pathpre + str(datasize) + "_mis.csv"

    original_data = pd.read_csv(original_data_file)
    mis_data = pd.read_csv(mis_data_file)

    less_attribute_data = original_data[selected_attributes]
    mis_class_data = mis_data[selected_attributes]
    overall_acc = 1 - len(mis_class_data) / len(less_attribute_data)

    tha = overall_acc - 0.2

    print("tha = {}, thc = {}".format(tha, thc))
    pattern_with_low_accuracy1, num_calculation1, execution_time1 = newalg.GraphTraverse(less_attribute_data,
                                                                                         mis_class_data, tha,
                                                                                         thc, time_limit)
    print("newalg, time = {} s, num_calculation = {}".format(execution_time1, num_calculation1), "\n",
          pattern_with_low_accuracy1)

    pattern_with_low_accuracy2, num_calculation2, execution_time2 = naivealg.NaiveAlg(less_attribute_data,
                                                                                      mis_class_data, tha,
                                                                                      thc, time_limit)
    print("naivealg, time = {} s, num_calculation = {}".format(execution_time2, num_calculation2), "\n",
          pattern_with_low_accuracy2)

    if ComparePatternSets(pattern_with_low_accuracy1, pattern_with_low_accuracy2) is False:
        print("sanity check fails!")

    print("{} patterns with low accuracy: \n {}".format(len(pattern_with_low_accuracy1), pattern_with_low_accuracy1))


    if execution_time1 > time_limit:
        print("optimized alg exceeds time limit")
    if execution_time2 > time_limit:
        print("naive alg exceeds time limit")


    return execution_time1, num_calculation1, execution_time2, num_calculation2, pattern_with_low_accuracy1


selected_attributes = ['limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_0']
data_sizes = [30000, 35000, 40000, 45000, 50000, 55000, 60000]
Thc = 50
original_data_file_pathprefix = "../../../../InputData/CreditcardDataset/LargeDatasets_cat/"


time_limit = 10*60
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
    print('\n\ndatasize = {}'.format(datasize))
    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, calculation1_, t2_, calculation2_, result = \
            GridSearch(original_data_file_pathprefix, datasize, Thc, selected_attributes)
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




output_path = r'../../../../OutputData/LowAccDetection/CreditcardDataset/data_size.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("execution time\n")
for n in range(len(data_sizes)):
    output_file.write('{} {} {}\n'.format(data_sizes[n], execution_time1[n], execution_time2[n]))


output_file.write("\n\nnumber of patterns visited\n")
for n in range(len(data_sizes)):
    output_file.write('{} {} {}\n'.format(data_sizes[n], num_patterns_checked1[n], num_patterns_checked2[n]))


fig, ax = plt.subplots()
plt.plot(data_sizes, execution_time1, label="optimized algorithm", color='blue', linewidth = 3.4)
plt.plot(data_sizes, execution_time2, label="naive algorithm", color='orange', linewidth = 3.4)
plt.xlabel('data size (K)')
plt.ylabel('execution time (s)')
plt.xticks([30000, 40000, 50000, 60000])
ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.subplots_adjust(bottom=0.15, left=0.18)
plt.legend()
plt.savefig("../../../../OutputData/LowAccDetection/CreditcardDataset/datasize_time.png")
plt.show()


fig, ax = plt.subplots()
plt.plot(data_sizes, num_patterns_checked1, label="optimized algorithm", color='blue', linewidth=3.4)
plt.plot(data_sizes, num_patterns_checked2, label="naive algorithm", color='orange', linewidth=3.4)
plt.xlabel('data size (K)')
plt.xticks([30000, 40000, 50000, 60000])
ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.ylabel('number of patterns visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.subplots_adjust(bottom=0.15, left=0.18)
plt.legend()
plt.savefig("../../../../OutputData/LowAccDetection/CreditcardDataset/datasize_calculations.png")
plt.show()


plt.close()
plt.clf()

