"""
This script is to do experiment on the data size.

CleanAdult2.txt: 45222 rows
data sizes: 100, 500, 1000, 5000, 10000, 40000
selected randomly, and generate files in InputData/DifferentDataSizes/



two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: data sizes: 100, 500, 1000, 5000, 10000, 40000

Other parameters:
selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass']
threshold of cardinality Thc = 30
threshold of minority group accuracy: overall acc - 20

"""


import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlgGeneral_0_20210412 as newalg
from Algorithms import NaiveAlgGeneral_0_20210515 as naivealg
from Algorithms import Predict_0_20210127 as predict
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


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

def GridSearch(original_data_file_pathpre, datasize, thc, selected_attributes, att_to_predict):
    original_data_file = original_data_file_pathpre + str(datasize) + ".csv"
    fairness_definition = 0
    delta_thf = 0.2

    less_attribute_data, TP, TN, FP, FN = predict.PredictWithMLReturnTPTNFPFN(original_data_file,
                                                                              selected_attributes,
                                                                              att_to_predict)

    pattern_with_low_fairness1, num_calculation1, execution_time1 = newalg.GraphTraverse(less_attribute_data,
                                                                                TP, TN, FP, FN, delta_thf,
                                                                                thc, time_limit, fairness_definition)

    print("newalg, time = {} s, num_calculation = {}".format(execution_time1, num_calculation1), "\n",
          pattern_with_low_fairness1)

    pattern_with_low_fairness2, num_calculation2, execution_time2 = naivealg.NaiveAlg(less_attribute_data,
                                                                     TP, TN, FP, FN, delta_thf,
                                                                     thc, time_limit, fairness_definition)

    print("naivealg, time = {} s, num_calculation = {}".format(execution_time2, num_calculation2), "\n",
          pattern_with_low_fairness2)

    if ComparePatternSets(pattern_with_low_fairness1, pattern_with_low_fairness2) is False:
        print("sanity check fails!")

    print("{} patterns with low accuracy: \n {}".format(len(pattern_with_low_fairness1), pattern_with_low_fairness2))


    if execution_time1 > time_limit:
        print("new alg exceeds time limit")
    if execution_time2 > time_limit:
        print("naive alg exceeds time limit")


    return execution_time1, num_calculation1, execution_time2, num_calculation2, pattern_with_low_fairness1


selected_attributes = ['limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_0', 'pay_2',
                       'pay_3']
data_sizes = [10000, 20000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]

Thc = 300
original_data_file_pathprefix = "../../../../InputData/CreditcardDataset/LargerDatasets/"
att_to_predict = 'default payment next month'
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
    print('\n\ndatasize = {}'.format(datasize))
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




output_path = r'../../../../OutputData/General/CreditcardDataset/data_size.txt'
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
plt.title('CreditcardDataset')
plt.xticks([10000, 20000, 30000, 40000, 50000, 60000])
ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.legend()
plt.savefig("../../../../OutputData/General/CreditcardDataset/datasize_time.png")
plt.show()


fig, ax = plt.subplots()
plt.plot(data_sizes, num_patterns_checked1, label="new algorithm", color='blue', linewidth=3.4)
plt.plot(data_sizes, num_patterns_checked2, label="naive algorithm", color='orange', linewidth=3.4)
plt.xlabel('data size (K)')
plt.xticks([10000, 20000, 30000, 40000, 50000, 60000])
ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.ylabel('number of cardinality calculations (K)')
plt.title('CreditcardDataset')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.legend()
plt.savefig("../../../../OutputData/General/CreditcardDataset/datasize_calculations.png")
plt.show()


plt.close()
plt.clf()

