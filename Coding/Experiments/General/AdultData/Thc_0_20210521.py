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


selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass', 'relationship',
                       'occupation', 'educational-num', 'capital-gain']
Thc_list = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]
original_data_file = "../../../../InputData/AdultDataset/CleanAdult2.csv"
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
num_loops = 1

fairness_definition = 0
delta_thf = 0.3


less_attribute_data, TP, TN, FP, FN = predict.PredictWithMLReturnTPTNFPFN(original_data_file,
                                                                         selected_attributes,
                                                                         att_to_predict)

for thc in Thc_list:
    print("\nthc = {}".format(thc))
    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        pattern_with_low_fairness1, calculation1_, t1_ = newalg.GraphTraverse(less_attribute_data,
                                                                                TP, TN, FP, FN, delta_thf,
                                                                                thc, time_limit, fairness_definition)

        print("newalg, time = {} s, num_calculation = {}, num_pattern = {}".format(t1_, calculation1_, len(pattern_with_low_fairness1)), "\n",
              pattern_with_low_fairness1)
        t1 += t1_
        calculation1 += calculation1_

        pattern_with_low_fairness2, calculation2_, t2_ = naivealg.NaiveAlg(less_attribute_data,
                                                                     TP, TN, FP, FN, delta_thf,
                                                                     thc, time_limit, fairness_definition)


        print("naivealg, time = {} s, num_calculation = {}, num_pattern = {}".format(t2_, calculation2_, len(pattern_with_low_fairness2)), "\n",
              pattern_with_low_fairness2)
        t2 += t2_
        calculation2 += calculation2_

        if ComparePatternSets(pattern_with_low_fairness1, pattern_with_low_fairness2) is False:
            print("sanity check fails!")

        if l == 0:
            result_cardinality = len(pattern_with_low_fairness1)
            patterns_found.append(pattern_with_low_fairness1)
            num_patterns_found.append(result_cardinality)

    t1 /= num_loops
    t2 /= num_loops
    calculation1 /= num_loops
    calculation2 /= num_loops

    execution_time1.append(t1)
    num_calculation1.append(calculation1)
    execution_time2.append(t2)
    num_calculation2.append(calculation2)





output_path = r'../../../../OutputData/General/AdultDataset/thc.txt'
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
plt.savefig("../../../../OutputData/General/AdultDataset/thc_time.png")
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
plt.savefig("../../../../OutputData/General/AdultDataset/thc_calculations.png")
plt.show()

plt.close()
plt.clf()

