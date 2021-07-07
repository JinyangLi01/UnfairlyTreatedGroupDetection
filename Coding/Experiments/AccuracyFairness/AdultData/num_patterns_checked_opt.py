"""
This script is to do experiment on the number of attribtues.

two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: the number of attributes, from 2 to 13.

other parameters:
CleanAdult2.csv
size threshold Thc = 50
threshold of minority group accuracy: overall acc - 20
"""


import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlg_1_20210529 as newalg

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



def DFSattributes(cur, last, comb, pattern, all_p, mcdes, attributes):
    # print("DFS", attributes)
    if cur == last:
        # print("comb[{}] = {}".format(cur, comb[cur]))
        # print("{} {}".format(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max'])))
        for a in range(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max']) + 1):
            s = pattern.copy()
            s[comb[cur]] = a
            all_p.append(s)
        return
    else:
        # print("comb[{}] = {}".format(cur, comb[cur]))
        # print("{} {}".format(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max'])))
        for a in range(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max']) + 1):
            s = pattern.copy()
            s[comb[cur]] = a
            DFSattributes(cur + 1, last, comb, s, all_p, mcdes, attributes)


def AllPatternsInComb(comb, NumAttribute, mcdes, attributes):  # comb = [1,4]
    # print("All", attributes)
    all_p = []
    pattern = [-1] * NumAttribute
    DFSattributes(0, len(comb) - 1, comb, pattern, all_p, mcdes, attributes)
    return all_p


thc = 50

original_data_file = "../../../../InputData/AdultDataset/ForClassification/CleanAdult_numerical_testdata_cat.csv"
original_data = pd.read_csv(original_data_file)
mis_data_file = "../../../../InputData/AdultDataset/ForClassification/CleanAdult_numerical_mis_cat.csv"
mis_data = pd.read_csv(mis_data_file)


all_attributes = ['age','workclass','education','educational-num',
              'marital-status', 'occupation','relationship','race','gender',
              'capital-gain','capital-loss','hours-per-week', 'native-country']



### compute total number of att

number_attributes = 12
selected_attributes = all_attributes[:number_attributes]


whole_data_frame = original_data.describe()
total_num_patterns = 1
for at in selected_attributes:
    total_num_patterns = total_num_patterns * (whole_data_frame[at]['max'] - whole_data_frame[at]['min'] + 2)

print("total_num_patterns = {}".format(total_num_patterns))


#
# time_limit = 10*60
# number_attributes = 12
# selected_attributes = all_attributes[:number_attributes]
# print("{} attributes: {}".format(number_attributes, selected_attributes))
#
# less_attribute_data = original_data[selected_attributes]
# mis_class_data = mis_data[selected_attributes]
# overall_acc = 1 - len(mis_class_data) / len(less_attribute_data)
# tha = overall_acc - 0.2
#
#
# print("tha = {}, thc = {}".format(tha, thc))
# pattern_with_low_accuracy1, num_calculation1, execution_time1 = newalg.GraphTraverse(less_attribute_data,
#                                                                       mis_class_data, tha,
#                                                                       thc, time_limit)
#
# print("execution time={}, num patterns check = {}".format(execution_time1, num_calculation1))
#
# print("{} patterns with low accuracy: \n {}".format(len(pattern_with_low_accuracy1), pattern_with_low_accuracy1))
#
#
#


#
#
#
#
# output_path = r'../../../../OutputData/LowAccDetection/AdultDataset/num_attribute.txt'
# output_file = open(output_path, "w")
# num_lines = len(execution_time1)
#
# output_file.write("overall accuracy: {}\n".format(overall_acc))
#
# output_file.write("execution time\n")
# for n in range(num_att_min, num_att_max_naive):
#     output_file.write('{} {} {}\n'.format(n, execution_time1[n-num_att_min], execution_time2[n-num_att_min]))
# for n in range(num_att_max_naive, num_att_max):
#     output_file.write('{} {}\n'.format(n, execution_time1[n - num_att_max_naive]))
# #output_file.write('\n'.join('{} {} {}'.format(index + num_att_min, x, y) for index, x, y in enumerate(execution_time1) and execution_time2))
#
#
# output_file.write("\n\nnumber of patterns checked\n")
# for n in range(num_att_min, num_att_max_naive):
#     output_file.write('{} {} {}\n'.format(n, num_calculation1[n-num_att_min], num_calculation2[n-num_att_min]))
# for n in range(num_att_max_naive, num_att_max):
#     output_file.write('{} {}\n'.format(n, num_calculation1[n-num_att_max_naive]))
#
# #output_file.write('\n'.join('{} {} {}'.format(index + num_att_min, x, y) for index, x, y in enumerate(num_calculation1) and num_calculation2))
#
#
#
# output_file.write("\n\nnumber of patterns found\n")
# for n in range(num_att_min, num_att_max):
#     output_file.write('{} {} \n {}\n'.format(n-num_att_min, num_patterns_found[n-num_att_min], patterns_found[n-num_att_min]))
#
#
#
#
# # when number of attributes = 8, naive algorithm running time > 10min
# # so we only use x[:6]
# x_new = list(range(num_att_min, num_att_max))
# x_naive = list(range(num_att_min, num_att_max_naive))
#
#
# plt.plot(x_new, execution_time1, label="optimized algorithm", color='blue', linewidth = 3.4)
# plt.plot(x_naive, execution_time2, label="naive algorithm", color='orange', linewidth = 3.4)
#
# plt.xlabel('number of attributes')
# plt.ylabel('execution time (s)')
# plt.xticks(x_new)
# plt.subplots_adjust(bottom=0.15, left=0.18)
# plt.legend()
# plt.savefig("../../../../OutputData/LowAccDetection/AdultDataset/num_att_time.png")
# plt.show()
#
#
# fig, ax = plt.subplots()
# plt.plot(x_new, num_calculation1, label="optimized algorithm", color='blue', linewidth = 3.4)
# plt.plot(x_naive, num_calculation2, label="naive algorithm", color='orange', linewidth = 3.4)
# plt.xlabel('number of attributes')
# plt.ylabel('number of patterns visited (K)')
# ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
#
#
# plt.xticks(x_new)
# plt.subplots_adjust(bottom=0.15, left=0.18)
# plt.legend()
# plt.savefig("../../../../OutputData/LowAccDetection/AdultDataset/num_att_calculations.png")
# plt.show()
#
# plt.close()
# plt.clf()
#
