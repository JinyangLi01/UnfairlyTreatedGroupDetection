"""
This script is to do experiment on the threshold of minority group sizes.

two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: Thc, from 1 to 1000

Other parameters:
RecidivismData_att_classified.txt: 6889 rows
selected_attributes = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree']
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

def GridSearch(original_data_file, selected_attributes, Thc, time_limit, att_to_predict, only_new_alg=False):

    sanity_check, pattern_with_low_accuracy1, num_patterns_checked1, execution_time1, \
    pattern_with_low_accuracy2, num_patterns_checked2, execution_time2, overall_acc, Tha, mis_class_data = \
        wholeprocess.WholeProcessWithTwoAlgorithms(original_data_file, selected_attributes, Thc, time_limit,
                                                   att_to_predict)

    if execution_time1 > time_limit:
        print("new alg exceeds time limit")
    if execution_time2 > time_limit:
        print("naive alg exceeds time limit")
    elif sanity_check is False:
        print("sanity check failes!")

    return execution_time1, execution_time2, num_patterns_checked1, num_patterns_checked2


selected_attributes = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree']
Thc_list = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]
original_data_file = "../../../InputData/RecidivismData/RecidivismData_att_classified.csv"
att_to_predict = 'is_recid'
time_limit = 10*60
execution_time1 = list()
execution_time2 = list()
num_patterns_checked1 = list()
num_patterns_checked2 = list()


for thc in Thc_list:
    print("Thc = {}".format(thc))
    t1, t2, n1, n2 = GridSearch(original_data_file, selected_attributes, thc, time_limit, att_to_predict)
    execution_time1.append(t1)
    num_patterns_checked1.append(n1)
    execution_time2.append(t2)
    num_patterns_checked2.append(n2)




output_path = r'../../../OutputData/RecidivismDataset/thc.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("execution time\n")
for n in range(len(Thc_list)):
    output_file.write('{} {} {}\n'.format(Thc_list[n], execution_time1[n], execution_time2[n]))


output_file.write("\n\nnumber of patterns checked\n")
for n in range(len(Thc_list)):
    output_file.write('{} {} {}\n'.format(Thc_list[n], num_patterns_checked1[n], num_patterns_checked2[n]))





plt.plot(Thc_list, execution_time1, label="new algorithm", color='blue', linewidth = 3.4)
plt.plot(Thc_list, execution_time2, label="naive algorithm", color='orange', linewidth = 3.4)

plt.xlabel('threshold of cardinality')
plt.ylabel('execution time (s)')
#plt.title('Title???')
plt.xticks(Thc_list)
plt.xscale("log")
plt.legend()
plt.show()


fig, ax = plt.subplots()
plt.plot(Thc_list, num_patterns_checked1, label="new algorithm", color='blue', linewidth = 3.4)
plt.plot(Thc_list, num_patterns_checked2, label="naive algorithm", color='orange', linewidth = 3.4)
plt.xlabel('threshold of cardinality')
plt.ylabel('number of patterns checked (K)')
#plt.title('Title???')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

#ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.xticks(Thc_list)
plt.xscale("log")
plt.legend()
plt.show()

plt.close()
plt.clf()

