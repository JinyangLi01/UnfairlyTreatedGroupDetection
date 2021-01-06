"""
This script is to do experiment on the threshold of minority group accuracy.

two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: diff_acc = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
threshold of minority group accuracy: overall acc - diff_acc

Other parameters:
CleanAdult2.csv
selected_attributes = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree']
Thc = 30

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

def GridSearch(original_data_file, selected_attributes, Thc, time_limit, att_to_predict, difference_from_overall_acc=0.2):

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

    print(pattern_with_low_accuracy1, pattern_with_low_accuracy2)

    return execution_time1, execution_time2, num_patterns_checked1, num_patterns_checked2


selected_attributes = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree']
diff_acc = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
original_data_file = "../../../InputData/RecidivismData/RecidivismData_att_classified.csv"
att_to_predict = 'is_recid'
time_limit = 10*60
execution_time1 = list()
execution_time2 = list()
num_patterns_checked1 = list()
num_patterns_checked2 = list()
thc = 30

for dif in diff_acc:
    print("dif = {}".format(dif))
    t1, t2, n1, n2 = GridSearch(original_data_file, selected_attributes, thc, time_limit, att_to_predict)
    execution_time1.append(t1)
    num_patterns_checked1.append(n1)
    execution_time2.append(t2)
    num_patterns_checked2.append(n2)




output_path = r'../../../OutputData/RecidivismDataset/tha.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("execution time\n")
for n in range(len(diff_acc)):
    output_file.write('{} {} {}\n'.format(diff_acc[n], execution_time1[n], execution_time2[n]))


output_file.write("\n\nnumber of patterns checked\n")
for n in range(len(diff_acc)):
    output_file.write('{} {} {}\n'.format(diff_acc[n], num_patterns_checked1[n], num_patterns_checked2[n]))





plt.plot(diff_acc, execution_time1, label="new algorithm", color='blue', linewidth = 3.4)
plt.plot(diff_acc, execution_time2, label="naive algorithm", color='orange', linewidth = 3.4)

plt.xlabel('threshold of accuracy')
plt.ylabel('execution time (s)')
#plt.title('Title???')
plt.xticks(diff_acc)
plt.legend()
plt.show()


fig, ax = plt.subplots()
plt.plot(diff_acc, num_patterns_checked1, label="new algorithm", color='blue', linewidth = 3.4)
plt.plot(diff_acc, num_patterns_checked2, label="naive algorithm", color='orange', linewidth = 3.4)
plt.xlabel('threshold of accuracy')
plt.ylabel('number of patterns checked (K)')
#plt.title('Title???')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

#ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.xticks(diff_acc)
plt.legend()
plt.show()

plt.close()
plt.clf()

