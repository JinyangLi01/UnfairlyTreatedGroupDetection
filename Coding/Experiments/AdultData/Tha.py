"""
This script is to do experiment on the threshold of minority group accuracy.

two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: diff_acc = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
threshold of minority group accuracy: overall acc - diff_acc

Other parameters:
CleanAdult2.csv
selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass']
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
    num_pattern_skipped_mis_c1, num_pattern_skipped_whole_c1, pattern_with_low_accuracy2, \
    num_patterns_checked2, execution_time2, num_pattern_skipped_mis_c2, num_pattern_skipped_whole_c2, \
    overall_acc, Tha, mis_class_data = \
        wholeprocess.WholeProcessWithTwoAlgorithms(original_data_file, selected_attributes, Thc, time_limit,
                                                   att_to_predict, difference_from_overall_acc)

    if execution_time1 > time_limit:
        print("new alg exceeds time limit")
    if execution_time2 > time_limit:
        print("naive alg exceeds time limit")
    elif sanity_check is False:
        print("sanity check failes!")

    return execution_time1, num_patterns_checked1, num_pattern_skipped_mis_c1, num_pattern_skipped_whole_c1, \
           execution_time2, num_patterns_checked2, num_pattern_skipped_mis_c2, num_pattern_skipped_whole_c2, \
           pattern_with_low_accuracy1



selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass']
diff_acc = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
original_data_file = "../../../InputData/AdultDataset/CleanAdult2.csv"
att_to_predict = 'income'
time_limit = 10*60
execution_time1 = list()
execution_time2 = list()
num_patterns_checked1 = list()
num_patterns_checked2 = list()
num_pattern_skipped_mis_c1 = list()
num_pattern_skipped_mis_c2 = list()
num_pattern_skipped_whole_c1 = list()
num_pattern_skipped_whole_c2 = list()
num_patterns_found = list()
patterns_found = list()
thc = 30
num_loops = 1

for dif in diff_acc:
    print("dif = {}".format(dif))
    t1, t2, check1, check2, skipmis1, skipmis2, skipwhole1, skipwhole2 = 0, 0, 0, 0, 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, check1_, skipmis1_, skipwhole1_, t2_, check2_, skipmis2_, skipwhole2_, result = \
        GridSearch(original_data_file, selected_attributes, thc, time_limit, att_to_predict, dif)
        t1 += t1_
        t2 += t2_
        check1 += check1_
        check2 += check2_
        skipmis1 += skipmis1_
        skipmis2 += skipmis2_
        skipwhole1 += skipwhole1_
        skipwhole2 += skipwhole2_
        if l == 0:
            result_cardinality = len(result)
            patterns_found.append(result)
            num_patterns_found.append(result_cardinality)
    t1 /= num_loops
    t2 /= num_loops
    check1 /= num_loops
    check2 /= num_loops
    skipmis1 /= num_loops
    skipmis2 /= num_loops
    skipwhole1 /= num_loops
    skipwhole2 /= num_loops

    execution_time1.append(t1)
    num_patterns_checked1.append(check1)
    num_pattern_skipped_mis_c1.append(skipmis1)
    num_pattern_skipped_whole_c1.append(skipwhole1)
    execution_time2.append(t2)
    num_patterns_checked2.append(check2)
    num_pattern_skipped_mis_c2.append(skipmis2)
    num_pattern_skipped_whole_c2.append(skipwhole2)




output_path = r'../../../OutputData/AdultDataset/tha.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("execution time\n")
for n in range(len(diff_acc)):
    output_file.write('{} {} {}\n'.format(diff_acc[n], execution_time1[n], execution_time2[n]))


output_file.write("\n\nnumber of patterns checked\n")
for n in range(len(diff_acc)):
    output_file.write('{} {} {}\n'.format(diff_acc[n], num_patterns_checked1[n], num_patterns_checked2[n]))


output_file.write("\n\nnumber of patterns skipped mis\n")
for n in range(len(diff_acc)):
    output_file.write('{} {} {}\n'.format(diff_acc[n], num_pattern_skipped_mis_c1[n], num_pattern_skipped_mis_c2[n]))




output_file.write("\n\nnumber of patterns skipped whole\n")
for n in range(len(diff_acc)):
    output_file.write('{} {} {}\n'.format(diff_acc[n], num_pattern_skipped_whole_c1[n], num_pattern_skipped_whole_c2[n]))


output_file.write("\n\nnumber of patterns found\n")
for n in range(len(diff_acc)):
    output_file.write('{} {} \n {}\n'.format(diff_acc[n], num_patterns_found[n], patterns_found[n]))





plt.plot(diff_acc, execution_time1, label="new algorithm", color='blue', linewidth = 3.4)
plt.plot(diff_acc, execution_time2, label="naive algorithm", color='orange', linewidth = 3.4)

plt.xlabel('threshold of accuracy')
plt.ylabel('execution time (s)')
#plt.title('Title???')
plt.xticks(diff_acc)
#plt.yscale('log')
plt.legend()
plt.show()



fig, ax = plt.subplots()
plt.plot(diff_acc, num_patterns_checked1, label="new algorithm", color='blue', linewidth = 3.4)
plt.plot(diff_acc, num_patterns_checked2, label="naive algorithm", color='orange', linewidth = 3.4)
"""
plt.plot(diff_acc, num_pattern_skipped_mis_c1, label="new algorithm mis", linewidth = 3.4)
plt.plot(diff_acc, num_pattern_skipped_mis_c2, label="naive algorithm mis",  linewidth = 3.4)

plt.plot(diff_acc, num_pattern_skipped_whole_c1, label="new algorithm whole", linewidth = 3.4)
plt.plot(diff_acc, num_pattern_skipped_whole_c2, label="naive algorithm whole", linewidth = 3.4)
"""
plt.xlabel('threshold of accuracy')
plt.ylabel('number of patterns checked (K)')
#plt.title('Title???')
#ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


plt.xticks(diff_acc)
plt.legend()
plt.show()

plt.close()
plt.clf()

