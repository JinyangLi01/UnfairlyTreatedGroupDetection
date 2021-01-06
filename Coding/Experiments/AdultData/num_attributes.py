"""
This script is to do experiment on the number of attribtues.

two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: the number of attributes, from 2 to 13.

other parameters:
CleanAdult2.csv
threshold of cardinality Thc = 30
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

def GridSearch(original_data_file, Thc, num_attributes, time_limit, att_to_predict, only_new_alg=False):
    original_data = pd.read_csv(original_data_file)
    selected_attributes = original_data.columns.tolist()[:num_attributes]

    if only_new_alg:
        pattern_with_low_accuracy1, num_patterns_checked1, execution_time1, num_pattern_skipped_mis_c1, \
        num_pattern_skipped_whole_c1, OverallAccuracy, Tha, mis_class_data = \
            wholeprocess.WholeProcessWithOneAlgorithm(original_data_file, selected_attributes, Thc, time_limit,
                                                      newalg.GraphTraverse, att_to_predict)
        return execution_time1, num_patterns_checked1, num_pattern_skipped_mis_c1, num_pattern_skipped_whole_c1, \
           0, 0, 0, 0, \
           pattern_with_low_accuracy1

    sanity_check, pattern_with_low_accuracy1, num_patterns_checked1, execution_time1, \
    num_pattern_skipped_mis_c1, num_pattern_skipped_whole_c1, pattern_with_low_accuracy2, \
    num_patterns_checked2, execution_time2, num_pattern_skipped_mis_c2, num_pattern_skipped_whole_c2, \
    overall_acc, Tha, mis_class_data = \
        wholeprocess.WholeProcessWithTwoAlgorithms(original_data_file, selected_attributes, Thc, time_limit,
                                                   att_to_predict)


    if execution_time1 > time_limit:
        print("new alg exceeds time limit")
    if execution_time2 > time_limit:
        print("naive alg exceeds time limit")
    elif sanity_check is False:
        print("sanity check failes!")

    return execution_time1, num_patterns_checked1, num_pattern_skipped_mis_c1, num_pattern_skipped_whole_c1, \
           execution_time2, num_patterns_checked2, num_pattern_skipped_mis_c2, num_pattern_skipped_whole_c2, \
           pattern_with_low_accuracy1

# selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass', 'relationship', 'occupation']
Thc = 30
original_data_file = "../../../InputData/AdultDataset/CleanAdult2.csv"
att_to_predict = 'income'
time_limit = 10*60
# based on experiments with the above parameters, when number of attributes = 8, naive algorithm running time > 10min
# so for naive alg, we only do when number of attributes <= 7
num_att_max_naive = 8
num_att_min = 3
num_att_max = 14
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
num_loops = 1


for number_attributes in range(num_att_min, num_att_max_naive):
    print("number of attributes = {}".format(number_attributes))
    t1, t2, check1, check2, skipmis1, skipmis2, skipwhole1, skipwhole2 = 0, 0, 0, 0, 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, check1_, skipmis1_, skipwhole1_, t2_, check2_, skipmis2_, skipwhole2_, result = \
            GridSearch(original_data_file, Thc, number_attributes, time_limit, att_to_predict)
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


for number_attributes in range(num_att_max_naive, num_att_max):
    print("number of attributes = {}".format(number_attributes))
    t1, check1, skipmis1, skipwhole1 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, check1_, skipmis1_, skipwhole1_, _, _, _, _, result = \
            GridSearch(original_data_file, Thc, number_attributes, time_limit, att_to_predict)
        t1 += t1_
        check1 += check1_
        skipmis1 += skipmis1_
        skipwhole1 += skipwhole1_
        if l == 0:
            result_cardinality = len(result)
            patterns_found.append(result)
            num_patterns_found.append(result_cardinality)
    t1 /= num_loops
    check1 /= num_loops
    skipmis1 /= num_loops
    skipwhole1 /= num_loops

    execution_time1.append(t1)
    num_patterns_checked1.append(check1)
    num_pattern_skipped_mis_c1.append(skipmis1)
    num_pattern_skipped_whole_c1.append(skipwhole1)




output_path = r'../../../OutputData/AdultDataset/num_attribute.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("execution time\n")
for n in range(num_att_min, num_att_max_naive):
    output_file.write('{} {} {}\n'.format(n, execution_time1[n-num_att_min], execution_time2[n-num_att_min]))
for n in range(num_att_max_naive, num_att_max):
    output_file.write('{} {}\n'.format(n, execution_time1[n - num_att_max_naive]))
#output_file.write('\n'.join('{} {} {}'.format(index + num_att_min, x, y) for index, x, y in enumerate(execution_time1) and execution_time2))


output_file.write("\n\nnumber of patterns checked\n")
for n in range(num_att_min, num_att_max_naive):
    output_file.write('{} {} {}\n'.format(n, num_patterns_checked1[n-num_att_min], num_patterns_checked2[n-num_att_min]))
for n in range(num_att_max_naive, num_att_max):
    output_file.write('{} {}\n'.format(n, num_patterns_checked1[n-num_att_max_naive]))

#output_file.write('\n'.join('{} {} {}'.format(index + num_att_min, x, y) for index, x, y in enumerate(num_patterns_checked1) and num_patterns_checked2))



output_file.write("\n\nnumber of patterns found\n")
for n in range(num_att_min, num_att_max_naive):
    output_file.write('{} {} \n {}\n'.format(n, num_patterns_found[n], patterns_found[n]))




# when number of attributes = 8, naive algorithm running time > 10min
# so we only use x[:6]
x_new = list(range(num_att_min, num_att_max))
x_naive = list(range(num_att_min, num_att_max_naive))


plt.plot(x_new, execution_time1, label="new algorithm", color='blue', linewidth = 3.4)
plt.plot(x_naive, execution_time2, label="naive algorithm", color='orange', linewidth = 3.4)

plt.xlabel('number of attributes')
plt.ylabel('execution time (s)')
#plt.title('Title???')
plt.xticks(x_new)
plt.legend()
plt.show()


fig, ax = plt.subplots()
plt.plot(x_new, num_patterns_checked1, label="new algorithm", color='blue', linewidth = 3.4)
plt.plot(x_naive, num_patterns_checked2, label="naive algorithm", color='orange', linewidth = 3.4)
plt.xlabel('number of attributes')
plt.ylabel('number of patterns checked (K)')
#plt.title('Title???')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

#ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.xticks(x_new)
plt.legend()
plt.show()

plt.close()
plt.clf()

