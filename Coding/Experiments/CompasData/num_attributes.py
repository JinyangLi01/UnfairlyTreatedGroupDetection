"""
This script is to do experiment on the number of attribtues.

two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: the number of attributes, from 2 to 13.

other parameters:
RecidivismData_att_classified.txt: 6889 rows
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
        pattern_with_low_accuracy1, num_calculation1, execution_time1, overall_acc, Tha, mis_class_data = \
            wholeprocess.WholeProcessWithOneAlgorithm(original_data_file, selected_attributes, Thc,
                                                      time_limit, newalg.GraphTraverse, att_to_predict)
        return execution_time1, num_calculation1, 0, 0, pattern_with_low_accuracy1

    sanity_check, pattern_with_low_accuracy1, num_calculation1, execution_time1, \
    num_pattern_skipped_mis_c1, num_pattern_skipped_whole_c1, pattern_with_low_accuracy2, \
    num_calculation2, execution_time2, \
    overall_acc, Tha, mis_class_data = \
        wholeprocess.WholeProcessWithTwoAlgorithms(original_data_file, selected_attributes, Thc, time_limit, att_to_predict)

    print("{} patterns with low accuracy: \n {}".format(len(pattern_with_low_accuracy1), pattern_with_low_accuracy1))

    if execution_time1 > time_limit:
        print("new alg exceeds time limit")
    if execution_time2 > time_limit:
        print("naive alg exceeds time limit")
    elif sanity_check is False:
        print("sanity check fails!")

    return execution_time1, num_calculation1, execution_time2, num_calculation2, \
           pattern_with_low_accuracy1


# selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass', 'relationship', 'occupation']
Thc = 30
original_data_file = "../../../InputData/RecidivismData/RecidivismData_att_classified.csv"
att_to_predict = 'is_recid'
time_limit = 30*60
# based on experiments with the above parameters, when number of attributes = 8, naive algorithm running time > 10min
# so for naive alg, we only do when number of attributes <= 7
num_att_max_naive = 6
num_att_min = 3
num_att_max = 12
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



for number_attributes in range(num_att_min, num_att_max_naive):
    print("number of attributes = {}".format(number_attributes))
    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, calculation1_,  t2_, calculation2_, result = \
            GridSearch(original_data_file, Thc, number_attributes, time_limit, att_to_predict)
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



for number_attributes in range(num_att_max_naive, num_att_max):
    print("number of attributes = {}".format(number_attributes))
    t1, calculation1 = 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, calculation1_,  _, _, result = \
            GridSearch(original_data_file, Thc, number_attributes, time_limit, att_to_predict, True)
        t1 += t1_
        calculation1 += calculation1_
        if l == 0:
            result_cardinality = len(result)
            patterns_found.append(result)
            num_patterns_found.append(result_cardinality)
    t1 /= num_loops
    calculation1 /= num_loops

    execution_time1.append(t1)
    num_calculation1.append(calculation1)





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
    output_file.write('{} {} {}\n'.format(n, num_calculation1[n-num_att_min], num_calculation2[n-num_att_min]))
for n in range(num_att_max_naive, num_att_max):
    output_file.write('{} {}\n'.format(n, num_calculation1[n-num_att_max_naive]))

#output_file.write('\n'.join('{} {} {}'.format(index + num_att_min, x, y) for index, x, y in enumerate(num_calculation1) and num_calculation2))



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
plt.title('CompasData')
plt.xticks(x_new)
plt.legend()
plt.savefig("../../../OutputData/RecidivismDataset/time.png")
plt.show()

fig, ax = plt.subplots()
plt.plot(x_new, num_calculation1, label="new algorithm", color='blue', linewidth = 3.4)
plt.plot(x_naive, num_calculation2, label="naive algorithm", color='orange', linewidth = 3.4)
plt.xlabel('number of attributes')
plt.ylabel('number of cardinality calculations (K)')
plt.title('CompasData')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


plt.xticks(x_new)
plt.legend()
plt.savefig("../../../OutputData/RecidivismDataset/num_att_calculations.png")
plt.show()

plt.close()
plt.clf()

