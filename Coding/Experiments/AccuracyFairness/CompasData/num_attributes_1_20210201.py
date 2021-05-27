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
from Algorithms import Predict_0_20210127 as predict
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16
plt.rc('figure', figsize=(7.1, 5.6))

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

def GridSearch(original_data_file, thc, num_attributes, time_limit, att_to_predict, only_new_alg=False):
    original_data = pd.read_csv(original_data_file)
    selected_attributes = original_data.columns.tolist()[1:num_attributes+1]
    print("selected_attributes:", selected_attributes)

    if only_new_alg:
        less_attribute_data, mis_class_data, overall_acc = predict.PredictWithML(original_data_file,
                                                                                 selected_attributes,
                                                                                 att_to_predict)
        tha = overall_acc - 0.2
        print("tha = {}, thc = {}".format(tha, thc))
        pattern_with_low_accuracy1, num_calculation1, execution_time1 = newalg.GraphTraverse(less_attribute_data,
                                                                              mis_class_data, tha,
                                                                              thc, time_limit)
        print("{} patterns with low accuracy: \n {}".format(len(pattern_with_low_accuracy1), pattern_with_low_accuracy1))
        return execution_time1, num_calculation1, 0, 0, pattern_with_low_accuracy1


    less_attribute_data, mis_class_data, overall_acc = predict.PredictWithML(original_data_file,
                                                                             selected_attributes,
                                                                             att_to_predict)
    tha = overall_acc - 0.2
    print("tha = {}, thc = {}".format(tha, thc))
    pattern_with_low_accuracy1, num_calculation1, execution_time1 = newalg.GraphTraverse(less_attribute_data,
                                                                                         mis_class_data, tha,
                                                                                         thc, time_limit)
    print("newalg, time = {} s, num_calculation = {}".format(execution_time1, num_calculation1), "\n", pattern_with_low_accuracy1)

    pattern_with_low_accuracy2, num_calculation2, execution_time2 = naivealg.NaiveAlg(less_attribute_data,
                                                                       mis_class_data, tha,
                                                                       thc, time_limit)
    print("naivealg, time = {} s, num_calculation = {}".format(execution_time2, num_calculation2), "\n",
          pattern_with_low_accuracy2)

    if ComparePatternSets(pattern_with_low_accuracy1, pattern_with_low_accuracy2) is False:
        print("sanity check fails!")


    print("{} patterns with low accuracy: \n {}".format(len(pattern_with_low_accuracy1), pattern_with_low_accuracy1))

    if execution_time1 > time_limit:
        print("new alg exceeds time limit")
    if execution_time2 > time_limit:
        print("naive alg exceeds time limit")

    return execution_time1, num_calculation1, execution_time2, num_calculation2, \
           pattern_with_low_accuracy1

# selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass', 'relationship', 'occupation']
Thc = 10
original_data_file = "../../../../InputData/CompasData/RecidivismData_att_classified.csv"
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
    print("\n\nnumber of attributes = {}".format(number_attributes))
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
    print("\n\nnumber of attributes = {}".format(number_attributes))
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




output_path = r'../../../../OutputData/LowAccDetection/CompasDataset/num_attribute.txt'
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
plt.title('CompasDataset')
plt.xticks(x_new)
plt.legend()
plt.savefig("../../../../OutputData/LowAccDetection/CompasDataset/num_att_time.png")
plt.show()


fig, ax = plt.subplots()
plt.plot(x_new, num_calculation1, label="new algorithm", color='blue', linewidth = 3.4)
plt.plot(x_naive, num_calculation2, label="naive algorithm", color='orange', linewidth = 3.4)
plt.xlabel('number of attributes')
plt.ylabel('number of cardinality calculations (K)')
plt.title('CompasDataset')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


plt.xticks(x_new)
plt.legend()
plt.savefig("../../../../OutputData/LowAccDetection/CompasDataset/num_att_calculations.png")
plt.show()

plt.close()
plt.clf()

