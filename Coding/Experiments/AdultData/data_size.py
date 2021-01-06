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
from Algorithms import NewAlg_0_20201128 as newalg
from Algorithms import NaiveAlg_0_20201111 as naivealg
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def thousands_formatter(x, pos):
    return int(x/1000)

def GridSearch(original_data_file_pathpre, datasize, Thc, selected_attributes, att_to_predict):
    original_data_file = original_data_file_pathpre + str(datasize) + ".csv"
    original_data = pd.read_csv(original_data_file)

    sanity_check, pattern_with_low_accuracy1, num_patterns_checked1, execution_time1, \
    pattern_with_low_accuracy2, num_patterns_checked2, execution_time2, overall_acc, Tha, mis_class_data = \
        wholeprocess.WholeProcessWithTwoAlgorithms(original_data_file, selected_attributes, Thc, time_limit, att_to_predict)


    if execution_time1 > time_limit:
        print("new alg exceeds time limit")
    if execution_time2 > time_limit:
        print("naive alg exceeds time limit")
    elif sanity_check is False:
        print("sanity check failes for datasize {}".format(datasize))

    return execution_time1, execution_time2, num_patterns_checked1, num_patterns_checked2


selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass']
data_sizes = [100, 500, 1000, 5000, 10000, 20000, 30000, 40000]
Thc = 30
original_data_file_pathprefix = "../../../InputData/DifferentDataSizes/"
att_to_predict = 'income'
time_limit = 10*60
# based on experiments with the above parameters, when number of attributes = 8, naive algorithm running time > 10min
# so for naive alg, we only do when number of attributes <= 7
execution_time1 = list()
execution_time2 = list()
num_patterns_checked1 = list()
num_patterns_checked2 = list()


for datasize in data_sizes:
    print('datasize = {}'.format(datasize))
    t1, t2, n1, n2 = GridSearch(original_data_file_pathprefix, datasize, Thc, selected_attributes, att_to_predict)
    execution_time1.append(t1)
    num_patterns_checked1.append(n1)
    execution_time2.append(t2)
    num_patterns_checked2.append(n2)




output_path = r'../../../OutputData/AdultDataset/data_size.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("execution time\n")
for n in range(len(data_sizes)):
    output_file.write('{} {} {}\n'.format(data_sizes[n], execution_time1[n], execution_time2[n]))
#output_file.write('\n'.join('{} {} {}'.format(index + num_att_min, x, y) for index, x, y in enumerate(execution_time1) and execution_time2))


output_file.write("\n\nnumber of patterns checked\n")
for n in range(len(data_sizes)):
    output_file.write('{} {} {}\n'.format(data_sizes[n], num_patterns_checked1[n], num_patterns_checked2[n]))



plt.plot(data_sizes, execution_time1, label="new algorithm", color='blue', linewidth = 3.4)
plt.plot(data_sizes, execution_time2, label="naive algorithm", color='orange', linewidth = 3.4)

plt.xlabel('data size')
plt.ylabel('execution time (s)')
#plt.title('Title???')
plt.xticks([0, 5000, 10000, 20000, 30000, 40000])
plt.legend()
plt.show()


fig, ax = plt.subplots()
plt.plot(data_sizes, num_patterns_checked1, label="new algorithm", color='blue', linewidth=3.4)
plt.plot(data_sizes, num_patterns_checked2, label="naive algorithm", color='orange', linewidth=3.4)
plt.xlabel('data size')
plt.ylabel('number of patterns checked (K)')
#plt.title('Title???')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

#ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.xticks([0, 5000, 10000, 20000, 30000, 40000])
plt.legend()
plt.show()

plt.close()
plt.clf()

