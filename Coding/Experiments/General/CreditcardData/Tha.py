"""
This script is to do experiment on the threshold of minority group accuracy.

two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: diff_acc = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
threshold of minority group accuracy: overall acc - diff_acc

Other parameters:
"../../../InputData/CreditcardDataset/credit_card_clients_categorized.csv"
selected_attributes = ['limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_0']
Thc = 10

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
    pattern_with_low_accuracy, num_calculation, execution_time, overall_acc, Tha, mis_class_data = \
        wholeprocess.WholeProcessWithOneAlgorithm(original_data_file, selected_attributes, Thc, time_limit,
                                                  newalg.GraphTraverse, att_to_predict, difference_from_overall_acc)

    print("{} patterns with low accuracy: \n {}".format(len(pattern_with_low_accuracy), pattern_with_low_accuracy))


    if execution_time > time_limit:
        print("new alg exceeds time limit")

    return execution_time, num_calculation, pattern_with_low_accuracy


selected_attributes = ['limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_0', 'pay_2',
                       'pay_3', 'pay_4', 'pay_5', 'pay_6']


diff_acc = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
original_data_file = "../../../../InputData/CreditcardDataset/credit_card_clients_categorized.csv"

att_to_predict = 'default payment next month'

time_limit = 20*60
execution_time = list()
num_calculations = list()
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
    t, calculations = 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t_, calculations_, result = GridSearch(original_data_file, selected_attributes,
                                                thc, time_limit, att_to_predict, dif)
        t += t_
        calculations += calculations_
        if l == 0:
            result_cardinality = len(result)
            patterns_found.append(result)
            num_patterns_found.append(result_cardinality)
    t /= num_loops
    calculations /= num_loops
    execution_time.append(t)
    num_calculations.append(calculations)




output_path = r'../../../../OutputData/CreditcardDataset/tha.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time)

output_file.write("execution time\n")
for n in range(len(diff_acc)):
    output_file.write('{} {}\n'.format(diff_acc[n], execution_time[n]))


output_file.write("\n\nnumber of calculations\n")
for n in range(len(diff_acc)):
    output_file.write('{} {}\n'.format(diff_acc[n], num_calculations[n]))


output_file.write("\n\nnumber of patterns found\n")
for n in range(len(diff_acc)):
    output_file.write('{} {} \n {}\n'.format(diff_acc[n], num_patterns_found[n], patterns_found[n]))





plt.plot(diff_acc, execution_time, label="new algorithm", color='blue', linewidth = 3.4)


plt.xlabel('threshold of accuracy')
plt.ylabel('execution time (s)')
plt.title('CreditcardData')
plt.xticks(diff_acc)
#plt.yscale('log')
plt.legend()
plt.savefig("../../../OutputData/CreditcardDataset/tha_time.png")
plt.show()


fig, ax = plt.subplots()
plt.plot(diff_acc, num_calculations, label="new algorithm", color='blue', linewidth = 3.4)

plt.xlabel('threshold of accuracy')
plt.ylabel('number of cardinality calculations (K)')
plt.title('CreditcardData')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


plt.xticks(diff_acc)
plt.legend()
plt.savefig("../../../OutputData/CreditcardDataset/tha_calculations.png")
plt.show()



plt.close()
plt.clf()

