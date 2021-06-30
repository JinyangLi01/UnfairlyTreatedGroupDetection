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
from Algorithms import NewAlg_1_20210529 as newalg
from Algorithms import NaiveAlg_1_20210528 as naivealg
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


def thousands_formatter(x, pos):
    return int(x/1000)



selected_attributes = ['limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_0', 'pay_2',
                       'pay_3', 'pay_4', 'pay_5']


diff_acc = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3 ]
original_data_file = "../../../../InputData/CreditcardDataset/ForClassification/credit_card_clients_categorized_testdata.csv"
original_data = pd.read_csv(original_data_file)
mis_data_file = "../../../../InputData/CreditcardDataset/ForClassification/credit_card_clients_cat_mis.csv"
mis_data = pd.read_csv(mis_data_file)


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
thc = 50
num_loops = 1




less_attribute_data = original_data[selected_attributes]
mis_class_data = mis_data[selected_attributes]
overall_acc = 1 - len(mis_class_data) / len(less_attribute_data)

print("overall_acc = {}".format(overall_acc))


for dif in diff_acc:
    print("\n\ndif = {}".format(dif))
    tha = overall_acc - dif
    t, calculations = 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        print("tha = {}, thc = {}".format(tha, thc))
        pattern_with_low_accuracy, num_calculation, t_ = newalg.GraphTraverse(less_attribute_data,
                                                                              mis_class_data, tha,
                                                                              thc, time_limit)
        print("time = {} s, num_calculation = {}".format(t_, num_calculation), "\n", pattern_with_low_accuracy)
        t += t_
        calculations += num_calculation
        if l == 0:
            result_cardinality = len(pattern_with_low_accuracy)
            patterns_found.append(pattern_with_low_accuracy)
            num_patterns_found.append(result_cardinality)
    t /= num_loops
    calculations /= num_loops
    execution_time.append(t)
    num_calculations.append(calculations)





output_path = r'../../../../OutputData/LowAccDetection/CreditcardDataset/tha.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time)

output_file.write("overall accuracy: {}\n".format(overall_acc))


output_file.write("execution time\n")
for n in range(len(diff_acc)):
    output_file.write('{} {}\n'.format(diff_acc[n], execution_time[n]))


output_file.write("\n\nnumber of calculations\n")
for n in range(len(diff_acc)):
    output_file.write('{} {}\n'.format(diff_acc[n], num_calculations[n]))


output_file.write("\n\nnumber of patterns found\n")
for n in range(len(diff_acc)):
    output_file.write('{} {} \n {}\n'.format(diff_acc[n], num_patterns_found[n], patterns_found[n]))





plt.plot(diff_acc, execution_time, label="optimized algorithm", color='blue', linewidth = 3.4)


plt.xlabel('delta fairness value')
plt.ylabel('execution time (s)')
plt.xticks(diff_acc)

plt.subplots_adjust(bottom=0.15, left=0.18)
plt.legend()
plt.savefig("../../../../OutputData/LowAccDetection/CreditcardDataset/tha_time.png")
plt.show()


fig, ax = plt.subplots()
plt.plot(diff_acc, num_calculations, label="optimized algorithm", color='blue', linewidth = 3.4)

plt.xlabel('delta fairness value')
plt.ylabel('number of patterns visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


plt.xticks(diff_acc)
plt.subplots_adjust(bottom=0.15, left=0.18)
plt.legend()
plt.savefig("../../../../OutputData/LowAccDetection/CreditcardDataset/tha_calculations.png")
plt.show()



plt.close()
plt.clf()


