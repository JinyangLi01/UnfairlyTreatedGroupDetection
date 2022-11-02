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

from Algorithms.DevelopingHistory import NewAlgGeneral_1_20210528 as newalg, Predict_0_20210127 as predict
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
                       'pay_3', 'pay_4', 'pay_5', 'pay_6']

diff_acc = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
original_data_file = "../../../../InputData/CreditcardDataset/ForClassification/credit_card_clients_categorized.csv"

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
thc = 300
num_loops = 1
fairness_definition = 0


less_attribute_data, TP, TN, FP, FN = predict.PredictWithMLReturnTPTNFPFN(original_data_file,
                                                                         selected_attributes,
                                                                         att_to_predict)


for dif in diff_acc:
    print("\n\ndif = {}, thc = {}".format(dif, thc))
    t, calculations = 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        pattern_with_low_fairness1, calculation1_, t1_ = newalg.GraphTraverse(less_attribute_data,
                                                                              TP, TN, FP, FN, dif,
                                                                              thc, time_limit, fairness_definition)

        print("newalg, time = {} s, num_calculation = {}, num_pattern = {}".format(t1_, calculation1_,
              len(pattern_with_low_fairness1)),
              "\n",
              pattern_with_low_fairness1)
        t += t1_
        calculations += calculation1_

        if t1_ > time_limit:
            print("new alg exceeds time limit")

        if l == 0:
            result_cardinality = len(pattern_with_low_fairness1)
            patterns_found.append(pattern_with_low_fairness1)
            num_patterns_found.append(result_cardinality)
    t /= num_loops
    calculations /= num_loops
    execution_time.append(t)
    num_calculations.append(calculations)




output_path = r'../../../../OutputData/General_0/CreditcardDataset/tha.txt'
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





plt.plot(diff_acc, execution_time, label="optimized algorithm", color='blue', linewidth = 3.4)


plt.xlabel('delta fairness value')
plt.ylabel('execution time (s)')
plt.xticks(diff_acc)

plt.subplots_adjust(bottom=0.15, left=0.18)
plt.legend()
plt.savefig("../../../../OutputData/General/CreditcardDataset/tha_time.png")
plt.show()


fig, ax = plt.subplots()
plt.plot(diff_acc, num_calculations, label="optimized algorithm", color='blue', linewidth = 3.4)

plt.xlabel('delta fairness value')
plt.ylabel('number of patterns visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


plt.xticks(diff_acc)
plt.subplots_adjust(bottom=0.15, left=0.18)
plt.legend()
plt.savefig("../../../../OutputData/General/CreditcardDataset/tha_calculations.png")
plt.show()


plt.close()
plt.clf()


