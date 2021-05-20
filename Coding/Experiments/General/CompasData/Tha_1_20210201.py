"""
This script is to do experiment on the threshold of minority group accuracy.

two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: diff_acc = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
threshold of minority group accuracy: overall acc - diff_acc

Other parameters:
original_data_file = "../../../InputData/CompasData/RecidivismData_att_classified.csv"
selected_attributes = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree',
                       'decile_score', 'c_days_from_compas_C', 'juv_fel_count_C', 'juv_misd_count_C',
                       'juv_other_count_C']
Thc = 30

"""



import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlg_0_20201128 as newalg
from Algorithms import Predict_0_20210127 as predict
from Algorithms import NaiveAlg_0_20201111 as naivealg
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def thousands_formatter(x, pos):
    return int(x/1000)



# all att:
selected_attributes = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree',
                       'decile_score', 'c_days_from_compas_C', 'juv_fel_count_C', 'juv_misd_count_C',
                       'juv_other_count_C']

diff_acc = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
original_data_file = "../../../../InputData/CompasData/RecidivismData_att_classified.csv"
att_to_predict = 'is_recid'
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
thc = 10
num_loops = 5


less_attribute_data, mis_class_data, overall_acc = predict.PredictWithML(original_data_file,
                                                                         selected_attributes,
                                                                         att_to_predict)





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





output_path = r'../../../../OutputData/CompasDataset/tha.txt'
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
plt.title('CompasDataset')
plt.xticks(diff_acc)
#plt.yscale('log')
plt.legend()
plt.savefig("../../../OutputData/CompasDataset/tha_time.png")
plt.show()


fig, ax = plt.subplots()
plt.plot(diff_acc, num_calculations, label="new algorithm", color='blue', linewidth = 3.4)

plt.xlabel('threshold of accuracy')
plt.ylabel('number of cardinality calculations (K)')
plt.title('CompasDataset')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


plt.xticks(diff_acc)
plt.legend()
plt.savefig("../../../OutputData/CompasDataset/tha_calculations.png")
plt.show()



plt.close()
plt.clf()


