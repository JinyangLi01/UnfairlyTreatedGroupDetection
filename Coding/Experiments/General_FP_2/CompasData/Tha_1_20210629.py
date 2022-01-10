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
from Algorithms import NewAlgGeneral_1_20210528 as newalg
from Algorithms import NaiveAlgGeneral_2_20211020 as naivealg
from Algorithms import Predict_0_20210127 as predict

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


sns.set_palette("Paired")
# sns.set_palette("deep")
sns.set_context("poster", font_scale=2)
sns.set_style("whitegrid")
# sns.palplot(sns.color_palette("deep", 10))
# sns.palplot(sns.color_palette("Paired", 9))

line_style = ['o-', 's--', '^:', '-.p']
color = ['C0', 'C1', 'C2', 'C3', 'C4']
plt_title = ["BlueNile", "COMPAS", "Credit Card"]

label = ["DUC", "Naive"]
line_width = 8
marker_size = 15
# f_size = (14, 10)

f_size = (14, 10)
def thousands_formatter(x, pos):
    return int(x/1000)



# all_attributes = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree', 'decile_score',
#                 'c_days_from_compas_C',
#                 'juv_fel_count_C', 'juv_misd_count_C', 'juv_other_count_C']


selected_attributes = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree', 'decile_score']

thc = 50

original_data_file = "../../../../InputData/CompasData/Preprocessed_classified/RecidivismData_13att_classified.csv"
less_attribute_data = pd.read_csv(original_data_file)[selected_attributes]
FP_data_file = "../../../../InputData/CompasData/Preprocessed_classified/RecidivismData_13att_classified_FP.csv"
FP = pd.read_csv(FP_data_file)[selected_attributes]
TP_data_file = "../../../../InputData/CompasData/Preprocessed_classified/RecidivismData_13att_classified_TP.csv"
TP = pd.read_csv(TP_data_file)[selected_attributes]
FN_data_file = "../../../../InputData/CompasData/Preprocessed_classified/RecidivismData_13att_classified_FN.csv"
FN = pd.read_csv(FN_data_file)[selected_attributes]
TN_data_file = "../../../../InputData/CompasData/Preprocessed_classified/RecidivismData_13att_classified_TN.csv"
TN = pd.read_csv(TN_data_file)[selected_attributes]

overall_FPR = len(FP) / (len(FP) + len(TN))

time_limit = 10*60
diff_acc = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


execution_time = list()
num_calculations = list()
num_patterns_checked2 = list()
num_pattern_skipped_mis_c1 = list()
num_pattern_skipped_mis_c2 = list()
num_pattern_skipped_whole_c1 = list()
num_pattern_skipped_whole_c2 = list()
num_patterns_found = list()
patterns_found = list()


num_loops = 1
fairness_definition = 1





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




output_path = r'../../../../OutputData/General_2/CompasDataset/tha.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time)

output_file.write("overall FPR: {}\n".format(overall_FPR))

output_file.write("execution time\n")
for n in range(len(diff_acc)):
    output_file.write('{} {}\n'.format(diff_acc[n], execution_time[n]))


output_file.write("\n\nnumber of calculations\n")
for n in range(len(diff_acc)):
    output_file.write('{} {}\n'.format(diff_acc[n], num_calculations[n]))


output_file.write("\n\nnumber of patterns found\n")
for n in range(len(diff_acc)):
    output_file.write('{} {} \n {}\n'.format(diff_acc[n], num_patterns_found[n], patterns_found[n]))






fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(diff_acc, execution_time, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
plt.xlabel('Delta fairness value')
plt.ylabel('Execution time (s)')
plt.xticks(diff_acc)
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/General_2/CompasDataset/tha_time.png", bbox_inches='tight')
plt.show()
plt.close()





fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(diff_acc, num_calculations, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
plt.xlabel('Delta fairness value')
plt.ylabel('Number of patterns visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.xticks(diff_acc)
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/General_2/CompasDataset/tha_calculations.png", bbox_inches='tight')
plt.show()
plt.close()


plt.clf()


