"""
This script is to do experiment on the threshold of minority group sizes.

two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: Thc, from 1 to 1000

Other parameters:
CleanAdult2.csv
selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass', 'relationship']
threshold of minority group accuracy: overall acc - 20


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


all_attributes = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree', 'decile_score']


Thc_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

original_data_file = "../../../../InputData/CompasData/Preprocessed_classified/RecidivismData_17att_classified.csv"
less_attribute_data = pd.read_csv(original_data_file)[all_attributes]
FP_data_file = "../../../../InputData/CompasData/Preprocessed_classified/RecidivismData_17att_classified_FP.csv"
FP = pd.read_csv(FP_data_file)[all_attributes]
TP_data_file = "../../../../InputData/CompasData/Preprocessed_classified/RecidivismData_17att_classified_TP.csv"
TP = pd.read_csv(TP_data_file)[all_attributes]
FN_data_file = "../../../../InputData/CompasData/Preprocessed_classified/RecidivismData_17att_classified_FN.csv"
FN = pd.read_csv(FN_data_file)[all_attributes]
TN_data_file = "../../../../InputData/CompasData/Preprocessed_classified/RecidivismData_17att_classified_TN.csv"
TN = pd.read_csv(TN_data_file)[all_attributes]

overall_FPR = len(FP) / (len(FP) + len(TN))
time_limit = 10*60
execution_time1 = list()

num_calculation1 = list()

num_pattern_skipped_mis_c1 = list()

num_pattern_skipped_whole_c1 = list()

num_patterns_found = list()
patterns_found = list()
num_loops = 1

fairness_definition = 1
delta_thf = 0.2




for thc in Thc_list:
    print("\nthc = {}".format(thc))
    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):

        pattern_with_low_fairness2, num_calculation2, execution_time2 = naivealg.NaiveAlg(less_attribute_data,
                                                                                          TP, TN, FP, FN, delta_thf,
                                                                                          thc, time_limit,
                                                                                          fairness_definition)
        print("naivealg, time = {} s, num_calculation = {}".format(execution_time2, num_calculation2), "\n",
              pattern_with_low_fairness2)

        t1 += execution_time2
        calculation1 += num_calculation2


        if execution_time2 > time_limit:
            print("new alg exceeds time limit")

        if l == 0:
            result_cardinality = len(pattern_with_low_fairness2)
            patterns_found.append(pattern_with_low_fairness2)
            num_patterns_found.append(result_cardinality)

    t1 /= num_loops
    calculation1 /= num_loops

    execution_time1.append(t1)
    num_calculation1.append(calculation1)






output_path = r'../../../../OutputData/General_2/CompasDataset/thcNaiveTime.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("execution time\n")
for n in range(len(Thc_list)):
    output_file.write('{} {}\n'.format(Thc_list[n], execution_time1[n]))


output_file.write("\n\nnumber of calculations\n")
for n in range(len(Thc_list)):
    output_file.write('{} {}\n'.format(Thc_list[n], num_calculation1[n]))


output_file.write("\n\nnumber of patterns found\n")
for n in range(len(Thc_list)):
    output_file.write('{} {} \n {}\n'.format(Thc_list[n], num_patterns_found[n], patterns_found[n]))




#
#
# fig, ax = plt.subplots(1, 1, figsize=f_size)
# plt.plot(Thc_list, execution_time1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
#          markersize=marker_size)
# plt.xlabel('Size threshold')
# plt.ylabel('Execution time (s)')
# plt.xticks(Thc_list)
# plt.grid(True)
# fig.tight_layout()
# plt.savefig("../../../../OutputData/General_2/CompasDataset/thc_time.png",
#             bbox_inches='tight')
# plt.show()
# plt.close()
#
#
#
#
# fig, ax = plt.subplots(1, 1, figsize=f_size)
# plt.plot(Thc_list, num_calculation1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
#          markersize=marker_size)
# plt.xlabel('Size threshold')
# plt.ylabel('Number of patterns visited (K)')
# ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
# plt.xticks(Thc_list)
# plt.grid(True)
# fig.tight_layout()
# plt.savefig("../../../../OutputData/General_2/CompasDataset/thc_calculations.png",
#             bbox_inches='tight')
# plt.show()
# plt.close()
#
#
#
# plt.clf()
#
