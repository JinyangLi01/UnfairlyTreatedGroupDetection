

import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlgGeneral_2_20211219 as newalg
from Algorithms import NaiveAlgGeneral_3_20211219 as naivealg
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
    return int(x / 1000)


all_attributes = ['AGE_C', 'RACE', 'K6SUM42_C',
                  'REGION', 'SEX', 'MARRY', 'FTSTU', 'ACTDTY',
                  'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX',
                  'ANGIDX', 'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX',
                  'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX', 'JTPAIN',
                  'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT',
                  'WLKLIM', 'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42',
                  'DFSEE42', 'ADSMOK42', 'PHQ242', 'EMPST', 'POVCAT',
                  'INSCOV']

# with 14 att, dif = 0.05, new alg needs 340 s
# with 14 att, dif = 0.3, new alg over time
selected_attributes = all_attributes[:8]

diff_acc = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

original_data_file = "../../../../InputData/MedicalDataset/train/train_41att.csv"
original_data = pd.read_csv(original_data_file)
FP_data_file = "../../../../InputData/MedicalDataset/train/train_FP_41att.csv"
FP = pd.read_csv(FP_data_file)
TP_data_file = "../../../../InputData/MedicalDataset/train/train_TP_41att.csv"
TP = pd.read_csv(TP_data_file)
FN_data_file = "../../../../InputData/MedicalDataset/train/train_FN_41att.csv"
FN = pd.read_csv(FN_data_file)
TN_data_file = "../../../../InputData/MedicalDataset/train/train_TN_41att.csv"
TN = pd.read_csv(TN_data_file)

overall_FPR = len(FP) / (len(FP) + len(TN))

time_limit = 10*60

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
fairness_definition = 1

less_attribute_data = original_data[selected_attributes]
FP = FP[selected_attributes]
TN = TN[selected_attributes]
FN = FN[selected_attributes]
TP = TP[selected_attributes]


for dif in diff_acc:
    print("\n\ndif = {}, thc = {}".format(dif, thc))
    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):

        pattern_with_low_fairness1, calculation1_, t1_ = newalg.GraphTraverse(less_attribute_data,
                                                                              TP, TN, FP, FN, dif,
                                                                              thc, time_limit,
                                                                              fairness_definition)
        print("newalg, time = {} s, num_calculation = {}".format(t1_, calculation1_))

        print(
            "{} patterns with low accuracy: \n {}".format(len(pattern_with_low_fairness1), pattern_with_low_fairness1))
        if t1_ > time_limit:
            raise Exception("new alg over time")

        t1 += t1_
        calculation1 += calculation1_

        if l == 0:
            result_cardinality = len(pattern_with_low_fairness1)
            patterns_found.append(pattern_with_low_fairness1)
            num_patterns_found.append(result_cardinality)
    t1 /= num_loops
    t2 /= num_loops
    calculation1 /= num_loops
    calculation2 /= num_loops
    execution_time.append(t1)
    num_calculations.append(calculation1)


output_path = r'../../../../OutputData/General_2/MedicalDataset/tha.txt'
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
plt.savefig("../../../../OutputData/General_2/MedicalDataset/tha_time.png", bbox_inches='tight')
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
plt.savefig("../../../../OutputData/General_2/MedicalDataset/tha_calculations.png",
            bbox_inches='tight')
plt.show()
plt.close()

plt.clf()
