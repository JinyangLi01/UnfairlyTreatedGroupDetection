import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlg_2_20211001 as newalg
from Algorithms import NaiveAlg_2_20211020 as naivealg
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

# 13 att is ok.
# with 13 att, thc =10, new alg needs 227 s
# with 14 att, thc = 10, new alg needs 486 s
# with 15 att, thc = 10, new alg over time
selected_attributes = all_attributes[:9]

Thc_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

original_data_file = "../../../../InputData/MedicalDataset/train/train_41att.csv"
original_data = pd.read_csv(original_data_file)
mis_data_file = "../../../../InputData/MedicalDataset/train/train_mis_41att.csv"
mis_data = pd.read_csv(mis_data_file)

time_limit = 10 * 60
execution_time1 = list()

num_calculation1 = list()

num_pattern_skipped_mis_c1 = list()

num_pattern_skipped_whole_c1 = list()

num_patterns_found = list()
patterns_found = list()
num_loops = 1

less_attribute_data = original_data[selected_attributes]
mis_class_data = mis_data[selected_attributes]
overall_acc = 1 - len(mis_class_data) / len(less_attribute_data)
print("overall_acc = {}".format(overall_acc))

for thc in Thc_list:
    print("\nthc = {}".format(thc))
    tha = overall_acc - 0.2
    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        print("tha = {}, thc = {}".format(tha, thc))

        pattern_with_low_accuracy1, sizes_of_patterns, fairness_values_of_patterns, calculation1_, t1_ \
            = newalg.GraphTraverse(less_attribute_data, mis_class_data, tha, thc, time_limit)

        print("time = {} s, num_calculation = {}".format(t1_, calculation1_))
        print("find {} patterns".format(len(pattern_with_low_accuracy1)))
        if t1_ > time_limit:
            raise Exception("thc = {}, new alg over time".format(thc))
        t1 += t1_
        calculation1 += calculation1_

        if l == 0:
            result_cardinality = len(pattern_with_low_accuracy1)
            patterns_found.append(pattern_with_low_accuracy1)
            num_patterns_found.append(result_cardinality)

    t1 /= num_loops
    t2 /= num_loops
    calculation1 /= num_loops
    calculation2 /= num_loops

    execution_time1.append(t1)
    num_calculation1.append(calculation1)

output_path = r'../../../../OutputData/LowAccDetection_2/MedicalDataset/thc.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("overall accuracy: {}\n".format(overall_acc))

output_file.write("execution time\n")
for n in range(len(Thc_list)):
    output_file.write('{} {}\n'.format(Thc_list[n], execution_time1[n]))

output_file.write("\n\nnumber of calculations\n")
for n in range(len(Thc_list)):
    output_file.write('{} {}\n'.format(Thc_list[n], num_calculation1[n]))

output_file.write("\n\nnumber of patterns found\n")
for n in range(len(Thc_list)):
    output_file.write('{} {} \n {}\n'.format(Thc_list[n], num_patterns_found[n], patterns_found[n]))

fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(Thc_list, execution_time1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
         markersize=marker_size)
plt.xlabel('Size threshold')
plt.ylabel('Execution time (s)')
plt.xticks(Thc_list)
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/LowAccDetection_2/MedicalDataset/thc_time.png",
            bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(Thc_list, num_calculation1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
         markersize=marker_size)
plt.xlabel('Size threshold')
plt.ylabel('Number of patterns visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.xticks(Thc_list)
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/LowAccDetection_2/MedicalDataset/thc_calculations.png",
            bbox_inches='tight')
plt.show()
plt.close()

plt.clf()
