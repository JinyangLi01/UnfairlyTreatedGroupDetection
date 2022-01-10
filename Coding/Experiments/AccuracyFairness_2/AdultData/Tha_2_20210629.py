
import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlg_2_20211001 as newalg
from Algorithms import Predict_0_20210127 as predict
from Algorithms import NaiveAlg_2_20211020 as naivealg

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


"""
all attributes:
predictors = ['age','workclass','education','educational-num',
              'marital-status', 'occupation','relationship','race','gender',
              'capital-gain','capital-loss','hours-per-week', 'native-country']

"""
selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass']

diff_acc = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

original_data_file = "../../../../InputData/AdultDataset/ForClassification/CleanAdult_numerical_testdata_cat.csv"
original_data = pd.read_csv(original_data_file)
mis_data_file = "../../../../InputData/AdultDataset/ForClassification/CleanAdult_numerical_mis_cat.csv"
mis_data = pd.read_csv(mis_data_file)

time_limit = 10 * 60
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

        pattern_with_low_accuracy, sizes_of_patterns, fairness_values_of_patterns, num_patterns_visited, t_ \
            = newalg.GraphTraverse(less_attribute_data, mis_class_data, tha, thc, time_limit)

        print("time = {} s, num_calculation = {}".format(t_, num_patterns_visited))
        print("find {} patterns".format(len(pattern_with_low_accuracy)))
        print(pattern_with_low_accuracy)
        t += t_
        calculations += num_patterns_visited
        if l == 0:
            result_cardinality = len(pattern_with_low_accuracy)
            patterns_found.append(pattern_with_low_accuracy)
            num_patterns_found.append(result_cardinality)
    t /= num_loops
    calculations /= num_loops
    execution_time.append(t)
    num_calculations.append(calculations)

output_path = r'../../../../OutputData/LowAccDetection_2/AdultDataset/tha.txt'
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

fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(diff_acc, execution_time, line_style[0], color=color[0], label=label[0], linewidth=line_width,
         markersize=marker_size)
plt.xlabel('Delta fairness value')
plt.ylabel('Execution time (s)')
plt.xticks(diff_acc)
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/LowAccDetection_2/AdultDataset/tha_time.png", bbox_inches='tight')
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
plt.savefig("../../../../OutputData/LowAccDetection_2/AdultDataset/tha_calculations.png", bbox_inches='tight')
plt.show()
plt.close()

plt.clf()
