"""
This script is to do experiment on the data size.

CompasData_att_classified.txt: 6889 rows
data sizes: 100, 500, 1000, 2000, 3000, 4000, 5000, 6000
selected randomly, and generate files in InputData/CompasData/DifferentDataSizes/



two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: data sizes: 100, 500, 1000, 2000, 3000, 4000, 5000, 6000

Other parameters:
selected_attributes = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree']
size threshold Thc = 30
threshold of minority group accuracy: overall acc - 20

"""



import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlg_1_20210529 as newalg # when deterministic attributes are -1, use NewAlg_1, when -2, use NewALg_2
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

def GridSearch(original_data_file_pathpre, datasize, thc, selected_attributes):
    original_data_file = original_data_file_pathpre + str(datasize) + "_testdata.csv"
    mis_data_file = original_data_file_pathpre + str(datasize) + "_mis.csv"

    original_data = pd.read_csv(original_data_file)
    mis_data = pd.read_csv(mis_data_file)

    less_attribute_data = original_data[selected_attributes]
    mis_class_data = mis_data[selected_attributes]
    overall_acc = 1 - len(mis_class_data) / len(less_attribute_data)

    tha = overall_acc - 0.2
    print("tha = {}, thc = {}".format(tha, thc))
    pattern_with_low_accuracy1, num_calculation1, execution_time1 \
        = newalg.GraphTraverse(less_attribute_data, mis_class_data, tha, thc, time_limit)

    print("newalg, time = {} s, num_calculation = {}".format(execution_time1, num_calculation1), "\n",
          pattern_with_low_accuracy1)

    if execution_time1 > time_limit:
        raise Exception("new alg exceeds time limit")
    pattern_with_low_accuracy2, num_calculation2, execution_time2 = naivealg.NaiveAlg(less_attribute_data,
                                                                                      mis_class_data, tha,
                                                                                      thc, time_limit)
    print("naivealg, time = {} s, num_calculation = {}".format(execution_time2, num_calculation2), "\n",
          pattern_with_low_accuracy2)

    if ComparePatternSets(pattern_with_low_accuracy1, pattern_with_low_accuracy2) is False:
        raise Exception("sanity check fails!")

    print("{} patterns with low accuracy: \n {}".format(len(pattern_with_low_accuracy1), pattern_with_low_accuracy1))


    if execution_time2 > time_limit:
        raise Exception("naive alg exceeds time limit")


    return execution_time1, num_calculation1, execution_time2, num_calculation2, pattern_with_low_accuracy1, overall_acc, tha

all_attributes = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree', 'decile_score',
                'c_days_from_compas_C',
                'juv_fel_count_C', 'juv_misd_count_C', 'juv_other_count_C', 'start_C', 'end_C',
                  'v_decile_score_C', 'Violence_score_C', 'event_C']

# with 8 att, naive needs 445 s for 20K data
# with 9, over time
selected_attributes = all_attributes[:8]


data_sizes = [6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
Thc = 50
original_data_file_pathprefix = "../../../../InputData/CompasData/LargerDatasets/"



time_limit = 10*60
# based on experiments with the above parameters, when number of attributes = 8, naive algorithm running time > 10min
# so for naive alg, we only do when number of attributes <= 7
execution_time1 = list()
execution_time2 = list()
num_patterns_checked1 = list()
num_patterns_checked2 = list()
num_patterns_found = list()
patterns_found = list()
overall_acc_list = list()
tha_list = list()
num_loops = 1
overall_acc = 0
tha = 0

for datasize in data_sizes:
    print('\n\ndatasize = {}'.format(datasize))
    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, calculation1_, t2_, calculation2_, result, overall_acc, tha  = \
            GridSearch(original_data_file_pathprefix, datasize, Thc, selected_attributes)
        t1 += t1_
        t2 += t2_
        calculation1 += calculation1_
        calculation2 += calculation2_
        if l == 0:
            result_cardinality = len(result)
            patterns_found.append(result)
            num_patterns_found.append(result_cardinality)
    t1 /= num_loops
    t2 /= num_loops
    calculation1 /= num_loops
    calculation2 /= num_loops

    execution_time1.append(t1)
    num_patterns_checked1.append(calculation1)
    execution_time2.append(t2)
    num_patterns_checked2.append(calculation2)
    overall_acc_list.append(overall_acc)
    tha_list.append(tha)



output_path = r'../../../../OutputData/LowAccDetection_2/CompasDataset/data_size.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("execution time\n")
for n in range(len(data_sizes)):
    output_file.write('{} {} {}\n'.format(data_sizes[n], execution_time1[n], execution_time2[n]))


output_file.write("\n\nnumber of calculations\n")
for n in range(len(data_sizes)):
    output_file.write('{} {} {}\n'.format(data_sizes[n], num_patterns_checked1[n], num_patterns_checked2[n]))


output_file.write("\n\nnumber of patterns found\n")
for n in range(len(data_sizes)):
    output_file.write('{} {} \n {}\n'.format(data_sizes[n], num_patterns_found[n], patterns_found[n]))



fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(data_sizes, execution_time1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
plt.plot(data_sizes, execution_time2, line_style[1], color=color[1], label=label[1], linewidth=line_width,
             markersize=marker_size)
plt.xlabel('Data size (K)')
plt.xticks([6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000])
ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.ylabel('Execution time (s)')
plt.legend(loc='best')
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/LowAccDetection_2/CompasDataset/datasize_time.png",
            bbox_inches='tight')
plt.show()
plt.close()




fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(data_sizes, num_patterns_checked1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
plt.plot(data_sizes, num_patterns_checked2, line_style[1], color=color[1], label=label[1], linewidth=line_width,
             markersize=marker_size)
plt.xlabel('Data size (K)')
plt.xticks([6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000])
ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.ylabel('Number of patterns visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.legend(loc='best')
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/LowAccDetection_2/CompasDataset/datasize_calculations.png",
            bbox_inches='tight')
plt.show()
plt.close()



plt.clf()

