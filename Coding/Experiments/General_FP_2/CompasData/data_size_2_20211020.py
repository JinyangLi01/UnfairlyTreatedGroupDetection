

import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlgGeneral_1_20210528 as newalg
from Algorithms import NaiveAlgGeneral_2_20211020 as naivealg

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
    original_data_file = original_data_file_pathpre + str(datasize) + ".csv"
    less_attribute_data = pd.read_csv(original_data_file)[selected_attributes]
    FP_data_file = original_data_file_pathpre + str(datasize) + "_FP.csv"
    FP = pd.read_csv(FP_data_file)[selected_attributes]
    FN_data_file = original_data_file_pathpre + str(datasize) + "_FN.csv"
    FN = pd.read_csv(FN_data_file)[selected_attributes]
    TP_data_file = original_data_file_pathpre + str(datasize) + "_TP.csv"
    TP = pd.read_csv(TP_data_file)[selected_attributes]
    TN_data_file = original_data_file_pathpre + str(datasize) + "_TN.csv"
    TN = pd.read_csv(TN_data_file)[selected_attributes]

    original_thf_FPR = len(FP) / (len(FP) + len(TN))

    delta_thf = 0.2
    fairness_definition = 1


    pattern_with_low_fairness1, num_calculation1, execution_time1 = newalg.GraphTraverse(less_attribute_data,
                                                                                TP, TN, FP, FN, delta_thf,
                                                                                thc, time_limit, fairness_definition)
    if execution_time1 > time_limit:
        raise Exception("optimized alg exceeds time limit")
    print("newalg, time = {} s, num_calculation = {}".format(execution_time1, num_calculation1), "\n",
          pattern_with_low_fairness1)

    pattern_with_low_fairness2, num_calculation2, execution_time2 = naivealg.NaiveAlg(less_attribute_data,
                                                                     TP, TN, FP, FN, delta_thf,
                                                                     thc, time_limit, fairness_definition)

    print("naivealg, time = {} s, num_calculation = {}".format(execution_time2, num_calculation2), "\n",
          pattern_with_low_fairness2)


    if ComparePatternSets(pattern_with_low_fairness1, pattern_with_low_fairness2) is False:
        raise Exception("sanity check fails!")

    print("{} patterns with low accuracy: \n {}".format(len(pattern_with_low_fairness1), pattern_with_low_fairness2))



    if execution_time2 > time_limit:
        raise Exception("naive alg exceeds time limit")


    return execution_time1, num_calculation1, execution_time2, num_calculation2, pattern_with_low_fairness1

all_attributes = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree', 'decile_score',
                'c_days_from_compas_C',
                'juv_fel_count_C', 'juv_misd_count_C', 'juv_other_count_C', 'start_C', 'end_C',
                  'v_decile_score_C', 'Violence_score_C', 'event_C']


# with 6 att, ok
# with 8 att, over time
# with 7 att, ok
selected_attributes = all_attributes[:7] # ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree']

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
num_loops = 1


for datasize in data_sizes:
    print('\n\ndatasize = {}'.format(datasize))
    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, calculation1_, t2_, calculation2_, result = \
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




output_path = r'../../../../OutputData/General_2/CompasDataset/data_size.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("execution time\n")
for n in range(len(data_sizes)):
    output_file.write('{} {} {}\n'.format(data_sizes[n], execution_time1[n], execution_time2[n]))


output_file.write("\n\nnumber of calculations\n")
for n in range(len(data_sizes)):
    output_file.write('{} {} {}\n'.format(data_sizes[n], num_patterns_checked1[n], num_patterns_checked2[n]))



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
plt.savefig("../../../../OutputData/General_2/CompasDataset/datasize_time.png",
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
plt.savefig("../../../../OutputData/General_2/CompasDataset/datasize_calculations.png",
            bbox_inches='tight')
plt.show()
plt.close()



plt.clf()

