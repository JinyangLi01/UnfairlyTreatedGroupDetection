


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

def GridSearch(original_data, TP, TN, FP, FN, all_attributes, thc, number_attributes, time_limit, only_new_alg=False):

    selected_attributes = all_attributes[:number_attributes]
    print("{} attributes: {}".format(number_attributes, selected_attributes))

    less_attribute_data = original_data[selected_attributes]
    FP = FP[selected_attributes]
    TN = TN[selected_attributes]
    FN = FN[selected_attributes]
    TP = TP[selected_attributes]

    original_thf_FPR = len(FP) / (len(FP) + len(TN))

    delta_thf = 0.2
    fairness_definition = 1

    print("original_thf_FPR = {}, delta_thf = {}, fairness_definition = {}".format(original_thf_FPR, delta_thf, fairness_definition))


    if only_new_alg:

        pattern_with_low_fairness1, num_calculation1, execution_time1 = newalg.GraphTraverse(less_attribute_data,
                                                                                TP, TN, FP, FN, delta_thf,
                                                                                thc, time_limit, fairness_definition)
        print("newalg, time = {} s, num_calculation = {}".format(execution_time1, num_calculation1))

        print("{} patterns with low accuracy: \n {}".format(len(pattern_with_low_fairness1), pattern_with_low_fairness1))
        if execution_time1 > time_limit:
            raise Exception("new alg over time")
        return execution_time1, num_calculation1, 0, 0, pattern_with_low_fairness1



    pattern_with_low_fairness1, num_calculation1, execution_time1 = newalg.GraphTraverse(less_attribute_data,
                                                                                         TP, TN, FP, FN, delta_thf,
                                                                                         thc, time_limit,
                                                                                         fairness_definition)

    if execution_time1 > time_limit:
        raise Exception("new alg exceeds time limit")
    print("newalg, time = {} s, num_calculation = {}".format(execution_time1, num_calculation1), "\n", pattern_with_low_fairness1)

    pattern_with_low_fairness2, num_calculation2, execution_time2 = naivealg.NaiveAlg(less_attribute_data,
                                                                     TP, TN, FP, FN, delta_thf,
                                                                     thc, time_limit, fairness_definition)
    print("naivealg, time = {} s, num_calculation = {}".format(execution_time2, num_calculation2), "\n",
          pattern_with_low_fairness2)


    if ComparePatternSets(pattern_with_low_fairness1, pattern_with_low_fairness2) is False:
        raise Exception("sanity check fails!")


    print("{} patterns with low accuracy: \n {}".format(len(pattern_with_low_fairness1), pattern_with_low_fairness1))

    if execution_time2 > time_limit:
        raise Exception("naive alg exceeds time limit")

    return execution_time1, num_calculation1, execution_time2, num_calculation2, \
           pattern_with_low_fairness1

all_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass', 'relationship',
                  'educational-num', 'occupation', 'capital-gain','capital-loss','hours-per-week', 'native-country']




thc = 50

original_data_file = "../../../../InputData/AdultDataset/ForClassification/CleanAdult_numerical_testdata_cat.csv"
original_data = pd.read_csv(original_data_file)
FP_data_file = "../../../../InputData/AdultDataset/ForClassification/CleanAdult_numerical_FP_cat.csv"
FP = pd.read_csv(FP_data_file)
TP_data_file = "../../../../InputData/AdultDataset/ForClassification/CleanAdult_numerical_TP_cat.csv"
TP = pd.read_csv(TP_data_file)
FN_data_file = "../../../../InputData/AdultDataset/ForClassification/CleanAdult_numerical_FN_cat.csv"
FN = pd.read_csv(FN_data_file)
TN_data_file = "../../../../InputData/AdultDataset/ForClassification/CleanAdult_numerical_TN_cat.csv"
TN = pd.read_csv(TN_data_file)

overall_FPR = len(FP) / (len(FP) + len(TN))

time_limit = 10*60



# so for naive alg, we only do when number of attributes <= 7
# when there are 6 att, naive alg runs faster than new alg with 12 att
# with 7 att, naive alg needs 120 s
# with 8 att, naive alg over time longer than 10min
# with 12 att, new alg needs 290 s
# with 13 att, new alg overtime > 10min
num_att_max_naive = 8
num_att_min = 3
num_att_max = 13 # if it's 14, new alg over time > 10min
execution_time1 = list()
execution_time2 = list()
num_calculation1 = list()
num_calculation2 = list()
num_pattern_skipped_mis_c1 = list()
num_pattern_skipped_mis_c2 = list()
num_pattern_skipped_whole_c1 = list()
num_pattern_skipped_whole_c2 = list()
num_patterns_found = list()
patterns_found = list()
num_loops = 1



for number_attributes in range(num_att_min, num_att_max_naive):
    print("\n\nnumber of attributes = {}".format(number_attributes))
    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, calculation1_,  t2_, calculation2_, result = \
            GridSearch(original_data, TP, TN, FP, FN, all_attributes, thc, number_attributes, time_limit)
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
    num_calculation1.append(calculation1)
    execution_time2.append(t2)
    num_calculation2.append(calculation2)



for number_attributes in range(num_att_max_naive, num_att_max):
    print("\n\nnumber of attributes = {}".format(number_attributes))
    t1, calculation1 = 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, calculation1_,  _, _, result = \
            GridSearch(original_data, TP, TN, FP, FN, all_attributes, thc, number_attributes, time_limit, True)
        t1 += t1_
        calculation1 += calculation1_
        if l == 0:
            result_cardinality = len(result)
            patterns_found.append(result)
            num_patterns_found.append(result_cardinality)
    t1 /= num_loops
    calculation1 /= num_loops

    execution_time1.append(t1)
    num_calculation1.append(calculation1)




output_path = r'../../../../OutputData/General_1_withStopCond/AdultDataset/num_attribute.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("overall FPR: {}\n".format(overall_FPR))

output_file.write("execution time\n")
for n in range(num_att_min, num_att_max_naive):
    output_file.write('{} {} {}\n'.format(n, execution_time1[n-num_att_min], execution_time2[n-num_att_min]))
for n in range(num_att_max_naive, num_att_max):
    output_file.write('{} {}\n'.format(n, execution_time1[n - num_att_min]))


output_file.write("\n\nnumber of patterns checked\n")
for n in range(num_att_min, num_att_max_naive):
    output_file.write('{} {} {}\n'.format(n, num_calculation1[n-num_att_min], num_calculation2[n-num_att_min]))
for n in range(num_att_max_naive, num_att_max):
    output_file.write('{} {}\n'.format(n, num_calculation1[n-num_att_min]))



output_file.write("\n\nnumber of patterns found\n")
for n in range(num_att_min, num_att_max):
    output_file.write('{} {} \n {}\n'.format(n, num_patterns_found[n-num_att_min], patterns_found[n-num_att_min]))




# when number of attributes = 8, naive algorithm running time > 10min
# so we only use x[:6]
x_new = list(range(num_att_min, num_att_max))
x_naive = list(range(num_att_min, num_att_max_naive))





fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(x_new, execution_time1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
plt.plot(x_naive, execution_time2 , line_style[1], color=color[1], label=label[1], linewidth=line_width,
          markersize=marker_size)
plt.xlabel('Number of attributes')
plt.ylabel('Execution time (s)')
plt.xticks([2, 4, 6, 8, 10, 12])
plt.legend(loc='best')
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/General_withStopCond/AdultDataset/num_att_time.png", bbox_inches='tight')
plt.show()
plt.close()





fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(x_new, num_calculation1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
plt.plot(x_naive, num_calculation2, line_style[1], color=color[1], label=label[1], linewidth=line_width,
          markersize=marker_size)
plt.xlabel('Number of attributes')
plt.ylabel('Number of patterns visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.xticks([2, 4, 6, 8, 10, 12])
plt.legend(loc='best')
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/General_withStopCond/AdultDataset/num_att_calculations.png", bbox_inches='tight')
plt.show()
plt.close()


plt.clf()

