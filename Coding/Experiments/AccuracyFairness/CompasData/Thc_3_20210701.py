"""
This script is to do experiment on the threshold of minority group sizes.

two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: Thc, from 1 to 1000

Other parameters:
RecidivismData_att_classified.txt: 6889 rows
selected_attributes = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree']
threshold of minority group accuracy: overall acc - 20


"""



import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlg_1_20210529 as newalg
from Algorithms import NaiveAlg_1_20210528 as naivealg
from Algorithms import Predict_0_20210127 as predict
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




selected_attributes =['sexC', 'ageC','raceC', 'MC', 'priors_count_C', 'c_charge_degree', 'decile_score']

# all_attributes =['sexC', 'ageC','raceC', 'MC', 'priors_count_C', 'c_charge_degree', 'decile_score', 'c_days_from_compas_C',
#  'juv_fel_count_C', 'juv_misd_count_C', 'juv_other_count_C']

Thc_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
original_data_file = "../../../../InputData/CompasData/Preprocessed_classified/RecidivismData_13att_classified_testdata.csv"
original_data = pd.read_csv(original_data_file)
mis_data_file = "../../../../InputData/CompasData/Preprocessed_classified/RecidivismData_13att_classified_mis.csv"
mis_data = pd.read_csv(mis_data_file)


time_limit = 20*60
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
        pattern_with_low_accuracy1, calculation1_, t1_ = newalg.GraphTraverse(less_attribute_data,
                                                                              mis_class_data, tha,
                                                                              thc, time_limit)
        print("time = {} s, num_calculation = {}".format(t1_, calculation1_))
        print("find {} patterns".format(len(pattern_with_low_accuracy1)))
        t1 += t1_
        calculation1 += calculation1_



        if l == 0:
            result_cardinality = len(pattern_with_low_accuracy1)
            patterns_found.append(pattern_with_low_accuracy1)
            num_patterns_found.append(result_cardinality)

    t1 /= num_loops

    calculation1 /= num_loops


    execution_time1.append(t1)
    num_calculation1.append(calculation1)






output_path = r'../../../../OutputData/LowAccDetection/CompasDataset/thc_optimized.txt'
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




plt.plot(Thc_list, execution_time1, label="optimized algorithm", color='blue', linewidth = 3.4)

plt.xlabel('size threshold')
plt.ylabel('execution time (s)')
plt.xticks(Thc_list)
plt.subplots_adjust(bottom=0.15, left=0.18)
plt.legend()
plt.savefig("../../../../OutputData/LowAccDetection/CompasDataset/thc_time_optimized.png")
plt.show()


fig, ax = plt.subplots()
plt.plot(Thc_list, num_calculation1, label="optimized algorithm", color='blue', linewidth = 3.4)


plt.xlabel('size threshold')
plt.ylabel('number of patterns visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


plt.xticks(Thc_list)
plt.subplots_adjust(bottom=0.15, left=0.18)
plt.legend()
plt.savefig("../../../../OutputData/LowAccDetection/CompasDataset/thc_calculations_optimized.png")
plt.show()

plt.close()
plt.clf()

