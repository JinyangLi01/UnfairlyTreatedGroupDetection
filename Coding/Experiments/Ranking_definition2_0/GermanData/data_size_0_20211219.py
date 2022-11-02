
import pandas as pd
from Algorithms.DevelopingHistory import NewAlgRanking_definition2_8_20211228 as newalg, \
    NaiveAlgRanking_definition2_3_20211207 as naivealg

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

label = ["UPR", "IterTD"]
line_width = 8
marker_size = 15
f_size = (14, 8)



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


def GridSearch(original_data_file_pathpre, datasize, thc, selected_attributes, alpha,
        k_min, k_max, time_limit):
    original_data_file = original_data_file_pathpre + str(datasize) + ".csv"
    ranked_data = pd.read_csv(original_data_file)[selected_attributes]
    # print(ranked_data[:4])
    pattern_treated_unfairly_lowerbound1, num_patterns_visited1_, t1_ \
        = newalg.GraphTraverse(
        ranked_data, selected_attributes, thc,
        alpha,
        k_min, k_max, time_limit)

    print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
    print("time = {} s, num of pattern_treated_unfairly_lowerbound = {} ".format(
            t1_, len(pattern_treated_unfairly_lowerbound1)))
    if t1_ > time_limit:
        raise Exception("new alg exceeds time limit")

    pattern_treated_unfairly_lowerbound2, \
    num_patterns_visited2_, t2_ = naivealg.NaiveAlg(ranked_data, selected_attributes, thc,
                                                    alpha,
                                                    k_min, k_max, time_limit)

    print("naive alg, num_patterns_visited = {}".format(num_patterns_visited2_))
    print("time = {} s, num of pattern_treated_unfairly_lowerbound = {}".format(
            t2_, len(pattern_treated_unfairly_lowerbound2)))

    if t2_ > time_limit:
        raise Exception("naive alg exceeds time limit")

    for k in range(0, k_max - k_min):
        if ComparePatternSets(pattern_treated_unfairly_lowerbound1[k], pattern_treated_unfairly_lowerbound2[k]) is False:
            raise Exception("sanity check fails! k = {}".format(k + k_min))



    return t1_, num_patterns_visited1_, t2_, num_patterns_visited2_, pattern_treated_unfairly_lowerbound2




all_attributes = ['StatusExistingAcc', 'DurationMonth_C', 'CreditHistory', 'Purpose', 'CreditAmount_C',
                  'SavingsAccount', 'EmploymentLength', 'InstallmentRate', 'MarriedNSex', 'Debtors',
                  'ResidenceLength', 'Property', 'Age_C', 'InstallmentPlans', 'Housing',
                  'ExistingCredit', 'Job', 'NumPeopleLiable', 'Telephone', 'ForeignWorker']


# 20 att, 2000 data, naive over time
selected_attributes = all_attributes[:15]

data_sizes = [1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]


Thc = 50

original_data_file_pathprefix = r"../../../../InputData/GermanCredit/LargerDataset/"

time_limit = 10*60


execution_time1 = list()
execution_time2 = list()
num_patterns_checked1 = list()
num_patterns_checked2 = list()
num_patterns_found_lowerbound = list()
patterns_found_lowerbound = list()
num_loops = 1
k_min = 10
k_max = 50


List_k = list(range(k_min, k_max))

alpha = 0.1

for datasize in data_sizes:
    num_patterns_visited1_datasize = 0
    num_patterns_visited2_datasize = 0
    print('\n\ndatasize = {}'.format(datasize))
    t1, t2 = 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, num_patterns_visited1_, t2_, num_patterns_visited2_, pattern_treated_unfairly_lowerbound = \
            GridSearch(original_data_file_pathprefix, datasize, Thc, selected_attributes, alpha,
        k_min, k_max, time_limit)
        t1 += t1_
        t2 += t2_
        num_patterns_visited1_datasize += num_patterns_visited1_
        num_patterns_visited2_datasize += num_patterns_visited2_
        if l == 0:
            patterns_found_lowerbound.append(pattern_treated_unfairly_lowerbound)
            num_patterns_found_lowerbound.append(len(pattern_treated_unfairly_lowerbound))

    t1 /= num_loops
    t2 /= num_loops

    execution_time1.append(t1)
    num_patterns_checked1.append(num_patterns_visited1_datasize)
    execution_time2.append(t2)
    num_patterns_checked2.append(num_patterns_visited2_datasize)




output_path = r'../../../../OutputData/Ranking_definition2_1/GermanData/data_size.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)



output_file.write("execution time\n")
for n in range(len(data_sizes)):
    output_file.write('{} {} {}\n'.format(data_sizes[n], execution_time1[n], execution_time2[n]))


output_file.write("\n\nnumber of patterns\n")
for n in range(len(data_sizes)):
    output_file.write('{} {} {}\n'.format(data_sizes[n], num_patterns_checked1[n], num_patterns_checked2[n]))




fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(data_sizes, execution_time1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
plt.plot(data_sizes, execution_time2, line_style[1], color=color[1], label=label[1], linewidth=line_width,
             markersize=marker_size)
plt.xlabel('Data size')
plt.ylabel('Execution time (s)')
plt.legend(loc='best')
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/Ranking_definition2_1/GermanData/datasize_time_urb_german.png",
            bbox_inches='tight')
plt.show()
plt.close()





fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(data_sizes, num_patterns_checked1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
plt.plot(data_sizes, num_patterns_checked2, line_style[1], color=color[1], label=label[1], linewidth=line_width,
             markersize=marker_size)
plt.xlabel('Data size')
plt.ylabel('Number of patterns visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.legend(loc='best')
plt.grid(True)
fig.tight_layout()
plt.savefig("../../../../OutputData/Ranking_definition2_1/GermanData/datasize_calculations_urb_german.png",
            bbox_inches='tight')
plt.show()
plt.close()

plt.clf()

