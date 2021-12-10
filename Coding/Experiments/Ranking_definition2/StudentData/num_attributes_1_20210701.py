



import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlgRanking_definition2_8_20211228 as newalg
from Algorithms import NaiveAlgRanking_definition2_3_20211207 as naivealg
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

label = ["Optimized", "Naive"]
line_width = 8
marker_size = 15
# f_size = (14, 10)

f_size = (14, 12)


def plot_runtime(input_file):
   fig, ax = plt.subplots(1, 1, figsize=f_size)
   delim = '\t'
   with open(input_file) as f:
      lines = [line.rstrip('\n') for line in f]

   bound = []
   naive = []
   opt = []

   for line in lines[1:]:
      bound.append(float(line.split(delim)[0].strip()))
      naive_time = float(line.split(delim)[2].strip())
      if naive_time > 0:
         naive.append(naive_time)
      opt.append(float(line.split(delim)[3].strip()))

   plt.plot(bound[0:len(naive)], naive, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
   plt.plot(bound, opt, line_style[1], color=color[1], label=label[1], linewidth=line_width,
             markersize=marker_size)

   plt.xlabel('Bound')
   plt.ylabel('Time [sec]')
   plt.legend(loc='best')
   plt.grid(True)

   fig = plt.gcf()
   plt.savefig(input_file + '_plot.pdf', bbox_inches='tight')
   plt.close()



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

def GridSearch(original_data, all_attributes, thc, alpha, number_attributes, time_limit, only_new_alg=False):

    selected_attributes = all_attributes[:number_attributes]
    print("{} attributes: {}".format(number_attributes, selected_attributes))

    less_attribute_data = original_data[selected_attributes]


    if only_new_alg:
        pattern_treated_unfairly1, num_patterns_visited1_, t1_ \
            = newalg.GraphTraverse(
            less_attribute_data, selected_attributes, thc,
            alpha,
            k_min, k_max, time_limit)

        print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
        print(
            "time = {} s".format(t1_), "\n",
            "patterns:\n",
            pattern_treated_unfairly1)

        return t1_, num_patterns_visited1_, 0, 0, pattern_treated_unfairly1

    pattern_treated_unfairly1, num_patterns_visited1_, t1_ \
        = newalg.GraphTraverse(
        less_attribute_data, selected_attributes, thc,
        alpha,
        k_min, k_max, time_limit)

    print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
    print(
        "time = {} s".format(t1_), "\n",
            "patterns:\n",
            pattern_treated_unfairly1)


    pattern_treated_unfairly2, \
    num_patterns_visited2_, t2_ = naivealg.NaiveAlg(less_attribute_data, selected_attributes, thc,
                                                    alpha,
                                                    k_min, k_max, time_limit)

    print("num_patterns_visited = {}".format(num_patterns_visited2_))
    print(
        "time = {} s".format(t2_), "\n",
        "patterns:\n",
        pattern_treated_unfairly2)

    for k in range(k_min, k_max):
        if ComparePatternSets(pattern_treated_unfairly1[k-k_min], pattern_treated_unfairly2[k-k_min]) is False:
            raise Exception("k={}, sanity check fails!".format(k))


    if t1_ > time_limit:
        print("new alg exceeds time limit")
    if t2_ > time_limit:
        print("naive alg exceeds time limit")

    return t1_, num_patterns_visited1_, t2_, num_patterns_visited2_, \
           pattern_treated_unfairly1


all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
                  'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C',
                  'failures_C', 'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C', 'higher_C',
                  'internet_C', 'romantic_C', 'famrel_C', 'freetime_C', 'goout_C', 'Dalc_C', 'Walc_C',
                  'health_C', 'absences_C', 'G1_C', 'G2_C', 'G3_C']

thc = 50

original_data_file = r"../../../../InputData/StudentDataset/ForRanking_1/student-mat_cat_ranked.csv"

original_data = pd.read_csv(original_data_file)[all_attributes]



time_limit = 5*60


# with 23 att, naive needs 517s
num_att_max_naive = 25 # if it's ??, naive out of time
num_att_min = 24
num_att_max = 25
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



k_min = 10
k_max = 50
List_k = list(range(k_min, k_max))
alpha = 0.1

for number_attributes in range(num_att_min, num_att_max_naive):
    print("\n\nnumber of attributes = {}".format(number_attributes))

    t1, t2, calculation1, calculation2 = 0, 0, 0, 0
    result_cardinality = 0
    for l in range(num_loops):
        t1_, calculation1_,  t2_, calculation2_, pattern_treated_unfairly = \
            GridSearch(original_data, all_attributes, thc, alpha, number_attributes, time_limit)
        t1 += t1_
        t2 += t2_
        calculation1 += calculation1_
        calculation2 += calculation2_
        if l == 0:
            patterns_found.append(pattern_treated_unfairly)
            num_patterns_found.append(len(pattern_treated_unfairly))
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
        t1_, calculation1_, t2_, calculation2_, pattern_treated_unfairly = \
            GridSearch(original_data, all_attributes, thc, alpha, number_attributes, time_limit, only_new_alg=True)
        t1 += t1_
        calculation1 += calculation1_
        if l == 0:
            patterns_found.append(pattern_treated_unfairly)
            num_patterns_found.append(len(pattern_treated_unfairly))
    t1 /= num_loops
    calculation1 /= num_loops

    execution_time1.append(t1)
    num_calculation1.append(calculation1)




output_path = r'../../../../OutputData/Ranking_definition2/StudentData/num_att.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)


output_file.write("execution time\n")
for n in range(num_att_min, num_att_max_naive):
    output_file.write('{} {} {}\n'.format(n, execution_time1[n-num_att_min], execution_time2[n-num_att_min]))
for n in range(num_att_max_naive, num_att_max):
    output_file.write('{} {}\n'.format(n, execution_time1[n - num_att_max_naive]))


output_file.write("\n\nnumber of patterns checked\n")
for n in range(num_att_min, num_att_max_naive):
    output_file.write('{} {} {}\n'.format(n, num_calculation1[n-num_att_min], num_calculation2[n-num_att_min]))
for n in range(num_att_max_naive, num_att_max):
    output_file.write('{} {}\n'.format(n, num_calculation1[n-num_att_max_naive]))



output_file.write("\n\nnumber of patterns found\n")
for n in range(num_att_min, num_att_max_naive):
    output_file.write('{} {} \n {}\n'.format(n, num_patterns_found[n-num_att_min],
                                             patterns_found[n-num_att_min]))
for n in range(num_att_max_naive, num_att_max):
    output_file.write('{} {} \n {}\n'.format(n, num_patterns_found[n-num_att_min],
                                             patterns_found[n-num_att_min]))




# when number of attributes = 8, naive algorithm running time > 10min
# so we only use x[:6]
x_new = list(range(num_att_min, num_att_max))
x_naive = list(range(num_att_min, num_att_max))


fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(x_new, execution_time1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
plt.plot(x_naive, execution_time2, line_style[1], color=color[1], label=label[1], linewidth=line_width,
             markersize=marker_size)

plt.xlabel('Number of attributes')
plt.ylabel('Execution time (s)')
plt.xticks(x_new)
plt.legend()
plt.grid(True)
plt.savefig("../../../../OutputData/Ranking_definition2/StudentData/num_att_time.png", bbox_inches='tight')
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


plt.xticks(x_new)
plt.legend()
plt.grid(True)
plt.savefig("../../../../OutputData/Ranking_definition2/StudentData/num_att_calculations.png", bbox_inches='tight')
plt.show()
plt.close()


plt.clf()


