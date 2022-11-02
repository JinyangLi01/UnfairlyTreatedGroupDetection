

import pandas as pd
from Algorithms import pattern_count
from Algorithms.DevelopingHistory import NewAlgRanking_19_20211216 as newalg

import matplotlib.pyplot as plt
import seaborn as sns

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



all_attributes = ['StatusExistingAcc', 'DurationMonth_C', 'CreditHistory', 'Purpose', 'CreditAmount_C',
                  'SavingsAccount', 'EmploymentLength', 'InstallmentRate', 'MarriedNSex', 'Debtors',
                  'ResidenceLength', 'Property', 'Age_C', 'InstallmentPlans', 'Housing',
                  'ExistingCredit', 'Job', 'NumPeopleLiable', 'Telephone', 'ForeignWorker']



selected_attributes = all_attributes[:20]


Thc = 600

k_min = 50
k_max = 51

Lowerbounds = [20] * 50

original_data_file = r"../../../../../InputData/GermanCredit/GermanCredit_ranked.csv"


ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]
time_limit = 10*60


List_k = list(range(k_min, k_max))



result1, num_patterns_visited1_, t1_ \
    = newalg.GraphTraverse(
    ranked_data, selected_attributes, Thc,
    Lowerbounds,
    k_min, k_max, time_limit)

print("newalg, num_patterns_visited = {}".format(num_patterns_visited1_))
print("time = {} s, num of pattern_treated_unfairly_lowerbound = {} ".format(
    t1_, len(result1)))

if t1_ > time_limit:
    raise Exception("new alg exceeds time limit")


pc_whole_data = pattern_count.PatternCounter(ranked_data, encoded=False)
pc_whole_data.parse_data()

whole_data_frame = ranked_data.describe(include='all')
patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k_min], encoded=False)
patterns_top_kmin.parse_data()


def num2string(pattern):
    st = ''
    for i in pattern:
        if i != -1:
            st += str(i)
        st += '|'
    st = st[:-1]
    return st



for k in range(0, k_max - k_min):
    print("{} patterns found".format(len(result1[k])))
    for r in result1[k]:
        st = num2string(r)
        whole_cardinality = pc_whole_data.pattern_count(st)
        num_top_k = patterns_top_kmin.pattern_count(st)
        print(", ".join(selected_attributes[i] + ": " + str(r[i]) for i in range(len(r)) if r[i] != -1))
        print(whole_cardinality, num_top_k)
    print("\n".join(str(v) for v in result1[k]))

