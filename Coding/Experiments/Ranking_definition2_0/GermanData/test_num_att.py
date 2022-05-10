import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlgRanking_definition2_13_20220509 as newalg
from Algorithms import NaiveAlgRanking_definition2_5_20220506 as naivealg
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter



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
        if t1_ > time_limit:
            raise Exception("new alg exceeds time limit")
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
    if t1_ > time_limit:
        raise Exception("new alg exceeds time limit")

    pattern_treated_unfairly2, \
    num_patterns_visited2_, t2_ = naivealg.NaiveAlg(less_attribute_data, selected_attributes, thc,
                                                    alpha,
                                                    k_min, k_max, time_limit)

    print("num_patterns_visited = {}".format(num_patterns_visited2_))
    print(
        "time = {} s".format(t2_), "\n",
        "patterns:\n",
        pattern_treated_unfairly2)


    if t2_ > time_limit:
        raise Exception("naive alg exceeds time limit")

    for k in range(k_min, k_max):
        if ComparePatternSets(pattern_treated_unfairly1[k-k_min], pattern_treated_unfairly2[k-k_min]) is False:
            raise Exception("k={}, sanity check fails!".format(k))


    return t1_, num_patterns_visited1_, t2_, num_patterns_visited2_, \
           pattern_treated_unfairly1



all_attributes = ['StatusExistingAcc', 'DurationMonth_C', 'CreditHistory', 'Purpose', 'CreditAmount_C',
                  'SavingsAccount', 'EmploymentLength', 'InstallmentRate', 'MarriedNSex', 'Debtors',
                  'ResidenceLength', 'Property', 'Age_C', 'InstallmentPlans', 'Housing',
                  'ExistingCredit', 'Job', 'NumPeopleLiable', 'Telephone', 'ForeignWorker']


Thc_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
k_min = 10
k_max = 50
original_data_file = r"../../../../InputData/GermanCredit/GermanCredit_ranked.csv"

original_data = pd.read_csv(original_data_file)[all_attributes]


time_limit = 10 * 60

alpha = 0.1

thc = 50


number_attributes = 12

t1_, calculation1_, t2_, calculation2_, pattern_treated_unfairly = \
            GridSearch(original_data, all_attributes, thc, alpha, number_attributes, time_limit, only_new_alg=True)


