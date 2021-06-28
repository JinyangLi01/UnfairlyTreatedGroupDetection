import pandas as pd

from itertools import combinations
from Algorithms import pattern_count
import time
from Algorithms import NewAlgGeneral_1_20210528 as newalg
from Algorithms import NaiveAlgGeneral_1_202105258 as naivealg
from Algorithms import Predict_0_20210127 as predict


"""
cox['sex'].replace(to_replace=['Male', 'Female'], value=[0, 1], inplace=True)


cox['age_cat'].replace(to_replace=['Less than 25', '25 - 45', 'Greater than 45'], value=[0, 1, 2], inplace=True)

cox['race'].replace(to_replace=['African-American', 'Asian', 'Caucasian', 'Hispanic', 'Native American', 'Other'], value=[0, 1, 2, 3, 4, 5], inplace=True)


"""

selected_attributes = ["sex", "age_cat", "race"]



original_data_file = r"../../../../../InputData/Compas_republica/compas-analysis-master/cox-parsed/cox-parsed_7214rows_cat.csv"
TP_data_file = r"../../../../../InputData/Compas_republica/compas-analysis-master/cox-parsed/cox-parsed-TP-cat.csv"
FP_data_file = r"../../../../../InputData/Compas_republica/compas-analysis-master/cox-parsed/cox-parsed-FP-cat.csv"
TN_data_file = r"../../../../../InputData/Compas_republica/compas-analysis-master/cox-parsed/cox-parsed-TN-cat.csv"
FN_data_file = r"../../../../../InputData/Compas_republica/compas-analysis-master/cox-parsed/cox-parsed-FN-cat.csv"



output_path = r'../../../../../OutputData/CaseStudy/ProPublica/fn_greater_than.txt'
output_file = open(output_path, "w")

output_file.write("selected_attributes: {}\n".format(selected_attributes))


def read_with_att(original_data_file, selected_attributes):
    original_data = pd.read_csv(original_data_file)
    less_attribute_data = original_data[selected_attributes]
    return less_attribute_data


less_attribute_data = read_with_att(original_data_file, selected_attributes)
TP = read_with_att(TP_data_file, selected_attributes)
FP = read_with_att(FP_data_file, selected_attributes)
TN = read_with_att(TN_data_file, selected_attributes)
FN = read_with_att(FN_data_file, selected_attributes)




thc = 20
time_limit = 5 * 60
# fairness_definition = 1 # FPR = FP/(FP+TN) False_positive_error_rate_balance

fairness_definition = 2 # FNR = FP/(FP+TN) False_positive_error_rate_balance, but for those treated too well
delta_thf = 0.1



output_file.write("fairness_definition = {}, thc = {}, delta_thf = {}\n".format(fairness_definition, thc, delta_thf))

print("less_attribute_data")
print(less_attribute_data)

pattern_with_low_fairness1, calculation1_, t1_ = newalg.GraphTraverse(less_attribute_data,
                                                                      TP, TN, FP, FN, delta_thf,
                                                                      thc, time_limit, fairness_definition)


print("newalg, time = {} s, num_calculation = {}\n".format(t1_, calculation1_))
print("num of patterns detected = {}".format(len(pattern_with_low_fairness1)))
for p in pattern_with_low_fairness1:
    print(p)

output_file.write("newalg, time = {} s, num_calculation = {}\n".format(t1_, calculation1_))
output_file.write("num of patterns detected = {}\n".format(len(pattern_with_low_fairness1)))
for p in pattern_with_low_fairness1:
    output_file.write(str(p))
    output_file.write("\n")

# pattern_with_low_accuracy2, calculation2_, t2_ = naivealg.NaiveAlg(less_attribute_data,
#                                                                    mis_class_data, tha,
#                                                                    thc, time_limit)
# print("naivealg, time = {} s, num_calculation = {}".format(t2_, calculation2_), "\n",
#       pattern_with_low_accuracy2)