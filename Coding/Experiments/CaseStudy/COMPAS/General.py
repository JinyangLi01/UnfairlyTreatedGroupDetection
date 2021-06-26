import pandas as pd

from itertools import combinations
from Algorithms import pattern_count
import time
from Algorithms import NewAlgGeneral_1_20210528 as newalg
from Algorithms import NaiveAlgGeneral_1_202105258 as naivealg
from Algorithms import Predict_0_20210127 as predict



# all_attributes = ["age_binary", "age_bucketized", "sex_binary", "race_C", "MarriageStatus_C", "juv_fel_count_C",
#                   "decile_score_C",
#                     "juv_misd_count_C", "juv_other_count_C", "priors_count_C", "days_b_screening_arrest_C",
#                   "c_days_from_compas_C", "c_charge_degree_C", "v_decile_score_C", "start_C", "end_C",
#                   "event_C"]

# selected_attributes = ["age_binary", "sex_binary", "race_C", "MarriageStatus_C",
#                         "juv_fel_count_C",
#                         "juv_misd_count_C", "juv_other_count_C", "priors_count_C"
#                        ]

selected_attributes = ["age_binary", "sex_binary", "race_C", "MarriageStatus_C"]


original_data_file = r"../../../../InputData/CompasData/general/compas_data_cat_necessary_att_ranked.csv"
att_to_predict = 'is_recid'
ranked_data = pd.read_csv(original_data_file)
ranked_data = ranked_data[selected_attributes]



output_path = r'../../../../OutputData/CaseStudy/COMPAS/general/4att_2.txt'
output_file = open(output_path, "w")

output_file.write("selected_attributes: {}\n".format(selected_attributes))
output_file.write("att_to_predict: {}\n".format(att_to_predict))




less_attribute_data, TP, TN, FP, FN = predict.PredictWithMLReturnTPTNFPFN(original_data_file,
                                                                         selected_attributes,
                                                                         att_to_predict)


thc = 5
time_limit = 5 * 60
fairness_definition = 3
delta_thf = 0.1, 0.1



output_file.write("fairness_definition = {}, thc = {}, delta_thf = {}\n".format(fairness_definition, thc, delta_thf))


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