import pandas as pd

from Algorithms.DevelopingHistory import NewAlg_1_20210529 as newalg

# all_attributes = ["age_binary", "age_bucketized", "sex_binary", "race_C", "MarriageStatus_C", "juv_fel_count_C",
#                   "decile_score_C",
#                     "juv_misd_count_C", "juv_other_count_C", "priors_count_C", "days_b_screening_arrest_C",
#                   "c_days_from_compas_C", "c_charge_degree_C", "v_decile_score_C", "start_C", "end_C",
#                   "event_C"]

# selected_attributes = ["age_binary", "sex_binary", "race_C", "MarriageStatus_C",
#                         "juv_fel_count_C",
#                         "juv_misd_count_C", "juv_other_count_C", "priors_count_C"
#                        ]



"""
COMPAS:
sexC: [0, M] [1, F]
ageC: [] [] [3, 69] [] 
raceC: [0, African-American] [1, Caucasian] [2, Hispanic] [3, other]
MC: [0, single] [1, married] [2, separate] [3, widowed] [4, significant other] [5, divorced] [6, unknown]

"""
# all_attributes = ['sexC', 'ageC','raceC', 'MC','priors_count_C']

selected_attributes = ['sexC',  'raceC', 'MC']

original_data_file = r"../../../../InputData/CompasData/Preprocessed_classified/RecidivismData_13att_classified_testdata.csv"
mis_data_file = r"../../../../InputData/CompasData/Preprocessed_classified/RecidivismData_13att_classified_mis.csv"

output_path = r'../../../../OutputData/CaseStudy/COMPAS/low_acc/3att_1.txt'
output_file = open(output_path, "w")

output_file.write("selected_attributes: {}\n".format(selected_attributes))

less_attribute_data = pd.read_csv(original_data_file)[selected_attributes]
mis_class_data = pd.read_csv(mis_data_file)[selected_attributes]

overall_acc = 1 - len(mis_class_data) / len(less_attribute_data)



print("overall_acc = {}\n".format(overall_acc))

thc = 20
time_limit = 5 * 60
tha = overall_acc - 0.15

output_file.write("overall_acc = {}, thc = {}, tha = {}\n".format(overall_acc, thc, tha))

pattern_with_low_accuracy1, calculation1_, t1_ = newalg.GraphTraverse(less_attribute_data,
                                                                      mis_class_data, tha,
                                                                      thc, time_limit)

print("newalg, time = {} s, num_calculation = {}\n".format(t1_, calculation1_))
print("num of patterns detected = {}".format(len(pattern_with_low_accuracy1)))
for p in pattern_with_low_accuracy1:
    print(p)

output_file.write("newalg, time = {} s, num_calculation = {}\n".format(t1_, calculation1_))
output_file.write("num of patterns detected = {}\n".format(len(pattern_with_low_accuracy1)))
for p in pattern_with_low_accuracy1:
    output_file.write(str(p))
    output_file.write("\n")

# pattern_with_low_accuracy2, calculation2_, t2_ = naivealg.NaiveAlg(less_attribute_data,
#                                                                    mis_class_data, tha,
#                                                                    thc, time_limit)
# print("naivealg, time = {} s, num_calculation = {}".format(t2_, calculation2_), "\n",
#       pattern_with_low_accuracy2)
