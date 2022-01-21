import pandas as pd

from itertools import combinations
from Algorithms import pattern_count
import time
from Algorithms import NewAlgGeneral_SizeFairnessValue_3_20210111 as newalg

"""
SEX:
1:male
2:female

RACE:
1: non hispanic white
0:other


REGION:
The values and states for each region include the following:

Value	Label	States
1	Northeast	Connecticut, Maine, Massachusetts, New Hampshire, New Jersey, New York, Pennsylvania, Rhode Island, and Vermont
2	Midwest	Indiana, Illinois, Iowa, Kansas, Michigan, Minnesota, Missouri, Nebraska, North Dakota, Ohio, South Dakota, and Wisconsin
3	South	Alabama, Arkansas, Delaware, District of Columbia, Florida, Georgia, Kentucky, Louisiana, Maryland, Mississippi, North Carolina, Oklahoma, South Carolina, Tennessee, Texas, Virginia, and West Virginia
4	West	Alaska, Arizona, California, Colorado, Hawaii, Idaho, Montana, Nevada, New Mexico, Oregon, Utah, Washington, and Wyoming
"""




def ReadCateFile(cate_file):
    translation = dict()
    f = open(cate_file, "r")
    Lines = f.readlines()
    start = True
    key = str()
    att = dict()
    LastLineIsEmpty = False
    for line in Lines:
        if line == "\n":
            if LastLineIsEmpty:
                break
            LastLineIsEmpty = True
            translation[key] = att
            att = dict()
            start = True
            continue
        LastLineIsEmpty = False
        line = line.strip()
        if start:
            att = dict()
            key = line
            start = False
        else:
            items = line.split(":")
            att[items[0]] = items[1]
    if not LastLineIsEmpty:
        translation[key] = att
    return translation


def TranslatePatternsToNonNumeric(pattern_with_low_fairness, translation_file, selected_attributes):

    translaion = ReadCateFile(translation_file)
    results = []
    for p in pattern_with_low_fairness:
        re = dict()
        idx = 0
        for i in p:
            if i == -2:
                idx += 1
                continue
            else:
                attribute = selected_attributes[idx]
                re[attribute] = translaion[attribute][str(i)]
            idx += 1
        results.append(re)
    return results






selected_attributes = ['SEX', 'REGION', 'MARRY', 'RACE', 'FTSTU',
                       'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX',
                       'CHDDX', 'ANGIDX', 'MIDX']



original_data_file = r"../../../../InputData/MedicalDataset/train/train_add_col2PREGNT.csv"
TP_data_file = r"../../../../InputData/MedicalDataset/train/train_TP_add_col2PREGNT.csv"
FP_data_file = r"../../../../InputData/MedicalDataset/train/train_FP_add_col2PREGNT.csv"
TN_data_file = r"../../../../InputData/MedicalDataset/train/train_TN_add_col2PREGNT.csv"
FN_data_file = r"../../../../InputData/MedicalDataset/train/train_FN_add_col2PREGNT.csv"


# original_data_file = r"../../../../InputData/MedicalDataset/train/train_41att.csv"
# TP_data_file = r"../../../../InputData/MedicalDataset/train/train_TP_41att.csv"
# FP_data_file = r"../../../../InputData/MedicalDataset/train/train_FP_41att.csv"
# TN_data_file = r"../../../../InputData/MedicalDataset/train/train_TN_41att.csv"
# FN_data_file = r"../../../../InputData/MedicalDataset/train/train_FN_41att.csv"


def read_with_att(original_data_file, selected_attributes):
    original_data = pd.read_csv(original_data_file)
    less_attribute_data = original_data[selected_attributes]
    return less_attribute_data


less_attribute_data = read_with_att(original_data_file, selected_attributes)
TP = read_with_att(TP_data_file, selected_attributes)
FP = read_with_att(FP_data_file, selected_attributes)
TN = read_with_att(TN_data_file, selected_attributes)
FN = read_with_att(FN_data_file, selected_attributes)



output_path = r'../../../../OutputData/CaseStudy/Medical/FP.txt'
output_file = open(output_path, "w")

output_file.write("selected_attributes: {}\n".format(selected_attributes))
print("selected_attributes:{}".format(selected_attributes))


thc = 150
time_limit = 5 * 60

fairness_definition = 1 # FPR = FP/(FP+TN) False_positive_error_rate_balance, but for those treated too well
delta_thf = 0.1


output_file.write("fairness_definition = {}, thc = {}, delta_thf = {}\n".format(fairness_definition, thc, delta_thf))


pattern_with_low_fairness, sizes_of_patterns, fairness_values_of_patterns, \
    calculation1_, t1_ = newalg.GraphTraverse(less_attribute_data,
                                                  TP, TN, FP, FN, delta_thf,
                                                  thc, time_limit, fairness_definition)


print("newalg, time = {} s, num_calculation = {}\n".format(t1_, calculation1_))
print("size threshold = {}, delta fairness = {}".format(thc, delta_thf))
print("num of patterns detected = {}".format(len(pattern_with_low_fairness)))
print("pattern, size, original accuracy:")
for i in range(len(pattern_with_low_fairness)):
    print(pattern_with_low_fairness[i], sizes_of_patterns[i], fairness_values_of_patterns[i])



output_file.write("newalg, time = {} s, num_calculation = {}\n".format(t1_, calculation1_))
output_file.write("num of patterns detected = {}\n".format(len(pattern_with_low_fairness)))
for p in pattern_with_low_fairness:
    output_file.write(str(p))
    output_file.write("\n")



