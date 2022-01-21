import pandas as pd

from itertools import combinations
from Algorithms import pattern_count
import time
from Algorithms import NewAlg_2_20211001 as newalg
from Algorithms import NaiveAlgGeneral_1_202105258 as naivealg
from Algorithms import Predict_0_20210127 as predict


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




selected_attributes = ["REGION", "SEX", "MARRY", "RACE", "FTSTU", "ACTDTY", "HONRDC",
                       "RTHLTH", "MNHLTH", "HIBPDX", "CHDDX", "ANGIDX", "MIDX"]

#
# selected_attributes = ["REGION", "SEX", "MARRY", "RACE", "FTSTU", "ACTDTY", "HONRDC",
#                        "RTHLTH", "MNHLTH", "HIBPDX", "CHDDX", "ANGIDX", "MIDX"]

original_data_file = r"../../../../InputData/MedicalDataset/train/train_add_col2PREGNT.csv"
original_data = pd.read_csv(original_data_file)
mis_data_file = r"../../../../InputData/MedicalDataset/train/train_mis_add_col2PREGNT.csv"
mis_data = pd.read_csv(mis_data_file)


less_attribute_data = original_data[selected_attributes]
mis_class_data = mis_data[selected_attributes]


overall_acc = 1 - len(mis_class_data) / len(less_attribute_data)


output_path = r'../../../../OutputData/CaseStudy/Medical/accuracy.txt'
output_file = open(output_path, "w")

output_file.write("selected_attributes: {}\n".format(selected_attributes))



thc = 110
time_limit = 5 * 60
dif = 0.1
tha = overall_acc - dif

print("\n\ndif = {}".format(dif))
print("tha = {}, thc = {}".format(tha, thc))


pattern_with_low_accuracy, sizes_of_patterns, fairness_values_of_patterns, num_calculation, t_ = newalg.GraphTraverse(less_attribute_data,
                                                                              mis_class_data, tha,
                                                                              thc, time_limit)

print("overall_acc = {}".format(overall_acc))
print("selected attributes: {}".format(selected_attributes))

print("time = {} s, num_calculation = {}\n".format(t_, num_calculation))
print("num of patterns detected = {}".format(len(pattern_with_low_accuracy)))


# translation_file = r"../../../../InputData/MedicalDataset/train/translation.txt"
print("pattern, size, original accuracy:")
for i in range(len(pattern_with_low_accuracy)):
    # translated = TranslatePatternsToNonNumeric(pattern_with_low_accuracy[i:i+1], translation_file, selected_attributes)
    print(pattern_with_low_accuracy[i], sizes_of_patterns[i], fairness_values_of_patterns[i])



# translation_file = r"../../../../InputData/MedicalDataset/train/translation.txt"
# print("pattern, size, original accuracy:")
# for i in range(len(pattern_with_low_accuracy)):
#     translated = TranslatePatternsToNonNumeric(pattern_with_low_accuracy[i:i+1], translation_file, selected_attributes)
#     print(pattern_with_low_accuracy[i], translated[0], sizes_of_patterns[i], fairness_values_of_patterns[i])
#






