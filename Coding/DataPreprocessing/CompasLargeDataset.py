import numpy as np
import pandas as pd
import random
from Algorithms import pattern_count


def generateTuple(original_data):
    describe_data = original_data.describe()
    attributes = original_data.columns.tolist()
    tuple = list()
    for att in attributes:
        if att == 'score' or att == "Violence_score" or att == "Recidivism_score":
            min = (describe_data[att]['min'])
            max = (describe_data[att]['max'])
            a = random.uniform(min, max)
            tuple.append(a)
        else:
            min = int(describe_data[att]['min'])
            max = int(describe_data[att]['max'])
            a = random.randint(min, max)
            tuple.append(a)
    return tuple

def extendDataset(newsize, filepath, output_pre):
    global tp
    original_data = pd.read_csv(filepath)

    oldsize = len(original_data)
    num_new_tuples = newsize - oldsize
    print("num_new_tuples={}".format(num_new_tuples))
    for i in range(num_new_tuples):
        tp = generateTuple(original_data)
        original_data.loc[len(original_data)] = tp
        if i % 1000 == 0:
            print(i, tp)
    output_path = output_pre + str(newsize) + ".csv"
    original_data.to_csv(output_path, index=False)


def UpdateAndRank(file_path, ranked_by):
    data = pd.read_csv(file_path)
    data['rank'] = data[ranked_by].rank(method='first', na_option='bottom', ascending=False)
    data = data.sort_values(by='rank', ascending=True)
    data.to_csv(file_path, index=False)



def ExtendAndRank(size, input_file_path, output_pre, ranked_by_att):
    extendDataset(size, input_file_path, output_pre)
    new_file_path = output_pre + str(size) + '.csv'
    UpdateAndRank(new_file_path, ranked_by_att)


input_file_path = "../../InputData/CompasData/ForRanking/LargeDatasets/10.csv"
output_pre = r"../../InputData/CompasData/ForRanking/LargeDatasets/"
ExtendAndRank(20, input_file_path, output_pre, 'score')


