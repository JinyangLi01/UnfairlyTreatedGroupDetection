
import numpy as np
import pandas as pd
import csv

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def classify(data, datafile_prefix):
    print(datafile_prefix)

    predictors = ['sexC', 'ageC', 'raceC', 'MC', 'priors_count_C', 'c_charge_degree', 'decile_score',
                'c_days_from_compas_C',
                'juv_fel_count_C', 'juv_misd_count_C', 'juv_other_count_C']

    # splitting data arrays into two subsets: for training data and for testing data
    X_train, X_test, y_train, y_test = train_test_split(data[predictors], data['is_recid'], test_size=0.5, random_state=1)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    num = 0
    for index, row in X_test.iterrows():
        if clf.predict([row.to_list()]) != y_test.loc[index]:
            num = num + 1
            # print ('-----------------')
            # print (str(num) + '\n')
            # print (row)

    print(num)
    print("accuacy", (1 - num / len(X_test)) * 100)  # accuracy

    X_test_res = X_test.copy()
    X_test_res['act'] = y_test  # actual result
    X_test_res['pred'] = X_test.apply(lambda x: clf.predict([x.to_list()]), axis=1)
    mis_class = X_test_res.loc[X_test_res['act'] != X_test_res['pred']]

    testdata_path = datafile_prefix + "_testdata.csv"
    X_test_res.to_csv(testdata_path, index=False)

    mis_class = X_test_res.loc[X_test_res['act'] != X_test_res['pred']]
    print("total={}, mis={}".format(len(data), len(mis_class)))

    misdata_path = datafile_prefix + "_mis.csv"
    mis_class.to_csv(misdata_path, index=False)

    TPdata = X_test_res.loc[(X_test_res['act'] == X_test_res['pred']) & (X_test_res['act'] == 1)]
    print("TP:", len(TPdata))
    TPdata.to_csv(datafile_prefix + "_TP.csv")

    TNdata = X_test_res.loc[(X_test_res['act'] == X_test_res['pred']) & (X_test_res['act'] == 0)]
    print("TN:", len(TNdata))
    TNdata.to_csv(datafile_prefix + "_TN.csv")

    FPdata = X_test_res.loc[(X_test_res['act'] != X_test_res['pred']) & (X_test_res['act'] == 1)]
    print("FP:", len(FPdata))
    FPdata.to_csv(datafile_prefix + "_FP.csv")

    FNdata = X_test_res.loc[ (X_test_res['act']!=X_test_res['pred']) & (X_test_res['act']== 0) ]
    print("FN:", len(FNdata))
    FNdata.to_csv(datafile_prefix + "_FN.csv")




for i in range(6000, 22000, 2000):
    datafile_prefix = str(i)
    data = pd.read_csv(datafile_prefix + ".csv")
    classify(data, datafile_prefix)










