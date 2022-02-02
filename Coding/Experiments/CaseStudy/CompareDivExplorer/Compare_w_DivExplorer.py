import pandas as pd
import os
from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer




inputDir=os.path.join(".", "datasets")


df= pd.read_csv(os.path.join(inputDir, "compas_discretized.csv"))
class_map={'N': 0, 'P': 1}
df.head()


import time

time1 = time.time()
min_sup=0.01 # 61.72 size threshold
# Input: a discretized dataframe with the true class and the predicted class.
# We specify their column names in the dataframe
# The class_map is a dictionary to specify the positive and the negative class (e.g. {"P":1, "N":0})
fp_diver=FP_DivergenceExplorer(df,"class", "predicted", class_map=class_map)
#Extract frequent patterns (FP) and compute divergence
##min_support: minimum support threshold
##metrics: metrics=["d_fpr", "d_fnr"]
# (default metric of interest: False Positive Rate (FPR) d_fpr, False Negative Rate (FNR) d_fnr, Accuracy divergence)
FP_fm=fp_diver.getFrequentPatternDivergence(min_support=min_sup, metrics=["d_fpr"])
time2 = time.time()
print("running time = {}s".format(time2 - time1))
print(f"Number of frequent patterns: {len(FP_fm)}")


FP_fm_unfair = FP_fm[FP_fm["d_fpr"] > 0.1]
print(len(FP_fm_unfair))
print(FP_fm_unfair[:20])



#####  my alg
print("======================= our alg ===================\n")

TP = df[(df['class'] == 1) & (df['predicted'] == 1)]
FP = df[(df['class'] == 0) & (df['predicted'] == 1)]
TN = df[(df['class'] == 0) & (df['predicted'] == 0)]
FN = df[(df['class'] == 1) & (df['predicted'] == 0)]

print(len(TP) + len(FP) + len(TN) + len(FN))

selected_attributes = ['age', 'charge', 'race', 'sex', '#prior', 'stay']
df = df[selected_attributes]
TP = TP[selected_attributes]
TN = TN[selected_attributes]
FP = FP[selected_attributes]
FN = FN[selected_attributes]


from Algorithms import NewAlgGeneral_StatisticalSignificant_0_20220125 as newalg

thc = 61.72
time_limit = 5 * 60
fairness_definition = 1  # FPR = FP/(FP+TN) False_positive_error_rate_balance, but for those treated too well
delta_thf = 0.1
pattern_with_low_fairness1, sizes_of_patterns, fairness_values_of_patterns, t_values_of_patterns,\
p_values_of_patterns, num_patterns, t1_ = newalg.GraphTraverse(df,
                                         TP, TN, FP, FN, delta_thf,
                                         thc, time_limit, fairness_definition)


print("newalg, time = {} s, num_calculation = {}\n".format(t1_, num_patterns))
print("num of patterns detected = {}".format(len(pattern_with_low_fairness1)))
for i in range(len(pattern_with_low_fairness1)):
    print("{} {} {} {} {}\n".format(str(pattern_with_low_fairness1[i]),
                              sizes_of_patterns[i], fairness_values_of_patterns[i],
                                 t_values_of_patterns[i], p_values_of_patterns[i]))

