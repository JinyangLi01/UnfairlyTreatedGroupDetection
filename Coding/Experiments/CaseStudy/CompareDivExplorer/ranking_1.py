import os
import pandas as pd
from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
# from divexplorer.FP_DivergenceExplorer_original import FP_DivergenceExplorer


inputDir=os.path.join(".", "datasets")


df= pd.read_csv(os.path.join(inputDir, "GermanCredit_ranked_w_classification.csv"))

# df= pd.read_csv(os.path.join(inputDir, "compas_discretized.csv"))

class_map={'N': 0, 'P': 1}
df.head()

min_sup = 0.1
fp_diver = FP_DivergenceExplorer(df,"truth", "prediction", class_map=class_map)
FP_fm=fp_diver.getFrequentPatternDivergence(min_support=min_sup,
                                            metrics=["d_accuracy"])


FP_fm_unfair = FP_fm[FP_fm["d_accuracy"] > 10]
len(FP_fm_unfair)


print(f"Number of frequent patterns: {len(FP_fm)}")

