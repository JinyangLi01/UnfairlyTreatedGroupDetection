import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from slice_finder import SliceFinder

from ipywidgets import interact, interactive
from IPython.display import display

from bokeh.layouts import widgetbox, row
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
output_notebook()

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 17})



# data_file = r"../../../../InputData/COMPAS_ProPublica/compas-analysis-master/cox-parsed/cox-parsed_7214rows_with_labels.csv"
data_file = r"../CompareDivExplorer/divexplorer-main/notebooks/datasets/compas_discretized.csv" \


compas_data = pd.read_csv(data_file)
compas_data = compas_data[['age', 'charge', 'race', 'sex', '#prior', 'stay', 'class']]


# # drop nan values
# adult_data = adult_data.dropna()

# Encode categorical features
encoders = {}
for column in compas_data.columns:
    if compas_data.dtypes[column] == np.object:
        le = LabelEncoder()
        compas_data[column] = le.fit_transform(compas_data[column])
        encoders[column] = le
        print(column, le.classes_, le.transform(le.classes_))

X, y = compas_data[compas_data.columns.difference(["class"])], compas_data["class"]

pickle.dump(encoders, open("compas.pkl", "wb"), protocol=2)

# Train a model
#lr = LogisticRegression()
#lr.fit(X, y)
lr = RandomForestClassifier(max_depth=5, n_estimators=10)
lr.fit(X, y)

sf = SliceFinder(lr, (X, y))
metrics_all = sf.evaluate_model((X,y))
reference = (np.mean(metrics_all), np.std(metrics_all), len(metrics_all))


# degree: number of att in a pattern
time1 = time.time()
recommendations = sf.find_slice(k=10, epsilon=0.4, degree=6, max_workers=4)
time2 = time.time()

output_path = r'Compare_SliceFinder_6att.txt'
output_file = open(output_path, "w")

output_file.write("selected_attributes: {}\n".format(compas_data.columns.to_list()))


print("time = {}s".format(time2 - time1))

output_file.write("time = {}s".format(time2 - time1))

for s in recommendations:
    print ('\n=====================\nSlice description:')
    output_file.write('\n=====================\nSlice description:')
    for k, v in list(s.filters.items()):
        values = ''
        if k in encoders:
            le = encoders[k]
            for v_ in v:
                values += '%s '%(le.inverse_transform(v_)[0])
        else:
            for v_ in sorted(v, key=lambda x: x[0]):
                if len(v_) > 1:
                    values += '%s ~ %s'%(v_[0], v_[1])
                else:
                    values += '%s '%(v_[0])
        print ('%s:%s'%(k, values))
        output_file.write('%s:%s' % (k, values))
    print ('---------------------\neffect_size: %s'%(s.effect_size))
    print ('---------------------\nmetric: %s'%(s.metric))
    print ('size: %s'%(s.size))
    output_file.write ('---------------------\neffect_size: %s'%(s.effect_size))
    output_file.write ('---------------------\nmetric: %s'%(s.metric))
    output_file.write ('size: %s'%(s.size))

