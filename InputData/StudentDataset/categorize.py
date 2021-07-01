import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



student_data = pd.read_csv("../../InputData/StudentDataset/student-mat.csv", sep=';')
print(len(student_data))