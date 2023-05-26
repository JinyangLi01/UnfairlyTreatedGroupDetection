
data:
RecidivismData_Original-categorized.csv  not used in my experiments

adult.csv.   main dataset used in my experiment

CleanAdult.csv       clean adult.csv by deleting illegal rows
miss_class.csv       miss-classified data from CleanAdult.csv



But some attributes have various values like age ranges from 19 to 99
Fix this by:
for index, row in data.iterrows():
    if row['age'] < 40:
        row['age'] = 0
    elif row['age'] < 60:
        row['age'] = 1
    elif row['age'] < 80:
        row['age'] = 2
    else:
        row['age'] = 3
        
    
for index, row in data.iterrows():
    if row['capital-gain'] == 0:
        row['capital-gain'] = 0
    elif row['capital-gain'] < 100:
        row['capital-gain'] = 1
    elif row['capital-gain'] < 200:
        row['capital-gain'] = 2
    elif row['capital-gain'] < 1000:
        row['capital-gain'] = 3
    else:
        row['capital-gain'] = 4
        
for index, row in data.iterrows():
    if row['capital-loss'] == 0:
        row['capital-loss'] = 0
    elif row['capital-loss'] < 50:
        row['capital-loss'] = 1
    elif row['capital-loss'] < 100:
        row['capital-loss'] = 2
    else:
        row['capital-loss'] = 3
        
for index, row in data.iterrows():
    if row['hours-per-week'] < 20:
        row['hours-per-week'] = 0
    elif row['hours-per-week'] < 40:
        row['hours-per-week'] = 1
    elif row['hours-per-week'] < 60:
        row['hours-per-week'] = 2
    elif row['hours-per-week'] < 80:
        row['hours-per-week'] = 3
    else:
        row['hours-per-week'] = 4



So we get 
CleanAdult2.csv     miss_class2.csv
CleanAdult3.csv : CleanAdult2 delete income, so it only contains attributes

miss_class3.csv : miss_class2.csv, delete the last two columns, so only attributes are left
miss_class4.csv : the first 20 rows of miss_class3.csv


Formal dataset: miss_class3.csv, CleanAdult3.csv



dataset: CleanAdult3.csv  size = 45222

M30: result in Deepdiver, Tao = 30 (delta fairness value * size threshold)
all misclassfied patterns, with tao = 30
size = 7550

P30: patterns satisfying both cardinality and accuracy requreiments
come from M30
size = 6640









