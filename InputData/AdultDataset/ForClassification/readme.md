# data files:
1. AdultDataset/adult.csv: original dataset
2. folder ForClassification: proprocessed data for classification experiments.
These csv files are out put from AdultDataset_otiginal.ipynb and AdultDataset_categorize.ipynb
   

- CleanAdult.csv: original dataset after removing invalid rows
- CleanAdult_numerical.csv: 
 change categorical values to numeric
 education and education-num are the same, one-to-one mapping
category_col =['workclass', 'race', 'education','marital-status', 'occupation',
               'relationship', 'gender', 'native-country', 'income']
  
- CleanAdult_numerical_cat.csv: 
categorize attributes: age, capital-gain, capital-loss, hours-per-week
According to AdultDataset_categorize.ipynb
  
- CleanAdult_numerical_mis.csv: apply classifier to CleanAdult_numerical.csv, and get mis-classified data
- CleanAdult_numerical_mis_cat.csv:
categorize attributes: age, capital-gain, capital-loss, hours-per-week
According to AdultDataset_categorize.ipynb
  


# Large datasets

1. In folders AdultDataset/LargeDatasets_cat/ and AdultDataset/LargeDatasets_without_cat

These are files w/ or w/o categorazing attributes like age



