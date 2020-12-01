## 2020.11.30

Implemented naive and new algorithm.

**Experiment results:**

***Experiment 1***
 
 whole_data_file = "../InputData/SmallDataset/SmallWhole_13_100.csv"
 mis_class_data_file = "../InputData/SmallDataset/SmallMisClass_13_50.csv"


| configuration | |
| --- | --- |
| data size (misclassfied/whole) | 50/100 |
| number of attributes| 13|
| Tha | 0.5|
| Thc | 6 |
| sanity check | yes|



| | New alg | Naive alg |
| ----   | ----- | ----
| execution time | 1.17s | 1.22h |
| num patterns checked |  121354 | 489440389 |
| number of patterns found | 33 | 33|
---


***Experiment 2***
 
whole_data_file = "../InputData/CleanAdult3.csv"
mis_class_data_file = "../InputData/mis_class3.csv"
Tha = 0.5
Thc = 30
number of attributes = 13



| | New alg | Naive alg |
| ----   | ----- | ----
| execution time | 25min | too long, interrupted |
| num patterns checked |  8361542 | |
| number of patterns found | 4 | |


