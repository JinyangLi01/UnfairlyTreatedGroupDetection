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


---
**Problems:**
- how to further improve new algorithm?
- time depends on attribute number, not data size!!!!


## 2020.12.02 meeting with Yuval 

1. use correct end-to-end experiment pipeline

2. Take part of attributes: age, gender, ... helps to debug and analyze

3. what is the most time-consuming part in the code, how to improve

4. what happens to the other datasets

5. whether we can explain the reason. look at the decision tree. [optional]


## 2020.12.11 work

finish 1 and 2 of the tasks.

TODO: 3, 4, 5






