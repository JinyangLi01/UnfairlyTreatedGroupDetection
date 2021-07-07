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

## 2020.12.18 work

The most time-consuming part of new algorithm: pattern_count :) 




## 2020.12.19 meeting with Yuval

1. pattern_count is time-consuming, maybe we need try to avoid executing this function in some cases.

2. generate graph for experiments. 
y axis: execution time, num_patterns_checked
x axis:
 - number of attributes
 - datasize
 - threshold of mis-classified size
 - delta fairness value
 
 - run on two different datasets, generate graphs for each

 use python or excel

 python: easy to change
 

3. find patterns with at most 5 attributes. patterns are not too specific

4. group have different grades from true grades, not binary


## 2020.12.26 working

1. the number of choices of an attribute: 15 or 4


## 2020.12.30 meeting with Yuval

1. data size experiment: start from the whole dataset, and increase it by generating random tuples

2. use only new alg, use whole dataset, all att, and all tuples.
check if the output set is empty

3. check a pattern means check its accuracy

4. tha: decrease? increase? check it! 

5. thc: make sense. run it 10 times and record average

6. number of attributes: running time problem. need to check

7. one scenario where the result is not empty. and show problematic patterns.

8. try more att in compas dataset? now I use 6. maybe try 7?


## 2020.1.5 meeting with Yuval

1. most naive alg

2. graphs:

- data_size graph: running time, number of calculations
- number of attribute: same
- thc: both. thc also affects naive alg, since thc is in definition

- tha: only do it for new alg

## 2020.1.15 working


| dataset | num_att | thre of car | thre of acc | datasize |
| --- | --- | --- | --- | --- |
| adult | running |  running | done, need explain  |   |
| compas |  done  | running  |  done, need explain  |    |
| credit card |     |    |     |    |

## 2021.1.20 Meeting with Yuval

1. check for AdultDataset/tha_calculations. Why it decreases from 0.05 to 0.1. 
done

2. thc_time for AdultDataset. Why it goes up finally? 

3. Other datasets have similar figures?

4. Start to write paper.

5. Prove no polynomial time. Similar to Theorem 1 in Coverage paper.
- Have a similar theory and prove it.
- Read Paper 'Fairness definitions explained'. See which definitions can be applied in our paper.
This is a better task! Which definitions can we support and what to do in order to support them.

6. Generate examples from each dataset. 

7. Complexity of algorithm. We can do it later.



## 2020.01.26 Working

1. the current alg has an error:
   
We should use machine learning once, generate the mis-classified set. 
   And then apply newalg/naivealg on this mis-classified set with different configurations.
   
But right now every time we run newalg/naivealg, we build machine learning model from scratch.
This means every time the mis-classified set is different.

2. thc_time for AdultDataset. Why it goes up finally? 

because 



## 2021.02.03 meeting with Yuval

1. look at the 7th attribute.

2. data size figure time limitation?

3. definition paper: apply to our algorithm, and fine groups which makes the ml model unfair w.r.t. the definition.


Tasks:
1. check figure issues.
2. apply different definitions to algorithm

Different definitions may give different groups.


3. we have a ranking algorithm, divide into good/bad ranking.

Read ranking papers.




