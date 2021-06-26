# 2021.05.30 working
### Explanations for figures
#### Low acc

- Adult data
    - datasize: why??
    - num att: ok
    - tha: ok.
      There are two points decided by Tha whether to go deeper or not.
        1.
                if mis_class_cardinality < (1 - Tha) * Thc:
                    continue
        The larger the delta, the smaller tha, the larger the right hand side of the inequation, the easier it is to be satisfied.
      
        So the more likely to skip here.
      
        2. 
                if accuracy >= Tha:
                    children = GenerateChildren(P, whole_data_frame, attributes)
                    S = S + children
                    continue
        The larger the delta, the smaller tha, the easier this will be satisfied, the more likely to go deeper here.
      
        These two points are competing against each other so the result is a trade off.
      
    - thc:  need explanation, so I run again
        
        The naive alg is getting faster and faster since when thc increase, some nodes are excluded early.
      
- Compas data
    - datasize: ok
    - num att: ok
    - tha: Need explanation, so rerunning
    - thc: how to explain the time? It's always 10-13s.
    
      
- creditcard data
    - datasize: doesn't make sense, running again
    - num att: ok
    - tha: too werid, running again
      
    - thc: running
    

#### General case

1. How to explain datasize and thc??????

- Adult data
    - datasize: werid, and maybe shouldn't have this figure.
    - num att: ok
    - tha: # of nodes always increase, but time increase then decrease
      Other datasets are the same, so print more info and rerun
    - thc: werid, running
    

- Compas data
    - datasize: werid, running
    - num att: ok
    - tha: running
    - thc: werid, running
    

- Creditcard data
    - datasize: running
    - num att: ok
    - tha: running
    - thc: werid, running
    


# 2021.06.02 meeting with Yuval

### Things I want to discuss:
1. Tha problem
2. ranking alg


### Tasks! DDL of the paper is July 2nd.
1. Have the result of new ranking alg by next week.
2. Complexity


# 2021.06.11 meeting with Yuval, Jag
1. New experiment: 50 datasets on UCI, show that they all have unfairness


# 2021.06.18 meeting with Yuval
1. Can change grades, but keep the true positives ...
2. better not change failures
3. Figure 11/12 b should decrease?
4. case study: compas, bad for black. 
black have higher false negative
hispanic female higher error
   
all definition: accuracy, false positive, ranking
find 


# 2021.06.22 meeting with Yuval
1. Finish the example in section 5. Yuval will change bullets to pseudocode.
2. Proof, second part, s_D >= n-m-1 !!! It's incorrect now
3. threshold of size: [10, 100]
4. experiments, run again and explain.

# 2021.06.23
1. proof of the two proposition
2. new experiments and results

1. remove figure 12, repeat for 10, 11, with constant lower bound/upper bound
2. case study: same preprocessing as coverage/propublica
need to show black have high FN, white have high FP
report other groups is good, but need to have what they find
3. Eg 5.2. Keep the order of attributes. Add the number of failures.




