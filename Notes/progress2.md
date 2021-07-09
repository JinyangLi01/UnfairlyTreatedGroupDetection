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



# 2021.06.30
1. related work: ranking done. Need related work for demo.
2. exp figures, default setting. Need times of data size.
3. figures for ranking, num att, data size


TODO:
1. related work for demo
done 2. update exp figures for classification.
3. data size is extended up to ? X???
4. anonymous github repo and gmail
done 5. log scale for thc

6. Figure 10-13 only blue lines, linear x-axis
7. we need to explain which attributes we use and why

If I have time, do:
1. mushroom
2. student dataset for classification
3. use more att in Figure 13, 14

TODO:
1. set up
done 2. figure 6b!!
   
done 3. make figures with only blue lines. 
done 4. make Figure 4 x-axis to 12.
5. use all attributes for ranking. 

6. Figure 15, 16 use all attributes! 
done 7. case study
8. table


Missing:
done 1. Figure 12 14 att
4. Figure 4, 5, use up to 16 attributes for optimized alg in (b), and same for (c).
   COMPAS: due to reason, use only 11
   creditcard:running num att low acc, and for general FPR
done 2. table
1. desctiption of datasets
3. ranking, data sizes


Questions:
1. use test+train as whole set
mis as mis


