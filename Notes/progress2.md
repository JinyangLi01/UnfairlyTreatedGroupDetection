# 2021.05.30 working
### Explanations for figures
#### Low acc

- Adult data
    - datasize: running
    - num att: running
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
      
    - thc: running
        The naive alg is getting faster and faster since when thc increase, some nodes are excluded early.
      
- Compas data
    - datasize: ok
    - num att: ok
    - tha: Same reason
      
    - thc: running
    
      
- creditcard data
    - datasize: doesn't make sense, running again
    - num att: increase num_att_max_naive to 9, running
    - tha: too werid, running again
      
    - thc: running
    

#### General case

- Adult data
    - datasize: werid, running
    - num att: set num_att_max_naive to 7, running
    - tha: running
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
2. 



