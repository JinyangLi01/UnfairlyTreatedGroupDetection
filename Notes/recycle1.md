# Oct. 13, 2021
1. Add experiments using medical dataset. (already done)
2. Naive algorithm needs to change: now it is too naive...
Set a stop condition: when all patterns with x deterministic attributes have a size smaller than the size threshold

# Oct 20, 2021
1. The stop condition does not work often.
We don't use too many attributes for naive alg to finish, so whent he maximum number of attributes is small,
some patterns always have large sizes, so the stop condition is never satisfied.
2. In the submitted paper, we said there is no pruning based on sizes for naive alg. A reviewer said this is not good.
The only way to prune based on size is to add such a stop condition, since we don't use the pattern graph, 
we can't prune based on sizes like the optimized alg.


# Nov. 1, 2021
Extend Section 5 ranking algorithm
   1. support another definition: proportationate to size of the group.
   2. explain this in text. In line x, we change x, ....
   3. go over survey paper ranking definitions, see whether they can be easily adopted.
   4. add a para to section 5.


# Nov. 4, 2021
1. proportional representation is the same category, is it ok?
2. Proportional representation can't use our alg...
   1. bounds for different patterns are different
   2. it is possible that a child pattern satisfy the bounds while parent doesn't.
   3. eg:                 top20   top22    top23               
      Dataset: 100         20       22      23        
      female: 50            7       8       8        
      black female: 20      2       3       3        
      
      F:     20:    20(0.5-0.15) <= x <= 20(0.5+0.15)        7 <= x <= 13       
        22: 7.7<=x<=...         
        23: 8.05 <= x <= ...       
      BF:    20:   20(0.2-0.15) <= x <= 20(0.2+0.15)         1 <= x <= 7        
        22:    1.1 <= x <= 7.7            
        23:   1.15 <= x <= 8.05             
   
        
3. This alg is not applicable to Discounted cumulative fairness
   Discounted cumulative fairness computes the cumulative values of top-10, 20, 30... with a discount.
   Only one result is got from discrete positions.

4. This alg is not applicable to fairness of exposure either.      
   Fairness of exposure computes one value for position 1-n cumulatively.

paragraph describing fairness of exposure:    

        Another definition is fairness of exposure approach \cite{SinghJ18}, which pays attention to the fact that there can be a large skew in the distribution of exposure due to position bias.
        This approach assigns a discount value $v_j$ to each position $j$ representing the importance of the position. It assumes ranks are probabilistic, with $P_{t,j}$ being the probability that tuple $t$ is ranked at position $j$.
        In a ranking, exposure of tuple $t$ is $Exposure(t) = \sum_{j=1}^{N}P_{t, j}v_j$, and exposure of pattern $p$ is $Exposure(p) = \frac{1}{|s_D(p)|} \sum_{t \in p} Exposure(t)$.



# Nov. 5, 2021 
meeting with Yuval
1. Think about the problem in ranking, when bounds are all different.     
    The worse case is to go all the way to the root. [check how alg does it now!!!]    
    Do math. See the relationship between patterns. Do we have other better solutions to prune?     

  

# Nov. 7, 2021

- current unfairness detection alg for ranking:
  - use alg for classification when $k=k_{min}$
  - when k increases by 1, there is a new tuple $t_{new}$
    - check all patterns related to $t_{new}$, whose number in top-k increases by 1 due to $t_{new}$    
    - check them, if lower than lower bound, add to result set and add the parent to stop set
    - if either of the bounds change, we check all patterns in the stop set for lower bound
    - go up until 1. the root or 2. a pattern above the lower bound

- when bounds for different patterns are different, the last point above is not applicable.
- it is possible that a child pattern has a size above its lower bound while it is the opposite for the parent
- so we need to go all the way until the root, to find the most general pattern violating the lower bound.

# Nov. 8, 2021
1. The current lag needs to go all the way until the root
2. Another approach: 
    - For patterns satisfying the new tuple, we have to check them anyway
    - For other patterns, we store the k value for each node, which means its bounds are good before a certain k value
    - And then for each node we store a pointer pointing to the ancestor node with the smallest k value

3. Implement alg in 2, and do experiment. 
4. Plan to submit this paper Dec 1st.
5. Read the paper TKDE and understand what’s the difference. We need to cite them.



# Nov. 9, 2021
questions while writing code implementing NewAlgRanking_definition2_0_20211108. Alg for ranking, definition 2

- Can we make original alg the same? Maintaining a set with smallest k?
- In this alg, can we make the stop set to be the nodes with smallest k, instead of where we stopped?
  - No. The next pattern at position k may satisfy these patterns so their k values will change.

# Nov. 21, 2021

If a pattern's size is too small, should it be added to stop set or should its parent?
- add itself: not allow dominance in stop set 
- add its parent: must allow dominance in stop set. But, when we update k value in function Update_k_value()    
When should we stop? We are supposed to stop when we reach stop set. So we can't allow dominance in stop set:)


# Nov. 28, 2021

Current problems:

- Some patterns in stop set need future checking for larger k, other don't. We can't tell.    
Eg. p is added to result set and then removed by another pattern dominating p.    
so p is also in stop set. I get calculating p repeatedly later...
    - solution: maintain two stop sets.
    - Now I'm implementing this solution in NewAlgRanking_definition2_2_20211121.py
  
- In CheckCandidatesForKValues(), we go up from stop set, so some patterns share the same upper branch but we check them twice.    
    - solution: remember patterns one level above stop set and avoid double checking for them.    
  




# Dec. 9, 2021

TODO:
1. finish implementation ASAP. Don't use recursive functions...
2. Do medical dataset for all classification experiments
3. Add description for medical dataset in paper draft
4. Good luck >_<






