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


