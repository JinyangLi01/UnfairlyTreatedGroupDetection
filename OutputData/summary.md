# 2022.May, experiments of ranking for CIKM

proportional bounds (definition2):


|            | student | compas | german |
|------------|---------| --- |--------|
| num of att | done    |  done  | done   |
| range k    | done    |   done   | done   |
| thc        | done    |  done     | done   |






## experiment progress

|   | adult | compas | credit card |
| --- | --- | --- | --- |
| datasize | done  |  running    |   running    |
| num of att | when 7 att, naive has less calculation but more time compared to 13 att new alg?   |  done  |  done  |
| thc   |  done    |   done   |   done    |
| tha    |   done   |  done     |   done    |


compas datasize:

Naive alg, the line of calculations is flat and high, but the line of time grows from low to high.

Possible reason: when data size is small, time of one cardinatlity calculation is small.


datasize: extend times:
Adult: 2.2
compas: 2.9
creditcard: 
