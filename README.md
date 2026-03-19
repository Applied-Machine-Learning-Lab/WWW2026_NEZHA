# Code for NEZHA

## Data 

Data can be downloaded from https://github.com/HonghuiBao2000/LETTER/tree/master/data 

## Implementation

1. Key components of NEZHA are included in *models/hack.py*:
  - ***MTP_Head*** for NEZHA architectures;
  - ***mtp_generate*** for GR inference;
  - ***ctr_forward*** for GR training;
2. Pipelines: *fintune.py* for training and *test.py* for inference. 
3. Data process: within *collator.py* and *dataset.py*
4. The running script is in *scripts/* (including the training and test).

## Contact 

Yejing Wang (email to:wangyejing.wyj@alibaba-inc.com)
