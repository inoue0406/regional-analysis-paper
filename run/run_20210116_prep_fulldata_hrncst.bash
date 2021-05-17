#!/usr/bin/bash

python ../src_prep/prep_hrncst_JMA_alljapan.py 2017-01 >& log-hr-01 
#python ../src_prep/prep_hrncst_JMA_alljapan.py 2017-02 >& log-hr-02 &
python ../src_prep/prep_hrncst_JMA_alljapan.py 2017-03 >& log-hr-03
#python ../src_prep/prep_hrncst_JMA_alljapan.py 2017-04 >& log-hr-04 &
#python ../src_prep/prep_hrncst_JMA_alljapan.py 2017-05 >& log-hr-05
#python ../src_prep/prep_hrncst_JMA_alljapan.py 2017-06 >& log-hr-06

