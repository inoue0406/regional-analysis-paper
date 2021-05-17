#!/usr/bin/bash

python ../src_prep/prep_radarJMA_fulldata_alljapan.py 2015 >& log1 &
python ../src_prep/prep_radarJMA_fulldata_alljapan.py 2016 >& log2 &
python ../src_prep/prep_radarJMA_fulldata_alljapan.py 2017 >& log3 &

