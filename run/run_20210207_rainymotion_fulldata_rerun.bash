#!/bin/bash

ijstr_list=(
"IJ_10_12"
"IJ_10_13"
"IJ_2_1"
"IJ_2_2"
"IJ_3_2"
"IJ_3_3"
"IJ_4_2"
"IJ_4_3"
"IJ_4_4"
"IJ_4_5"
"IJ_4_6"
"IJ_4_7"
#"IJ_5_6"
#"IJ_5_7"
#"IJ_5_8"
#"IJ_5_9"
#"IJ_6_6"
#"IJ_6_7"
#"IJ_6_8"
#"IJ_6_9"
#"IJ_7_10"
#"IJ_7_7"
#"IJ_7_8"
#"IJ_7_9"
#"IJ_8_10"
#"IJ_8_11"
#"IJ_8_12"
#"IJ_8_13"
#"IJ_8_8"
#"IJ_8_9"
#"IJ_9_10"
#"IJ_9_11"
#"IJ_9_12"
#"IJ_9_13"
#"IJ_9_9"
)

for ijstr in "${ijstr_list[@]}"; do
    case="result_20210207_rainymotion_fulldata_${ijstr}"

    # running script for Persistence foreacst
    python ../src/main_rainymotion_jma.py\
           --data_path ../data/data_alljapan_fulldata/$ijstr/\
           --valid_data_path ../data/data_alljapan_fulldata/$ijstr/ \
           --train_path ../data/filelist_fulldata/train_JMARadar_$ijstr.csv\
           --valid_path ../data/filelist_fulldata/valid_JMARadar_$ijstr.csv\
           --test --eval_threshold 0.5 10 20\
           --test_path ../data/filelist_fulldata/valid_JMARadar_$ijstr.csv\
           --result_path $case\
           --test_tail $ijstr\
           --tdim_use 12 --learning_rate 0.01 --batch_size 100\
           --n_epochs 10 --n_threads 4
done
