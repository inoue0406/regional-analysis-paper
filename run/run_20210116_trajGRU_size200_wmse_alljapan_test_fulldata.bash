#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

case="result_20210116_trajGRU_size200_alljapan_wmse_test_fulldata"

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
"IJ_5_6"
"IJ_5_7"
"IJ_5_8"
"IJ_5_9"
"IJ_6_6"
"IJ_6_7"
"IJ_6_8"
"IJ_6_9"
"IJ_7_10"
"IJ_7_7"
"IJ_7_8"
"IJ_7_9"
"IJ_8_10"
"IJ_8_11"
"IJ_8_12"
"IJ_8_13"
"IJ_8_8"
"IJ_8_9"
"IJ_9_10"
"IJ_9_11"
"IJ_9_12"
"IJ_9_13"
"IJ_9_9"
)

for ijstr in "${ijstr_list[@]}"; do
    # running script for Rainfall Prediction with ConvLSTM
    python ../src/main_trajGRU_jma.py --model_name trajgru --no_train\
       --dataset radarJMA --model_mode run --data_scaling linear\
       --aug_rotate 0 --aug_resize 0\
       --data_path ../data/data_alljapan/ --image_size 200\
       --valid_data_path ../data/data_alljapan_fulldata/$ijstr/ \
       --train_path ../data/train_alljapan_2yrs_JMARadar.csv \
       --valid_path ../data/filelist_fulldata/valid_JMARadar_$ijstr.csv\
       --test --eval_threshold 20 --test_path ../data/filelist_fulldata/valid_JMARadar_$ijstr.csv\
       --result_path $case --tdim_use 12 --tdim_loss 12 --learning_rate 0.0002 --lr_decay 0.9\
       --test_tail $ijstr\
       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --loss_function WeightedMSE\
       --optimizer adam
    # post plot
    #python ../src_post/plot_comp_prediction_trajgru.py $case
    #python ../src_post/gif_animation.py $case
done

