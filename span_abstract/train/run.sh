#!/usr/bin/bash

basepath=$(cd `dirname $0`; pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath


# 以下是样例，你可以自定义修改
python train.py \
    --input_file_dir=$TRAIN_FILE_DIR \
    --save_model_dir=$SAVE_MODEL_DIR \
    --dev_file_dir=$PREDICT_FILE_DIR \
    --bert_model_dir="/read-only/common/pretrain_model/tensorflow/chinese_rbt3_L-3_H-768_A-12/"