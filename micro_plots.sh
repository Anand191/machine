#! /bin/sh

TEST_PATH=../machine-tasks/CommaiMini-^$/New_Guidance/Sample1/Long
EXPT_DIR=./MicroTask_Runs/lstm_hard_sample1_E128_H128/
OUT_DIR=attention_plots/micro_new
ATTN='pre-rnn'
ATTN_METHOD='hard'
TF=1
echo "Run in inference mode"
python infer_micro.py \
        --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) \
        --test $TEST_PATH \
        --output_dir $OUT_DIR \
        --attention $ATTN \
        --attention_method $ATTN_METHOD \
        --teacher_forcing $TF \
        #--use_attention_loss