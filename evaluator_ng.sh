#! /bin/sh

TEST_PATH=../machine-tasks/CommaiMini-^$/New_Guidance
EXPT_DIR=./MicroTask_Runs/lstm_hard_sample5_E128_H128/
ATTN='pre-rnn'
ATTN_METHOD='hard'
TF=1

echo "Run in evaluation mode"
python evaluate3.py \
        --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) \
        --test_data $TEST_PATH \
        --attention $ATTN \
        --attention_method $ATTN_METHOD \
        --teacher_forcing $TF
        #--use_attention_loss