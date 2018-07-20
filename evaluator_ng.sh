#! /bin/sh

TEST_PATH=../machine-tasks/CommaiMini-^$/New_AG
EXPT_DIR=./New_AG/gru_learned_ng_longer_E128_H128/
ATTN='pre-rnn'
ATTN_METHOD='mlp'
TF=1

echo "Run in evaluation mode"
python evaluate3.py \
        --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) \
        --test_data $TEST_PATH \
        --attention $ATTN \
        --attention_method $ATTN_METHOD \
        --use_attention_loss \
        --teacher_forcing $TF