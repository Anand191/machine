#! /bin/sh

TEST_PATH=../machine-tasks/CommaiMini-^$/New_AG/Long
EXPT_DIR=./New_AG/lstm_learned_ng_longer_E128_H128/
OUT_DIR=attention_plots/micro
ATTN='pre-rnn'
ATTN_METHOD='mlp'
TF=1
echo "Run in inference mode"
python infer_micro.py \
        --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) \
        --test $TEST_PATH \
        --output_dir $OUT_DIR \
        --attention $ATTN \
        --attention_method $ATTN_METHOD \
        --use_attention_loss \
        --teacher_forcing $TF