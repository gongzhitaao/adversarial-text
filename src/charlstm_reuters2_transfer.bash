#!/bin/bash

seqlen=100
wordlen=20
dataset=reuters2
adv=hotflip
n_classes=2

for c in 5 10 15 20 25 30 35 40; do # hotflip
# for c in 5; do
    fn=MS_wenlu/${dataset}-char-${adv}-c${c}-b3-c2c-seqlen-${seqlen}-wordlen-${wordlen}.npz
    python eval_charlstm.py \
           --batch_size 20 \
           --data ${fn} \
           --drop_rate 0.2 \
           --embedding_dim 128 \
           --epochs 5 \
           --feature_maps 25 50 75 100 125 150 \
           --highways 1 \
           --kernel_size 1 2 3 4 5 6 \
           --lstm_units 256 \
           --lstms 2 \
           --n_classes ${n_classes} \
           --name ${dataset}-char-tanh-seqlen-${seqlen}-wordlen-${wordlen} \
           --seqlen ${seqlen} \
           --bipolar \
           --vocab_size 128 \
           --wordlen ${wordlen}
done
