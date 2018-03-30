#!/bin/bash

seqlen=100
wordlen=20

python run_charcnn.py \
       --batch_size 20 \
       --data ~/data/reuters/reuters2/reuters2-char-seqlen-${seqlen}-wordlen-${wordlen}.npz \
       --drop_rate 0.2 \
       --embedding_dim 128 \
       --epochs 5 \
       --feature_maps 25 50 75 100 125 150 \
       --highways 1 \
       --kernel_size 1 2 3 4 5 6 \
       --lstm_units 256 \
       --lstms 2 \
       --n_classes 2 \
       --name reuters2-char-tanh-seqlen-${seqlen}-wordlen${wordlen} \
       --seqlen ${seqlen} \
       --bipolar \
       --vocab_size 128 \
       --wordlen 20
