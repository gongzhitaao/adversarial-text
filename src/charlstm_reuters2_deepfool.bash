#!/bin/bash

seqlen=100
wordlen=20
adv_eps=50
batch_size=20
dataset=reuters2

python charlstm_deepfool.py \
       --adv_epochs 5 \
       --adv_eps ${adv_eps} \
       --batch_size ${batch_size} \
       --data ~/data/reuters/${dataset}/${dataset}-char-seqlen-${seqlen}-wordlen-${wordlen}.npz \
       --drop_rate 0.2 \
       --embedding_dim 128 \
       --feature_maps 25 50 75 100 125 150 \
       --highways 1 \
       --keepall \
       --kernel_size 1 2 3 4 5 6 \
       --lstm_units 256 \
       --lstms 2 \
       --n_classes 2 \
       --name ${dataset}-char-tanh-seqlen-${seqlen}-wordlen-${wordlen} \
       --seqlen ${seqlen} \
       --bipolar \
       --vocab_size 128 \
       --wordlen 20 \
       --outfile ${dataset}-char-deepfool-eps-${adv_eps}-baseline
