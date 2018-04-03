#!/bin/bash

seqlen=100
wordlen=20
adv_eps=0.2
dataset=reuters2

python charcnn_fgm.py \
       --batch_size 20 \
       --data ~/data/reuters/${dataset}/${dataset}-char-seqlen-${seqlen}-wordlen-${wordlen}.npz \
       --drop_rate 0.2 \
       --embedding_dim 128 \
       --feature_maps 25 50 75 100 125 150 \
       --highways 1 \
       --kernel_size 1 2 3 4 5 6 \
       --lstm_units 256 \
       --lstms 2 \
       --n_classes 2 \
       --name ${dataset}-char-sigm-seqlen-${seqlen}-wordlen-${wordlen} \
       --seqlen ${seqlen} \
       --unipolar \
       --vocab_size 128 \
       --wordlen 20 \
       --adv_epochs 5 \
       --adv_eps ${adv_eps} \
       --fgsm \
       --outfile ${dataset}-char-fgsm-eps-${adv_eps} \
       --samples -1 \
