#!/bin/bash

seqlen=100
adv_eps=0.2
dataset=reuters2

python wordcnn_fgm.py \
       --adv_batch_size 16 \
       --adv_epochs 5 \
       --adv_eps ${adv_eps} \
       --batch_size 64 \
       --data ~/data/reuters/${dataset}/${dataset}-word-seqlen-${seqlen}.npz \
       --drop_rate 0.2 \
       --embedding ~/data/glove/glove.840B.300d.w2v.vectors.npy \
       --filters 128 \
       --kernel_size 3 \
       --n_classes 2 \
       --name ${dataset}-word-sigm-seqlen-${seqlen} \
       --outfile ${dataset}-word-fgsm-eps-${adv_eps} \
       --samples 16 \
       --seqlen ${seqlen} \
       --fgsm \
       --units 128 \
       --w2v '~/data/glove/glove.840B.300d.w2v'
