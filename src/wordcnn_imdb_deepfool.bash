#!/bin/bash

seqlen=300
adv_eps=20
data=imdb

python wordcnn_deepfool.py \
       --adv_batch_size 16 \
       --adv_epochs 5 \
       --adv_eps ${adv_eps} \
       --batch_size 64 \
       --data ~/data/${data}/${data}-word-seqlen-${seqlen}.npz \
       --drop_rate 0.2 \
       --embedding ~/data/glove/glove.840B.300d.w2v.vectors.npy \
       --epochs 5 \
       --filters 128 \
       --kernel_size 3 \
       --n_classes 2 \
       --name ${data}-word-tanh-seqlen-${seqlen} \
       --outfile ${data}-word-deepfool-eps-${adv_eps} \
       --samples 16 \
       --seqlen ${seqlen} \
       --bipolar \
       --units 128
