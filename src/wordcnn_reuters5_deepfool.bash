#!/bin/bash

seqlen=100
adv_eps=20
data=reuters5
n_classes=5

python wordcnn_deepfool.py \
       --adv_batch_size 16 \
       --adv_epochs 5 \
       --adv_eps ${adv_eps} \
       --batch_size 64 \
       --data ~/data/reuters/${data}/${data}-word-seqlen-${seqlen}.npz \
       --drop_rate 0.2 \
       --embedding ~/data/glove/glove.840B.300d.w2v.vectors.npy \
       --epochs 5 \
       --filters 128 \
       --kernel_size 3 \
       --n_classes ${n_classes} \
       --name ${data}-word-seqlen-${seqlen} \
       --outfile ${data}-word-deepfool-eps-${adv_eps} \
       --samples 16 \
       --seqlen ${seqlen} \
       --units 128
