#!/bin/bash

seqlen=100
adv_eps=20

python wordcnn_deepfool.py \
       --adv_batch_size 64 \
       --adv_epochs 5 \
       --adv_eps 20 \
       --batch_size 64 \
       --data ~/data/reuters/reuters2/reuters2-word-seqlen-${seqlen}.npz \
       --drop_rate 0.2 \
       --embedding ~/data/glove/glove.840B.300d.w2v.vectors.npy \
       --epochs 5 \
       --filters 128 \
       --kernel_size 3 \
       --n_classes 2 \
       --name reuters2-word-tanh-seqlen-${seqlen} \
       --output reuters2-word-deepfool-eps-${adv_eps} \
       --sample_ratio 0.1 \
       --seqlen ${seqlen} \
       --bipolar \
       --units 128
