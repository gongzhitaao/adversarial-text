#!/bin/bash

seqlen=100
adv_eps=10
data=reuters2

for eps in 10 15 18 20 25 27 30 35 40 50 60; do
# for eps in 10; do               # for debugging
    python wordcnn_deepfool.py \
           --adv_batch_size 64 \
           --adv_epochs 5 \
           --adv_eps ${eps} \
           --batch_size 64 \
           --data ~/data/reuters/${data}/${data}-word-seqlen-${seqlen}.npz \
           --drop_rate 0.2 \
           --embedding ~/data/glove/glove.840B.300d.w2v.vectors.npy \
           --filters 128 \
           --indexer '~/data/glove/glove.840B.300d.annoy' \
           --keepall \
           --kernel_size 3 \
           --n_classes 2 \
           --name ${data}-word-tanh-seqlen-${seqlen} \
           --outfile ${data}-word-deepfool-eps-${eps}-baseline \
           --seqlen ${seqlen} \
           --bipolar \
           --units 128 \
           --w2v '~/data/glove/glove.840B.300d.w2v'
done
