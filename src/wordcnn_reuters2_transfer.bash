#!/bin/bash

seqlen=100
data=reuters2
adv=deepfool

for eps in 10 15 18 20 25 27 30 35 40 50; do # deepfool
# for eps in 18; do
    fn=out/reuters2-char-${adv}-eps-${eps}-baseline-c2w-seqlen-${seqlen}.npz
    python eval_wordcnn.py \
           --batch_size 128 \
           --data ${fn} \
           --drop_rate 0.2 \
           --embedding ~/data/glove/glove.840B.300d.w2v.vectors.npy \
           --epochs 5 \
           --filters 128 \
           --kernel_size 3 \
           --n_classes 2 \
           --name ${data}-word-tanh-seqlen-${seqlen} \
           --seqlen ${seqlen} \
           --bipolar \
           --units 128
done
