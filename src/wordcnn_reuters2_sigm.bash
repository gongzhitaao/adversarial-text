#!/bin/bash

seqlen=300

python run_wordcnn.py \
       --batch_size 128 \
       --data ~/data/reuters/reuters2/reuters2-word-seqlen-${seqlen}.npz \
       --drop_rate 0.2 \
       --embedding ~/data/glove/glove.840B.300d.w2v.vectors.npy \
       --epochs 5 \
       --filters 128 \
       --kernel_size 3 \
       --n_classes 2 \
       --name reuters2-word-sigm-seqlen-${seqlen} \
       --seqlen ${seqlen} \
       --unipolar \
       --units 128
