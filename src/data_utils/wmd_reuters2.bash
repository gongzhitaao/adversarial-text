#!/bin/bash

data=reuters2
adv=deepfool
datapath=~/data/reuters/${data}

python wmd.py \
       --origin ${datapath}/test-0-sample-seqlen-100.txt \
       --others ../out/${data}-word-${adv}-eps-*-0.txt \
       --outfile ../out/${data}-word-${adv}-wmd.txt \
       --w2v ~/data/glove/glove.840B.300d.w2v
