#!/bin/bash

data=reuters2
adv=fgsm
datapath=~/data/reuters/${data}

python wmd.py \
       --origin ${datapath}/test-0-seqlen-100.txt \
       --others ../out/${data}-char-${adv}-eps-*-0.txt \
       --outfile ../out/${data}-char-${adv}-wmd.txt \
       --w2v ~/data/glove/glove.840B.300d.w2v
