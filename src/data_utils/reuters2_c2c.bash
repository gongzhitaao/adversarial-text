#!/bin/bash

seqlen=100
n_classes=2
adv=hotflip
wordlen=20

for c in 5 10 15 20 25 30 35 40; do # hotflip
# for eps in 10 15 18 20 25 27 30 35 40 50 60; do # deepfool
# for c in 5; do
    name=../MS_wenlu/reuters2-char-${adv}-c${c}-b3
    ./prepare_c2c.bash ${name} ${seqlen} ${wordlen} ${n_classes}
done
