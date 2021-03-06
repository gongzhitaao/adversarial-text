#!/bin/bash

seqlen=100
n_classes=2
adv=hotflip
beam_width=3

# for eps in 10 15 18 20 25 27 30 35 40 50; do # deepfool
for c in 5 10 15 20 25 30 35 40; do
    name=../out/reuters2-char-${adv}-c${c}-b${beam_width}
    ./prepare_c2w.bash ${name} ${seqlen} ${n_classes}
done
