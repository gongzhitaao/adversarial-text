#!/bin/bash

seqlen=100
n_classes=2
adv=deepfool
wordlen=20

for eps in 10 15 18 20 25 27 30 35 40 50 60; do # deepfool
# for eps in 10; do
    name=../out/reuters2-word-${adv}-eps-${eps}-baseline
    ./prepare_w2c.bash ${name} ${seqlen} ${wordlen} ${n_classes}
done
