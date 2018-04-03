#!/bin/bash

datapath=../out
prefix=reuters2-char-fgsm-eps-0.2
seqlen=100
n_classes=2

./prepare_output.bash ${datapath} ${prefix} ${seqlen} ${n_classes}
