#!/bin/bash

datapath=~/data/reuters/reuters2
name=reuters2
seqlen=100
n_classes=2

./prepare_word.bash ${datapath} ${name} ${seqlen} ${n_classes}
