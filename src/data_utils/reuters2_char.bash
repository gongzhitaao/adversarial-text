#!/bin/bash

datapath=~/data/reuters/reuters2
name=reuters2
seqlen=100
wordlen=20
n_classes=2

./prepare_char.bash ${datapath} ${name} ${seqlen} ${wordlen} ${n_classes}
