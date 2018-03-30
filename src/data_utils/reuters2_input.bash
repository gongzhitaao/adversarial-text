#!/bin/bash

prefix=~/data/reuters/reuters2
name=reuters2
seqlen=100
wordlen=20
n_classes=2

./prepare_input.bash ${prefix} ${name} ${seqlen} ${wordlen} ${n_classes}
