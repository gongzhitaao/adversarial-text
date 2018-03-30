#!/bin/bash

prefix=~/data/reuters/reuters5
name=reuters5
seqlen=100
wordlen=20
n_classes=5

./prepare_input.bash ${prefix} ${name} ${seqlen} ${wordlen} ${n_classes}
