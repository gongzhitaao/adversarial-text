#!/bin/bash

prefix=~/data/reuters/reuters5
name=reuters5
seqlen=300
wordlen=5
n_classes=5

./prepare_input.bash ${prefix} ${name} ${seqlen} ${wordlen} ${n_classes}
