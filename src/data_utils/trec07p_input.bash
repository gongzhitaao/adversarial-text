#!/bin/bash

prefix=~/data/trec07p
name=trec07p
seqlen=50
wordlen=5
n_classes=2

./prepare_input.bash ${prefix} ${name} ${seqlen} ${wordlen} ${n_classes}
