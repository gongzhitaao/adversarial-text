#!/bin/bash

prefix=~/data/fakenews
name=fakenews
seqlen=50
wordlen=5
n_classes=2

./prepare_input.bash ${prefix} ${name} ${seqlen} ${wordlen} ${n_classes}
