#!/bin/bash

prefix=~/data/imdb
name=imdb
seqlen=300
wordlen=20
n_classes=2

./prepare_input.bash ${prefix} ${name} ${seqlen} ${wordlen} ${n_classes}
