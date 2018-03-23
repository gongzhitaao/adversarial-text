#!/bin/bash

datapath=../out
prefix=imdb_char_hotflip
seqlen=300
n_classes=2

./prepare_output.bash ${datapath} ${prefix} ${seqlen} ${n_classes}
