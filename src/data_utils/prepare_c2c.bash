#!/bin/bash

name=$1                         # base name for the output
seqlen=$2                       # maximum sequence length
wordlen=$3                      # maximum word length
n_classes=$4                    # number of categories

labels=$(seq 0 $((${n_classes} - 1)))

function charpad {
    for lab in ${labels}; do
        python -u 1_charpad.py \
               --seqlen ${seqlen} --wordlen ${wordlen} \
               --ascii --encode \
               --sow '{' --eow '}' --eos '+' --pad ' ' --unk '|' \
               ${name}-${lab}.txt > \
               ${name}-${lab}-c2c-tmp.txt
    done
}

function charmerge {
    [ -f ${name}-c2c.txt ] && mv ${name}-c2c.txt{,.bak}
    for lab in ${labels}; do
        sed -e "s/^/${lab} /" \
            ${name}-${lab}-c2c-tmp.txt \
            >> ${name}-c2c.txt
    done
}

function char2index {
    python 2_char2index.py \
           --test ${name}-c2c.txt \
           --output ${name}-c2c-seqlen-${seqlen}-wordlen-${wordlen}.npz
}

charpad
charmerge
char2index
