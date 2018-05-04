#!/bin/bash

name=$1                         # base name for the output
seqlen=$2                       # maximum sequence length
n_classes=$3                    # number of categories

labels=$(seq 0 $((${n_classes} - 1)))

function wordunpad {
    for lab in ${labels}; do
        python -u 1_wordpad.py \
               --decode \
               --seqlen ${seqlen} \
               --pad '<pad>' --eos '<eos>' --unk '<unk>' \
               ${name}-${lab}.txt > ${name}-${lab}-unpad.txt
    done
}

function wordpad {
    for lab in ${labels}; do
        python -u 1_wordpad.py \
               --encode \
               --seqlen ${seqlen} \
               --pad '<pad>' --eos '<eos>' --unk '<unk>' \
               ${name}-${lab}-unpad.txt > ${name}-${lab}-pad.txt
    done
}

function wordmerge {
    fn=${name}-w2w-seqlen-${seqlen}.txt
    [ -f ${fn} ] && mv ${fn}{,.bak}
    for lab in ${labels}; do
        sed -e "s/^/${lab} /" ${name}-${lab}-pad.txt >> ${fn}
    done
}

function word2index {
    python 2_token2index.py \
           --w2v ~/data/glove/glove.840B.300d.w2v \
           --test ${name}-w2w-seqlen-${seqlen}.txt \
           --output ${name}-w2w-seqlen-${seqlen}.npz
}

wordunpad
wordpad
wordmerge
word2index
