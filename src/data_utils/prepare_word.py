#!/bin/bash

datapath=$1                     # path to the text data
name=$2                         # base name for the output
seqlen=$3                       # maximum sequence length
n_classes=$4                    # number of categories

labels=$(seq 0 $((${n_classes} - 1)))

function tokenize {
    for pre in train test; do
        for lab in ${labels}; do
            fn=${pre}-${lab}
            [[ ! -f "${datapath}/${fn}-tokens.txt" ]] && \
                python -u 0_tokenize.py \
                       --unescape --cleanup \
                       ${datapath}/${fn}.txt > \
                       ${datapath}/${fn}-tokens.txt
        done
    done
}

function wordpad {
    for pre in train test; do
        for lab in ${labels}; do
            fn=${pre}-${lab}
            python -u 1_wordpad.py \
                   --seqlen ${seqlen} \
                   --pad '<pad>' --eos '<eos>' --unk '<unk>' \
                   ${datapath}/${fn}-tokens.txt > \
                   ${datapath}/${fn}-seqlen-${seqlen}.txt
        done
    done
}

function wordmerge {
    for pre in train test; do
        [ -f ${datapath}/${pre}-word.txt ] && \
            mv ${datapath}/${pre}-word.txt{,.bak}
        for lab in ${labels}; do
            fn=${pre}-${lab}-seqlen-${seqlen}
            sed -e "s/^/${lab} /" ${datapath}/${fn}.txt >> \
                ${datapath}/${pre}-word.txt
        done
    done
}

function word2index {
    python 2_token2index.py \
           --w2v ~/data/glove/glove.840B.300d.w2v \
           --train ${datapath}/train-word.txt --test ${datapath}/test-word.txt \
           --output ${datapath}/${name}-word-seqlen-${seqlen}.npz
}

tokenize
wordpad
wordmerge
word2index
