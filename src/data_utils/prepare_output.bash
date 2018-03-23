#!/bin/bash

datapath=$1
prefix=$2
seqlen=$3
n=$4

labels=$(seq 0 $((${n} - 1)))

function index2char {
    for lab in ${labels}; do
        fn=${datapath}/${prefix}-${lab}
        python -u 3_index2char.py ${fn}.npy > ${fn}-padded.txt
        echo "wrote ${fn}.txt"
    done
}

function decode {
    for lab in ${labels}; do
        fn=${datapath}/${prefix}-${lab}
        python -u 1_charpad.py \
               --decode \
               --sow '{' --eow '}' --eos '+' --pad ' ' --unk '|' \
               ${fn}-padded.txt > ${fn}.txt
        echo "wrote ${fn}.txt"
    done
}

function wordpad {
    for lab in ${labels}; do
        fn=${datapath}/${prefix}-${lab}
        python -u 1_wordpad.py \
               --seqlen ${seqlen} \
               --pad '<pad>' --eos '<eos>' --unk '<unk>' \
               ${fn}.txt > ${fn}-seqlen-${seqlen}.txt
    done
}

function wordmerge {
    fn=${datapath}/${prefix}
    [ -f ${fn}.txt ] && mv ${fn}.txt{,.bak}
    for lab in ${labels}; do
        sed -e "s/^/${lab} /" \
            ${fn}-${lab}-seqlen-${seqlen}.txt >> ${fn}.txt
    done
}

function word2index {
    fn=${datapath}/${prefix}
    python -u 2_token2index.py \
           --w2v ~/data/glove/glove.840B.300d.w2v \
           --test ${fn}.txt \
           --output ${fn}-seqlen-${seqlen}.npz
}

index2char
decode

wordpad
wordmerge
word2index
