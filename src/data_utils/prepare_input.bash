#!/bin/bash

datapath=$1
name=$2
seqlen=$3
wordlen=$4
n=$5

labels=$(seq 0 $((${n} - 1)))

function tokenize {
    for pre in train test; do
        for lab in ${labels}; do
            fn=${pre}-${lab}
            python -u 0_tokenize.py \
                   --unescape --cleanup \
                   ${datapath}/${fn}.txt > \
                   ${datapath}/${fn}-tokens.txt
        done
    done
}

function charpad {
    for pre in train test; do
        for lab in ${labels}; do
            fn=${pre}-${lab}
            python -u 1_charpad.py \
                   --seqlen ${seqlen} --wordlen ${wordlen} \
                   --ascii --encode \
                   --sow '{' --eow '}' --eos '+' --pad ' ' --unk '|' \
                   ${datapath}/${fn}-tokens.txt > \
                   ${datapath}/${fn}-seqlen-${seqlen}-wordlen-${wordlen}.txt
        done
    done
}

function charmerge {
    for pre in train test; do
        [ -f ${datapath}/${pre}-char.txt ] && \
            mv ${datapath}/${pre}-char.txt{,.bak}
        for lab in ${labels}; do
            fn=${datapath}/${pre}-${lab}-seqlen-${seqlen}-wordlen-${wordlen}.txt
            sed -e "s/^/${lab} /" \
                ${fn} >> ${datapath}/${pre}-char.txt
        done
    done
}

function char2index {
    python 2_char2index.py \
           --train ${datapath}/train-char.txt \
           --test ${datapath}/test-char.txt \
           --output ${datapath}/${name}-char-seqlen-${seqlen}-wordlen-${wordlen}.npz
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

# tokenize

charpad
charmerge
char2index

wordpad
wordmerge
word2index
