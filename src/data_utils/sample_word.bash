#!/bin/bash

datapath=$1                     # path to the text data
name=$2                         # base name for the output
seqlen=$3                       # maximum sequence length
n_classes=$4                    # number of categories
n_samples=$5                    # number of samples

labels=$(seq 0 $((${n_classes} - 1)))

function sample {
    for lab in ${labels}; do
        fn=test-${lab}
        shuf -n ${n_samples} \
               ${datapath}/${fn}.txt > \
               ${datapath}/${fn}-sample.txt
    done
}

function tokenize {
    for lab in ${labels}; do
        fn=test-${lab}-sample
        python -u 0_tokenize.py \
               --unescape --cleanup \
               ${datapath}/${fn}.txt > \
               ${datapath}/${fn}-tokens.txt
    done
}

function wordpad {
    for lab in ${labels}; do
        fn=test-${lab}-sample
        python -u 1_wordpad.py \
               --seqlen ${seqlen} \
               --pad '<pad>' --eos '<eos>' --unk '<unk>' \
               ${datapath}/${fn}-tokens.txt > \
               ${datapath}/${fn}-seqlen-${seqlen}.txt
    done
}

function wordmerge {
    # If exists, rename it to file.bak, since we are appending to files.
    [ -f ${datapath}/test-sample-word.txt ] && \
        mv ${datapath}/test-sample-word.txt{,.bak}
    for lab in ${labels}; do
        fn=test-${lab}-sample-seqlen-${seqlen}
        sed -e "s/^/${lab} /" ${datapath}/${fn}.txt >> \
            ${datapath}/test-sample-word.txt
    done
}

function word2index {
    out=${datapath}/${name}-word-sample-${n_samples}-seqlen-${seqlen}.npz
    python 2_token2index.py \
           --w2v ~/data/glove/glove.840B.300d.w2v \
           --test ${datapath}/test-sample-word.txt \
           --output ${out}
}

sample
tokenize
wordpad
wordmerge
word2index
