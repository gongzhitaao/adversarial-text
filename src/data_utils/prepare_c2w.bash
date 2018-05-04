#!/bin/bash

name=$1                         # base name for the output
seqlen=$2                       # maximum sequence length
n_classes=$3                    # number of categories

labels=$(seq 0 $((${n_classes} - 1)))

function charunpad {
    for lab in ${labels}; do
        python -u 1_charpad.py \
               --decode \
               --sow '{' --eow '}' --eos '+' --pad ' ' --unk '|' \
               ${name}-${lab}.txt > ${name}-${lab}-unchar.txt
        echo "Wrote ${name}-${lab}-unchar.txt"
    done
}

function wordpad {
    for lab in ${labels}; do
        python -u 1_wordpad.py \
               --encode \
               --seqlen ${seqlen} \
               --pad '<pad>' --eos '<eos>' --unk '<unk>' \
               ${name}-${lab}-unchar.txt > ${name}-${lab}-pad.txt
        echo "Wrote ${name}-${lab}-pad.txt"
    done
}

function wordmerge {
    fn=${name}-c2w-seqlen-${seqlen}.txt
    [ -f ${fn} ] && mv ${fn}{,.bak}
    for lab in ${labels}; do
        sed -e "s/^/${lab} /" ${name}-${lab}-pad.txt >> ${fn}
    done
    echo "Wrote ${fn}"
}

function word2index {
    python 2_token2index.py \
           --w2v ~/data/glove/glove.840B.300d.w2v \
           --test ${name}-c2w-seqlen-${seqlen}.txt \
           --output ${name}-c2w-seqlen-${seqlen}.npz
}

charunpad
wordpad
wordmerge
word2index
