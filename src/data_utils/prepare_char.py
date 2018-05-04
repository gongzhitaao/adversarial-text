#!/bin/bash

datapath=$1                     # path to the text data
name=$2                         # base name for the output
seqlen=$3                       # maximum sequence length
wordlen=$4                      # maximum word length
n_classes=$5                    # number of categories

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

tokenize
charpad
charmerge
char2index
