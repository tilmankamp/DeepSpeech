#!/bin/bash

data="${SHARED_DIR}/data"
fis="${data}/LDC/fisher"
swb="${data}/LDC/LDC97S62/swb"
lbs="${data}/OpenSLR/LibriSpeech/librivox"
cov="${data}/mozilla/CommonVoice/v2.0/en/clips"
alphabet="${SRC_DIR}/data/alphabet.txt"
noise="${data}/UPF/freesound-cc0/"
target_dir="${ML_GROUP_DIR}/ds/training/augmented"

mkdir -p "${target_dir}"
cd "${target_dir}"

print_head() {
    printf "\n$1\n===========================================\n\n"
}

process_lot() {
    target_lot=$1
    shift
    target_set=$1
    shift
    target_prefix="${target_dir}/ds_${target_set}_${target_lot}"
    if [ -f "${target_prefix}.csv" ]; then
        print_head "Skipping ${target_lot} version of ${target_set} set (${target_prefix}.csv already exists)."
    else
        print_head "Generating ${target_lot} version of ${target_set} set..."
        vocoto "$@" write "${target_prefix}"
    fi
}

process_set() {
    target_set=$1
    clean_set="${target_dir}/ds_${target_set}_clean.csv"
    print_head "Processing ${target_set} set..."

    process_lot clean "$@" $LIMIT 2>&1 | sed "s/^/\t/"

    shift
    process_lot noise1 \
        "${target_set}" \
        add "${noise}${target_set}.csv" $LIMIT stash noise \
        add "${clean_set}" shuffle $LIMIT stash crosstalk \
        "$@" $LIMIT \
        shuffle \
        stash remaining \
        slice remaining 80 \
            augment noise -gain -5 \
        push result \
        clear \
        slice remaining 80 \
            augment crosstalk -times 10 -gain -10 \
        push result \
        clear \
        slice remaining 20 \
            compr 4 \
        push result \
        clear \
        add remaining \
        drop remaining \
            rate 8000 \
            rate 16000 \
        push result \
        clear \
        add result 2>&1 | sed "s/^/\t/"

    process_lot noise2 \
        "${target_set}" \
        add "${noise}${target_set}.csv" $LIMIT stash noise \
        add "${clean_set}" shuffle $LIMIT stash crosstalk \
        "$@" $LIMIT \
        shuffle \
        stash remaining \
        slice remaining 80 \
            augment noise -times 2 \
        push result \
        clear \
        slice remaining 80 \
            augment crosstalk -times 5 -gain -5 \
        push result \
        clear \
        slice remaining 20 \
            compr 2 \
        push result \
        clear \
        add remaining \
        drop remaining \
            rate 4000 \
            rate 16000 \
        push result \
        clear \
        add result 2>&1 | sed "s/^/\t/"
}

process_set train \
    add "${fis}-train.csv" \
    add "${swb}-train.csv" \
    add "${lbs}-train-clean-100.csv" \
    add "${lbs}-train-clean-360.csv" \
    add "${lbs}-train-other-500.csv" \
    add "${cov}/train.csv"

process_set dev \
    add "${lbs}-dev-clean.csv" \
    add "${lbs}-dev-other.csv" \
    add "${cov}/dev.csv"

process_set test \
    add "${lbs}-test-clean.csv" \
    add "${lbs}-test-other.csv" \
    add "${cov}/test.csv"
