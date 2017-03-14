#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

python -u DeepSpeech.py \
  --importer ted \
  --train_batch_size 16 \
  --dev_batch_size 8 \
  --ds_test_batch_size 8 \
  --learning_rate 0.0001 \
  "$@"
