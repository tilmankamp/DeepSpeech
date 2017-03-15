#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

python -u DeepSpeech.py \
  --importer ldc93s1 \
  --train_batch_size 1 \
  --dev_batch_size 1 \
  --ds_test_batch_size 1 \
  "$@"
