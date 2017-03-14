#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

python -u DeepSpeech.py \
  --importer LDC97S62 \
  --train_batch_size 48 \
  --dev_batch_size 32 \
  --ds_test_batch_size 32 \
  "$@"
