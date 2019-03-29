#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -f "data/ldc93s1/ldc93s1.csv" ]; then
    echo "Downloading and preprocessing LDC93S1 example data, saving in ./data/ldc93s1."
    python -u bin/import_ldc93s1.py ./data/ldc93s1
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))')
fi

python -u DeepSpeech.py --noshow_progressbar --noearly_stop \
  --train_files data/ldc93s1/ldc93s1.csv \
  --dev_files data/ldc93s1/ldc93s1.csv \
  --test_files data/ldc93s1/ldc93s1.csv \
  --train_batch_size 1 \
  --dev_batch_size 1 \
  --test_batch_size 1 \
  --n_hidden 100 \
  --epoch 200 \
  --checkpoint_dir "$checkpoint_dir" \
  "$@"
