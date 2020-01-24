#!/bin/bash

set -ex

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ -n "$1" ]; then
  lm_dir="$1"
else
  lm_dir="$dir/lm"
fi

if type "pigz" > /dev/null; then
  gzip="pigz"
else
  gzip="gzip"
fi

if type "unpigz" > /dev/null; then
  gunzip="unpigz"
else
  gunzip="gunzip"
fi

if type "pv" > /dev/null; then
  pv="pv"
else
  pv="cat"
fi

unprepared_txt_gz="$lm_dir/upper.txt.gz"
prepared_txt_gz="$lm_dir/lower.txt.gz"
vocab="$lm_dir/vocab.txt"
arpa="$lm_dir/lm.arpa"
filtered_arpa="$lm_dir/lm_filtered.arpa"
lm="$lm_dir/lm.binary"
tmp="$lm_dir/tmp"

mkdir -p "$lm_dir"

if [ ! -f "$unprepared_txt_gz" ]; then
  wget -O "$unprepared_txt_gz" https://traces1.inria.fr/oscar/files/Compressed/de_dedup.txt.gz
fi

if [ ! -f "$vocab" ] || [ ! -f "$prepared_txt_gz" ]; then
  $pv "$unprepared_txt_gz" | $unpigz | python "$dir/prepare_lm_vocab.py" "$vocabular" | $pigz >"$prepared_txt_gz"
fi

lmplz --skip_symbols --order 5 --temp_prefix "$tmp" --memory 25% --text "$prepared_txt_gz" --arpa "$arpa" --prune 0 0 1
"$pv" "$vocabular" | filter single "model:${arpa}" "$filtered_arpa"
build_binary -a 255 -q 8 trie "$filtered_arpa" "$lm"