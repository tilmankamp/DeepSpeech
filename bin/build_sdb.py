#!/usr/bin/env python
'''
Builds Sample Databases (.sdb files)
Use "python3 build_sdb.py -h" for help
'''
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import json
import argparse
import progressbar

from util.downloader import SIMPLE_BAR
from util.audio import convert_samples, AUDIO_TYPE_OPUS
from util.collections import samples_from_files

BIG_ENDIAN = 'big'
INT_SIZE = 4
BIGINT_SIZE = 2 * INT_SIZE
MAGIC = b'SAMPLEDB'
META = {
    'schema': [
        {'content': 'speech', 'mime-type': 'audio/opus'},
        {'content': 'transcript', 'mime-type': 'text/plain'}
    ]
}


def build_sample_entry(sample):
    def to_bytes(n):
        return n.to_bytes(INT_SIZE, BIG_ENDIAN)
    opus = sample.audio.getbuffer()
    opus_len = to_bytes(len(opus))
    transcript = sample.transcript.encode()
    transcript_len = to_bytes(len(transcript))
    entry_len = to_bytes(len(opus_len) + len(opus) + len(transcript_len) + len(transcript))
    return b''.join([entry_len, opus_len, opus, transcript_len, transcript])


def build_sdb():
    def to_bytes(n):
        return n.to_bytes(BIGINT_SIZE, BIG_ENDIAN)
    samples = samples_from_files(CLI_ARGS.sources)
    offsets = []
    with open(CLI_ARGS.target, 'wb') as sdb_file:
        sdb_file.write(MAGIC)

        print('Writing meta data...')
        meta_data = json.dumps(META).encode()
        sdb_file.write(to_bytes(len(meta_data)))
        sdb_file.write(meta_data)

        print('Writing samples...')
        offset_samples = sdb_file.tell()
        sdb_file.write(to_bytes(0))
        sdb_file.write(to_bytes(len(samples)))
        num_workers = os.cpu_count() * CLI_ARGS.load_factor if CLI_ARGS.load_factor > 0 else 1
        bar = progressbar.ProgressBar(max_value=len(samples), widgets=SIMPLE_BAR)
        for sample in bar(convert_samples(samples, audio_type=AUDIO_TYPE_OPUS, processes=num_workers)):
            buffer = build_sample_entry(sample)
            offsets.append(sdb_file.tell())
            sdb_file.write(buffer)
        offset_index = sdb_file.tell()
        sdb_file.seek(offset_samples)
        sdb_file.write(to_bytes(offset_index - offset_samples - BIGINT_SIZE))

        print('Writing indices...')
        sdb_file.seek(offset_index + BIGINT_SIZE)
        sdb_file.write(to_bytes(len(samples)))
        for offset in offsets:
            sdb_file.write(to_bytes(offset))
        offset_end = sdb_file.tell()
        sdb_file.seek(offset_index)
        sdb_file.write(to_bytes(offset_end - offset_index - BIGINT_SIZE))


def handle_args():
    parser = argparse.ArgumentParser(description='Tool for building Sample Databases (.sdb files) '
                                                 'from DeepSpeech CSV files')
    parser.add_argument('--load_factor', type=int, default=1,
                        help='CPU-multiplier for the number of parallel workers - 0 for 1 worker')
    parser.add_argument('sources', nargs='+', help='Source collections (.csv and Sample DB files)')
    parser.add_argument('target', help='Sample DB to create')
    return parser.parse_args()


if __name__ == "__main__":
    CLI_ARGS = handle_args()
    build_sdb()
