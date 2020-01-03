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

import csv
import json
import wave
import argparse
import progressbar

from pathlib import Path
from multiprocessing.pool import Pool
from util.downloader import SIMPLE_BAR
from util.audio import encode_opus, get_audio_format

BIG_ENDIAN = 'big'

INT_SIZE = 4
BIGINT_SIZE = 2 * INT_SIZE
MAGIC = b'SAMPLEDB'
LEN_OFFSET = len(MAGIC)
INDEX_OFFSET_OFFSET = LEN_OFFSET + BIGINT_SIZE
FIRST_ENTRY_OFFSET = INDEX_OFFSET_OFFSET + BIGINT_SIZE
META = {
    'schema': [
        {
            'content': 'speech',
            'mime-type': 'audio/opus'
        },
        {
            'content': 'transcript',
            'mime-type': 'text/plain'
        }
    ]
}

class Sample:
    def __init__(self, wav_filename, wav_filesize, transcript):
        self.wav_filename = wav_filename
        self.wav_filesize = wav_filesize
        self.transcript = transcript


def read_csvs():
    samples = []
    for csv_path in CLI_ARGS.csvs:
        csv_dir = Path(csv_path).parent
        print('Reading "{}"...'.format(csv_path))
        with open(csv_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                wav_filename = Path(row['wav_filename'])
                if not wav_filename.is_absolute():
                    wav_filename = csv_dir / wav_filename
                sample = Sample(str(wav_filename), row['wav_filesize'], row['transcript'])
                samples.append(sample)
    print('Sorting samples...')
    samples.sort(key=lambda s: s.wav_filesize)
    return samples


def build_sample_entry(sample):
    with wave.open(sample.wav_filename, 'r') as wav_file:
        audio_format = get_audio_format(wav_file)
        pcm_data = wav_file.readframes(wav_file.getnframes())
        opus = encode_opus(audio_format, pcm_data)
    opus_len = len(opus).to_bytes(INT_SIZE, BIG_ENDIAN)
    transcript = sample.transcript.encode()
    transcript_len = len(transcript).to_bytes(INT_SIZE, BIG_ENDIAN)
    entry_len = (len(opus_len) + len(opus) + len(transcript_len) + len(transcript)).to_bytes(INT_SIZE, BIG_ENDIAN)
    return b''.join([entry_len, opus_len, opus, transcript_len, transcript])


def build_sdb():
    samples = read_csvs()
    offsets = []
    with open(CLI_ARGS.target, 'wb') as sdb_file:
        sdb_file.write(MAGIC)

        print('Writing meta data...')
        meta_data = json.dumps(META).encode()
        sdb_file.write(len(meta_data).to_bytes(BIGINT_SIZE, BIG_ENDIAN))
        sdb_file.write(meta_data)

        print('Writing samples...')
        offset_samples = sdb_file.tell()
        sdb_file.write((0).to_bytes(BIGINT_SIZE, BIG_ENDIAN))
        sdb_file.write(len(samples).to_bytes(BIGINT_SIZE, BIG_ENDIAN))
        with Pool() as pool:
            bar = progressbar.ProgressBar(max_value=len(samples), widgets=SIMPLE_BAR)
            for buffer in bar(pool.imap(build_sample_entry, samples)):
                offsets.append(sdb_file.tell())
                sdb_file.write(buffer)
        offset_index = sdb_file.tell()
        sdb_file.seek(offset_samples)
        sdb_file.write((offset_index - offset_samples - BIGINT_SIZE).to_bytes(BIGINT_SIZE, BIG_ENDIAN))

        print('Writing indices...')
        sdb_file.seek(offset_index)
        sdb_file.write((0).to_bytes(BIGINT_SIZE, BIG_ENDIAN))
        sdb_file.write(len(samples).to_bytes(BIGINT_SIZE, BIG_ENDIAN))
        for offset in offsets:
            sdb_file.write(offset.to_bytes(BIGINT_SIZE, BIG_ENDIAN))
        offset_end = sdb_file.tell()
        sdb_file.seek(offset_index)
        sdb_file.write((offset_end - offset_index - BIGINT_SIZE).to_bytes(BIGINT_SIZE, BIG_ENDIAN))


def handle_args():
    parser = argparse.ArgumentParser(description='Tool for building Sample Databases (.sdb files) '
                                                 'from DeepSpeech CSV files')
    parser.add_argument('csvs', nargs='+', help='CSV files')
    parser.add_argument('target', help='Sample DB to create')
    return parser.parse_args()


if __name__ == "__main__":
    CLI_ARGS = handle_args()
    build_sdb()
