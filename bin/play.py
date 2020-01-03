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
import random
import argparse
import simpleaudio

from pathlib import Path
from collections.abc import Sequence
from util.audio import decode_opus, get_audio_format

BIG_ENDIAN = 'big'
INT_SIZE = 4
BIGINT_SIZE = 2 * INT_SIZE
MAGIC = b'SAMPLEDB'


def fail(message):
    print(message)
    sys.exit(1)


class Sample:
    def __init__(self, audio_format, pcm_data, transcript):
        self.audio_format = audio_format
        self.pcm_data = pcm_data
        self.transcript = transcript

    def play(self):
        rate, channels, width = self.audio_format
        wave_obj = simpleaudio.WaveObject(self.pcm_data, channels, width, rate)
        play_obj = wave_obj.play()
        play_obj.wait_done()


class SDB(Sequence):
    def __init__(self, sdb_filename):
        super().__init__()
        self.sdb_filename = sdb_filename
        with open(sdb_filename, 'rb') as sdb_file:
            magic = sdb_file.read(len(MAGIC))
            if magic != MAGIC:
                fail('No Sample Database')
            meta_chunk_len = int.from_bytes(sdb_file.read(BIGINT_SIZE), BIG_ENDIAN)
            self.meta = json.loads(sdb_file.read(meta_chunk_len))
            if 'schema' not in self.meta:
                fail('Missing schema')
            self.schema = self.meta['schema']
            self.speech_index = self.find_column(content='speech')
            if self.speech_index == -1:
                fail('No speech data (missing in schema)')
            self.transcript_index = self.find_column(content='transcript')
            if self.transcript_index == -1:
                fail('No transcript data (missing in schema)')
            sample_chunk_len = int.from_bytes(sdb_file.read(BIGINT_SIZE), BIG_ENDIAN)
            sdb_file.seek(sample_chunk_len, 1)
            sdb_file.read(BIGINT_SIZE)
            num_samples = int.from_bytes(sdb_file.read(BIGINT_SIZE), BIG_ENDIAN)
            self.offsets = []
            for _ in range(num_samples):
                self.offsets.append(int.from_bytes(sdb_file.read(BIGINT_SIZE), BIG_ENDIAN))

    def find_column(self, content=None, mime_type=None):
        criteria = []
        if content is not None:
            criteria.append(('content', content))
        if mime_type is not None:
            criteria.append(('mime-type', mime_type))
        if len(criteria) == 0:
            raise RuntimeError('At least one of "content" or "mime-type" has to be provided')
        for index, column in enumerate(self.schema):
            matched = 0
            for field, value in criteria:
                if column[field] == value:
                    matched += 1
            if matched == len(criteria):
                return index
        return -1

    def read_row(self, row_index, *columns):
        columns = list(columns)
        column_data = [None] * len(columns)
        found = 0
        if not 0 <= row_index < len(self.offsets):
            fail('Wrong sample index')
        with open(self.sdb_filename, 'rb') as sdb_file:
            sdb_file.seek(self.offsets[row_index] + INT_SIZE)
            for index in range(len(self.schema)):
                chunk_len = int.from_bytes(sdb_file.read(INT_SIZE), BIG_ENDIAN)
                if index in columns:
                    column_data[columns.index(index)] = sdb_file.read(chunk_len)
                    found += 1
                    if found == len(columns):
                        return tuple(column_data)
                else:
                    sdb_file.seek(chunk_len, 1)
        return tuple(column_data)

    def __getitem__(self, i):
        opus_data, transcript = self.read_row(i, self.speech_index, self.transcript_index)
        transcript = transcript.decode()
        audio_format, pcm_data = decode_opus(opus_data)
        return Sample(audio_format, pcm_data, transcript)

    def __len__(self):
        return len(self.offsets)


class CSV(Sequence):
    def __init__(self, csv_filename):
        super().__init__()
        self.rows = []
        csv_dir = Path(csv_filename).parent
        with open(csv_filename, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                wav_filename = Path(row['wav_filename'])
                if not wav_filename.is_absolute():
                    wav_filename = csv_dir / wav_filename
                self.rows.append((str(wav_filename), row['transcript']))

    def __getitem__(self, i):
        wav_filename, transcript = self.rows[i]
        with wave.open(wav_filename, 'r') as wav_file:
            audio_format = get_audio_format(wav_file)
            num_samples = wav_file.getnframes()
            pcm_data = wav_file.readframes(num_samples)
        return Sample(audio_format, pcm_data, transcript)

    def __len__(self):
        return len(self.rows)


def play_db():
    db = None
    suffix = CLI_ARGS.db.split('.')[-1].lower()
    if suffix == 'sdb':
        db = SDB(CLI_ARGS.db)
    elif suffix == 'csv':
        db = CSV(CLI_ARGS.db)
    else:
        fail('Unknown file suffix: ".{}"'.format(suffix))
    while True:
        index = random.randrange(0, len(db))
        sample = db[index]
        print('Sample index: {} - transcript: "{}"'.format(index, sample.transcript))
        sample.play()


def handle_args():
    parser = argparse.ArgumentParser(description='Tool for playing samples from Sample Databases (.sdb files) '
                                                 'and DeepSpeech .csv files')
    parser.add_argument('db', help='Sample DB or CSV file to play samples from')
    return parser.parse_args()


if __name__ == "__main__":
    CLI_ARGS = handle_args()
    play_db()
