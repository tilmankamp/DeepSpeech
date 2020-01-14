# -*- coding: utf-8 -*-
import os
import io
import csv
import json
import time

from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool
from collections.abc import Sequence, Iterable

from util.audio import AUDIO_TYPE_WAV, AUDIO_TYPE_OPUS, read_duration, read_audio, pcm_to_np

BIG_ENDIAN = 'big'
INT_SIZE = 4
BIGINT_SIZE = 2 * INT_SIZE
MAGIC = b'SAMPLEDB'
BUFFER_SIZE = 128 * 1024 * 1024


class Sample:
    def __init__(self, sample_id, audio_type, raw_data, transcript):
        self.id = sample_id
        self.audio_type = audio_type
        self.audio_format = None
        self.audio = None
        self.audio_file = io.BytesIO(raw_data)
        self.transcript = transcript
        self.duration = read_duration(audio_type, self.audio_file)

    def load_audio(self):
        self.audio_format, pcm_data = read_audio(self.audio_type, self.audio_file)
        self.audio = pcm_to_np(self.audio_format, pcm_data)
        self.audio_file.close()
        self.audio_file = None


class SDB(Iterable):
    def __init__(self, sdb_filename):
        super().__init__()
        self.sdb_filename = sdb_filename
        with open(sdb_filename, 'rb', buffering=BUFFER_SIZE) as sdb_file:
            magic = sdb_file.read(len(MAGIC))
            if magic != MAGIC:
                raise RuntimeError('No Sample Database')
            meta_chunk_len = int.from_bytes(sdb_file.read(BIGINT_SIZE), BIG_ENDIAN)
            self.meta = json.loads(sdb_file.read(meta_chunk_len))
            if 'schema' not in self.meta:
                raise RuntimeError('Missing schema')
            self.schema = self.meta['schema']
            self.speech_index = self.find_column(content='speech')
            if self.speech_index == -1:
                raise RuntimeError('No speech data (missing in schema)')
            self.transcript_index = self.find_column(content='transcript')
            if self.transcript_index == -1:
                raise RuntimeError('No transcript data (missing in schema)')
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
            raise RuntimeError('Wrong sample index: {} - has to be between 0 and {}'.format(row_index, len(self.offsets)))
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
        sample_id = self.sdb_filename + ':' + str(i)
        return Sample(sample_id, AUDIO_TYPE_OPUS, opus_data, transcript)

    def __iter__(self):
        for i in range(len(self.offsets)):
            yield self[i]

    def __len__(self):
        return len(self.offsets)


class CSV(Iterable):
    def __init__(self, csv_filename):
        super().__init__()
        self.csv_filename = csv_filename
        self.rows = []
        csv_dir = Path(csv_filename).parent
        with open(csv_filename, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                wav_filename = Path(row['wav_filename'])
                if not wav_filename.is_absolute():
                    wav_filename = csv_dir / wav_filename
                self.rows.append((str(wav_filename), int(row['wav_filesize']), row['transcript']))
        self.rows.sort(key=lambda r: r[1])

    def __getitem__(self, i):
        wav_filename, _, transcript = self.rows[i]
        with open(wav_filename, 'rb') as wav_file:
            return Sample(wav_filename, AUDIO_TYPE_WAV, wav_file.read(), transcript)

    def __iter__(self):
        for i in range(len(self.rows)):
            yield self[i]

    def __len__(self):
        return len(self.rows)


class Interleaved(Iterable):
    def __init__(self, *cols):
        super().__init__()
        self.cols = cols

    def __iter__(self):
        firsts = []
        for index, col in enumerate(self.cols):
            it = iter(col)
            try:
                first = next(it)
                firsts.append((index, it, first))
            except StopIteration:
                continue
        while len(firsts) > 0:
            firsts.sort(key=lambda it_first: it_first[2].duration)
            index, it, first = firsts.pop(0)
            yield first
            try:
                first = next(it)
            except StopIteration:
                continue
            firsts.append((index, it, first))

    def __len__(self):
        return sum(map(lambda c: len(c), self.cols))


class LimitingPool:
    def __init__(self, processes=None, limit_factor=2, sleeping_for=0.1):
        self.processes = os.cpu_count() if processes is None else processes
        self.pool = ThreadPool(processes=processes)
        self.sleeping_for = sleeping_for
        self.max_ahead = self.processes * limit_factor
        self.processed = 0

    def __enter__(self):
        return self

    def limit(self, it):
        for obj in it:
            while self.processed >= self.max_ahead:
                time.sleep(self.sleeping_for)
            self.processed += 1
            yield obj

    def map(self, fun, it):
        for obj in self.pool.imap(fun, self.limit(it)):
            self.processed -= 1
            yield obj

    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.close()


def _load_audio(sample):
    sample.load_audio()
    return sample


def prepare_audio(col):
    with LimitingPool() as pool:
        for current_sample in pool.map(_load_audio, col):
            yield current_sample


def _samples_from_file(filename):
    col = None
    suffix = filename.split('.')[-1].lower()
    if suffix == 'sdb':
        col = SDB(filename)
    elif suffix == 'csv':
        col = CSV(filename)
    else:
        raise RuntimeError('Unknown file suffix: ".{}"'.format(suffix))
    return col


def samples_from_file(filename):
    return prepare_audio(_samples_from_file(filename))


def samples_from_files(filenames):
    if len(filenames) == 0:
        raise RuntimeError('No files')
    if len(filenames) == 1:
        return _samples_from_file(filenames[0])
    cols = list(map(lambda filename: _samples_from_file(filename), filenames))
    return prepare_audio(Interleaved(*cols))
