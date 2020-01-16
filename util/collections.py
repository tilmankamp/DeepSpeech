# -*- coding: utf-8 -*-
import os
import csv
import json

from pathlib import Path
from functools import partial
from util.audio import Sample, AUDIO_TYPE_WAV, LOADABLE_FILE_FORMATS

FILE_EXTENSION_CSV = '.csv'
FILE_EXTENSION_SDB = '.sdb'

BIG_ENDIAN = 'big'
INT_SIZE = 4
BIGINT_SIZE = 2 * INT_SIZE
MAGIC = b'SAMPLEDB'
BUFFER_SIZE = 128 * 1024 * 1024

SCHEMA_KEY = 'schema'
CONTENT_KEY = 'content'
MIME_TYPE_KEY = 'mime-type'
MIME_TYPE_TEXT = 'text/plain'
CONTENT_TYPE_SPEECH = 'speech'
CONTENT_TYPE_TRANSCRIPT = 'transcript'

COLUMN_FILENAME = 'wav_filename'
COLUMN_FILESIZE = 'wav_filesize'
COLUMN_TRANSCRIPT = 'transcript'


class CollectionSample(Sample):
    def __init__(self, sample_id, audio_type, raw_data, transcript):
        super().__init__(audio_type, raw_data)
        self.id = sample_id
        self.transcript = transcript


class SDB:
    def __init__(self, sdb_filename, buffering=BUFFER_SIZE):
        self.meta = {}
        self.schema = []
        self.offsets = []
        self.sdb_filename = sdb_filename
        self.sdb_file = open(sdb_filename, 'rb', buffering=buffering)
        magic = self.sdb_file.read(len(MAGIC))
        if magic != MAGIC:
            raise RuntimeError('No Sample Database')
        meta_chunk_len = int.from_bytes(self.sdb_file.read(BIGINT_SIZE), BIG_ENDIAN)
        self.meta = json.loads(self.sdb_file.read(meta_chunk_len))
        if SCHEMA_KEY not in self.meta:
            raise RuntimeError('Missing schema')
        self.schema = self.meta[SCHEMA_KEY]
        self.speech_index = self.find_column(content=CONTENT_TYPE_SPEECH)
        if self.speech_index == -1:
            raise RuntimeError('No speech data (missing in schema)')
        self.audio_type = self.schema[self.speech_index][MIME_TYPE_KEY]
        if self.audio_type not in LOADABLE_FILE_FORMATS:
            raise RuntimeError('Unsupported audio format: {}'.format(self.audio_type))
        self.transcript_index = self.find_column(content=CONTENT_TYPE_TRANSCRIPT)
        if self.transcript_index == -1:
            raise RuntimeError('No transcript data (missing in schema)')
        text_type = self.schema[self.transcript_index][MIME_TYPE_KEY]
        if text_type != MIME_TYPE_TEXT:
            raise RuntimeError('Unsupported text type: {}'.format(text_type))
        sample_chunk_len = int.from_bytes(self.sdb_file.read(BIGINT_SIZE), BIG_ENDIAN)
        self.sdb_file.seek(sample_chunk_len, 1)
        self.sdb_file.read(BIGINT_SIZE)
        num_samples = int.from_bytes(self.sdb_file.read(BIGINT_SIZE), BIG_ENDIAN)
        for _ in range(num_samples):
            self.offsets.append(int.from_bytes(self.sdb_file.read(BIGINT_SIZE), BIG_ENDIAN))

    def find_column(self, content=None, mime_type=None):
        criteria = []
        if content is not None:
            criteria.append((CONTENT_KEY, content))
        if mime_type is not None:
            criteria.append((MIME_TYPE_KEY, mime_type))
        if len(criteria) == 0:
            raise ValueError('At least one of "content" or "mime-type" has to be provided')
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
            raise ValueError('Wrong sample index: {} - has to be between 0 and {}'
                             .format(row_index, len(self.offsets) - 1))
        self.sdb_file.seek(self.offsets[row_index] + INT_SIZE)
        for index in range(len(self.schema)):
            chunk_len = int.from_bytes(self.sdb_file.read(INT_SIZE), BIG_ENDIAN)
            if index in columns:
                column_data[columns.index(index)] = self.sdb_file.read(chunk_len)
                found += 1
                if found == len(columns):
                    return tuple(column_data)
            else:
                self.sdb_file.seek(chunk_len, 1)
        return tuple(column_data)

    def __getitem__(self, i):
        audio_data, transcript = self.read_row(i, self.speech_index, self.transcript_index)
        transcript = transcript.decode()
        sample_id = self.sdb_filename + ':' + str(i)
        return CollectionSample(sample_id, self.audio_type, audio_data, transcript)

    def __iter__(self):
        for i in range(len(self.offsets)):
            yield self[i]

    def __len__(self):
        return len(self.offsets)
    
    def __del__(self):
        if self.sdb_file is not None:
            self.sdb_file.close()


class CSV:
    def __init__(self, csv_filename):
        self.csv_filename = csv_filename
        self.rows = []
        csv_dir = Path(csv_filename).parent
        with open(csv_filename, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                wav_filename = Path(row[COLUMN_FILENAME])
                if not wav_filename.is_absolute():
                    wav_filename = csv_dir / wav_filename
                self.rows.append((str(wav_filename), int(row[COLUMN_FILESIZE]), row[COLUMN_TRANSCRIPT]))
        self.rows.sort(key=lambda r: r[1])

    def __getitem__(self, i):
        wav_filename, _, transcript = self.rows[i]
        with open(wav_filename, 'rb') as wav_file:
            return CollectionSample(wav_filename, AUDIO_TYPE_WAV, wav_file.read(), transcript)

    def __iter__(self):
        for i in range(len(self.rows)):
            yield self[i]

    def __len__(self):
        return len(self.rows)


class Interleaved:
    def __init__(self, *cols):
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


def samples_from_file(filename, buffering=BUFFER_SIZE):
    ext = os.path.splitext(filename)[1].lower()
    if ext == FILE_EXTENSION_SDB:
        return SDB(filename, buffering=buffering)
    if ext == FILE_EXTENSION_CSV:
        return CSV(filename)
    raise ValueError('Unknown file type: "{}"'.format(ext))


def samples_from_files(filenames, buffering=BUFFER_SIZE):
    if len(filenames) == 0:
        raise ValueError('No files')
    if len(filenames) == 1:
        return samples_from_file(filenames[0], buffering=buffering)
    cols = list(map(partial(samples_from_file, buffering=buffering), filenames))
    return Interleaved(*cols)
