# -*- coding: utf-8 -*-
import os
import csv
import json

from pathlib import Path
from functools import partial
from util.helpers import MEGABYTE, GIGABYTE
from util.audio import Sample, AUDIO_TYPE_WAV, AUDIO_TYPE_OPUS, LOADABLE_FILE_FORMATS

FILE_EXTENSION_CSV = '.csv'
FILE_EXTENSION_SDB = '.sdb'

BIG_ENDIAN = 'big'
INT_SIZE = 4
BIGINT_SIZE = 2 * INT_SIZE
MAGIC = b'SAMPLEDB'

BUFFER_SIZE = 1 * MEGABYTE
CACHE_SIZE = 1 * GIGABYTE

SCHEMA_KEY = 'schema'
CONTENT_KEY = 'content'
MIME_TYPE_KEY = 'mime-type'
MIME_TYPE_TEXT = 'text/plain'
CONTENT_TYPE_SPEECH = 'speech'
CONTENT_TYPE_TRANSCRIPT = 'transcript'
META = {
    SCHEMA_KEY: [
        {CONTENT_KEY: CONTENT_TYPE_SPEECH, MIME_TYPE_KEY: AUDIO_TYPE_OPUS},
        {CONTENT_KEY: CONTENT_TYPE_TRANSCRIPT, MIME_TYPE_KEY: MIME_TYPE_TEXT}
    ]
}

COLUMN_FILENAME = 'wav_filename'
COLUMN_FILESIZE = 'wav_filesize'
COLUMN_TRANSCRIPT = 'transcript'


class CollectionSample(Sample):
    def __init__(self, sample_id, audio_type, raw_data, transcript):
        super().__init__(audio_type, raw_data)
        self.id = sample_id
        self.transcript = transcript


class DirectSDBWriter:
    def __init__(self, sdb_filename, buffering=BUFFER_SIZE):
        self.sdb_filename = sdb_filename
        self.sdb_file = open(sdb_filename, 'wb', buffering=buffering)
        self.offsets = []
        self.num_samples = 0

        self.sdb_file.write(MAGIC)

        meta_data = json.dumps(META).encode()
        self.write_big_int(len(meta_data))
        self.sdb_file.write(meta_data)

        self.offset_samples = self.sdb_file.tell()
        self.sdb_file.seek(2 * BIGINT_SIZE, 1)

    def write_int(self, n):
        return self.sdb_file.write(n.to_bytes(INT_SIZE, BIG_ENDIAN))

    def write_big_int(self, n):
        return self.sdb_file.write(n.to_bytes(BIGINT_SIZE, BIG_ENDIAN))

    def __enter__(self):
        return self

    def add(self, sample):
        def to_bytes(n):
            return n.to_bytes(INT_SIZE, BIG_ENDIAN)
        sample.convert(AUDIO_TYPE_OPUS)
        opus = sample.audio.getbuffer()
        opus_len = to_bytes(len(opus))
        transcript = sample.transcript.encode()
        transcript_len = to_bytes(len(transcript))
        entry_len = to_bytes(len(opus_len) + len(opus) + len(transcript_len) + len(transcript))
        buffer = b''.join([entry_len, opus_len, opus, transcript_len, transcript])
        self.offsets.append(self.sdb_file.tell())
        self.sdb_file.write(buffer)
        self.num_samples += 1

    def close(self):
        if self.sdb_file is None:
            return
        offset_index = self.sdb_file.tell()
        self.sdb_file.seek(self.offset_samples)
        self.write_big_int(offset_index - self.offset_samples - BIGINT_SIZE)
        self.write_big_int(self.num_samples)

        self.sdb_file.seek(offset_index + BIGINT_SIZE)
        self.write_big_int(self.num_samples)
        for offset in self.offsets:
            self.write_big_int(offset)
        offset_end = self.sdb_file.tell()
        self.sdb_file.seek(offset_index)
        self.write_big_int(offset_end - offset_index - BIGINT_SIZE)
        self.sdb_file.close()

    def __len__(self):
        return len(self.offsets)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SortingSDBWriter:  # pylint: disable=too-many-instance-attributes
    def __init__(self, sdb_filename, tmp_sdb_filename=None, cache_size=CACHE_SIZE, buffering=BUFFER_SIZE):
        self.sdb_filename = sdb_filename
        self.buffering = buffering
        self.tmp_sdb_filename = (sdb_filename + '.tmp') if tmp_sdb_filename is None else tmp_sdb_filename
        self.tmp_sdb = DirectSDBWriter(self.tmp_sdb_filename, buffering=buffering)
        self.cache_size = cache_size
        self.buckets = []
        self.bucket = []
        self.bucket_offset = 0
        self.bucket_size = 0
        self.overall_size = 0

    def __enter__(self):
        return self

    def finish_bucket(self):
        if len(self.bucket) == 0:
            return
        self.bucket.sort(key=lambda s: s.duration)
        for sample in self.bucket:
            self.tmp_sdb.add(sample)
        self.buckets.append((self.bucket_offset, len(self.bucket)))
        self.bucket_offset += len(self.bucket)
        self.bucket = []
        self.overall_size += self.bucket_size
        self.bucket_size = 0

    def add(self, sample):
        sample.convert(AUDIO_TYPE_OPUS)
        self.bucket.append(sample)
        self.bucket_size += len(sample.audio.getbuffer())
        if self.bucket_size > self.cache_size:
            self.finish_bucket()

    def close(self):
        if self.tmp_sdb is None:
            return
        self.finish_bucket()
        num_samples = len(self.tmp_sdb)
        self.tmp_sdb.close()
        avg_sample_size = self.overall_size / num_samples
        max_cached_samples = self.cache_size / avg_sample_size
        buffer_size = max(1, int(max_cached_samples / len(self.buckets)))
        sdb_reader = SDB(self.tmp_sdb_filename, buffering=self.buffering)

        def buffered_view(start, end):
            buffer = []
            current_offset = start
            while current_offset < end:
                while len(buffer) < buffer_size and current_offset < end:
                    buffer.insert(0, sdb_reader[current_offset])
                    current_offset += 1
                while len(buffer) > 0:
                    yield buffer.pop(-1)

        bucket_views = list(map(lambda b: buffered_view(b[0], b[0] + b[1]), self.buckets))
        interleaved = Interleaved(*bucket_views)
        with DirectSDBWriter(self.sdb_filename, buffering=self.buffering) as sdb_writer:
            for sample in interleaved:
                sdb_writer.add(sample)
        os.unlink(self.tmp_sdb_filename)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SDB:
    def __init__(self, sdb_filename, buffering=BUFFER_SIZE):
        self.meta = {}
        self.schema = []
        self.offsets = []
        self.sdb_filename = sdb_filename
        self.sdb_file = open(sdb_filename, 'rb', buffering=buffering)
        if self.sdb_file.read(len(MAGIC)) != MAGIC:
            raise RuntimeError('No Sample Database')
        meta_chunk_len = self.read_big_int()
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
        sample_chunk_len = self.read_big_int()
        self.sdb_file.seek(sample_chunk_len + BIGINT_SIZE, 1)
        num_samples = self.read_big_int()
        for _ in range(num_samples):
            self.offsets.append(self.read_big_int())

    def read_int(self):
        return int.from_bytes(self.sdb_file.read(INT_SIZE), BIG_ENDIAN)

    def read_big_int(self):
        return int.from_bytes(self.sdb_file.read(BIGINT_SIZE), BIG_ENDIAN)

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
            chunk_len = self.read_int()
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

    def close(self):
        if self.sdb_file is not None:
            self.sdb_file.close()

    def __del__(self):
        self.close()


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
            try:
                it = iter(col)
            except TypeError:
                it = col
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
