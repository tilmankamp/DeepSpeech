# -*- coding: utf-8 -*-
import os
import re
import csv
import math
import json
import random
import numpy as np

from pathlib import Path
from functools import partial
from multiprocessing import Queue, Process
from .helpers import MEGABYTE, GIGABYTE, Interleaved, LimitingPool
from .audio import Sample, DEFAULT_FORMAT, AUDIO_TYPE_WAV, AUDIO_TYPE_OPUS, AUDIO_TYPE_NP, SERIALIZABLE_AUDIO_TYPES

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

OVERLAY_PARSER = re.compile(r'^([^\[]*)(\[([^\[]*)\])?$')
OVERLAY_QUEUE_SIZE = 16


class Overlay:
    def __init__(self,
                 sample_source,
                 snr_db_min=3.0,
                 snr_db_max=30.0,
                 n_layers_min=5,
                 n_layers_max=5,
                 p_factor=1.0):
        self.sample_source = sample_source
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max
        self.n_layers_min = n_layers_min
        self.n_layers_max = n_layers_max
        self.p_factor = p_factor


class LabeledSample(Sample):
    """In-memory labeled audio sample representing an utterance.
    Derived from util.audio.Sample and used by sample collection readers and writers."""
    def __init__(self, audio_type, raw_data, transcript, audio_format=DEFAULT_FORMAT, sample_id=None):
        """
        Parameters
        ----------
        audio_type : str
            See util.audio.Sample.__init__ .
        raw_data : binary
            See util.audio.Sample.__init__ .
        transcript : str
            Transcript of the sample's utterance
        audio_format : tuple
            See util.audio.Sample.__init__ .
        sample_id : str
            Tracking ID - should indicate sample's origin as precisely as possible.
            It is typically assigned by collection readers.
        """
        super().__init__(audio_type, raw_data, audio_format=audio_format, sample_id=sample_id)
        self.transcript = transcript


class DirectSDBWriter:
    """Sample collection writer for creating a Sample DB (SDB) file"""
    def __init__(self, sdb_filename, buffering=BUFFER_SIZE, audio_type=AUDIO_TYPE_OPUS, id_prefix=None, labeled=True):
        """
        Parameters
        ----------
        sdb_filename : str
            Path to the SDB file to write
        buffering : int
            Write-buffer size to use while writing the SDB file
        audio_type : str
            See util.audio.Sample.__init__ .
        id_prefix : str
            Prefix for IDs of written samples - defaults to sdb_filename
        labeled : bool or None
            If True: Writes labeled samples (util.sample_collections.LabeledSample) only.
            If False: Ignores transcripts (if available) and writes (unlabeled) util.audio.Sample instances.
        """
        self.sdb_filename = sdb_filename
        self.id_prefix = sdb_filename if id_prefix is None else id_prefix
        self.labeled = labeled
        if audio_type not in SERIALIZABLE_AUDIO_TYPES:
            raise ValueError('Audio type "{}" not supported'.format(audio_type))
        self.audio_type = audio_type
        self.sdb_file = open(sdb_filename, 'wb', buffering=buffering)
        self.offsets = []
        self.num_samples = 0

        self.sdb_file.write(MAGIC)

        schema_entries = [{CONTENT_KEY: CONTENT_TYPE_SPEECH, MIME_TYPE_KEY: audio_type}]
        if self.labeled:
            schema_entries.append({CONTENT_KEY: CONTENT_TYPE_TRANSCRIPT, MIME_TYPE_KEY: MIME_TYPE_TEXT})
        meta_data = {SCHEMA_KEY: schema_entries}
        meta_data = json.dumps(meta_data).encode()
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
        sample.change_audio_type(self.audio_type)
        opus = sample.audio.getbuffer()
        opus_len = to_bytes(len(opus))
        if self.labeled:
            transcript = sample.transcript.encode()
            transcript_len = to_bytes(len(transcript))
            entry_len = to_bytes(len(opus_len) + len(opus) + len(transcript_len) + len(transcript))
            buffer = b''.join([entry_len, opus_len, opus, transcript_len, transcript])
        else:
            entry_len = to_bytes(len(opus_len) + len(opus))
            buffer = b''.join([entry_len, opus_len, opus])
        self.offsets.append(self.sdb_file.tell())
        self.sdb_file.write(buffer)
        sample.sample_id = '{}:{}'.format(self.id_prefix, self.num_samples)
        self.num_samples += 1
        return sample.sample_id

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
        self.sdb_file = None

    def __len__(self):
        return len(self.offsets)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SDB:  # pylint: disable=too-many-instance-attributes
    """Sample collection reader for reading a Sample DB (SDB) file"""
    def __init__(self, sdb_filename, buffering=BUFFER_SIZE, id_prefix=None, labeled=True):
        """
        Parameters
        ----------
        sdb_filename : str
            Path to the SDB file to read samples from
        buffering : int
            Read-buffer size to use while reading the SDB file
        id_prefix : str
            Prefix for IDs of read samples - defaults to sdb_filename
        labeled : bool or None
            If True: Reads util.sample_collections.LabeledSample instances. Fails, if SDB file provides no transcripts.
            If False: Ignores transcripts (if available) and reads (unlabeled) util.audio.Sample instances.
            If None: Automatically determines if SDB schema has transcripts
            (reading util.sample_collections.LabeledSample instances) or not (reading util.audio.Sample instances).
        """
        self.sdb_filename = sdb_filename
        self.id_prefix = sdb_filename if id_prefix is None else id_prefix
        self.sdb_file = open(sdb_filename, 'rb', buffering=buffering)
        self.offsets = []
        if self.sdb_file.read(len(MAGIC)) != MAGIC:
            raise RuntimeError('No Sample Database')
        meta_chunk_len = self.read_big_int()
        self.meta = json.loads(self.sdb_file.read(meta_chunk_len).decode())
        if SCHEMA_KEY not in self.meta:
            raise RuntimeError('Missing schema')
        self.schema = self.meta[SCHEMA_KEY]

        speech_columns = self.find_columns(content=CONTENT_TYPE_SPEECH, mime_type=SERIALIZABLE_AUDIO_TYPES)
        if not speech_columns:
            raise RuntimeError('No speech data (missing in schema)')
        self.speech_index = speech_columns[0]
        self.audio_type = self.schema[self.speech_index][MIME_TYPE_KEY]

        self.transcript_index = None
        if labeled is not False:
            transcript_columns = self.find_columns(content=CONTENT_TYPE_TRANSCRIPT, mime_type=MIME_TYPE_TEXT)
            if transcript_columns:
                self.transcript_index = transcript_columns[0]
            else:
                if labeled is True:
                    raise RuntimeError('No transcript data (missing in schema)')

        sample_chunk_len = self.read_big_int()
        self.sdb_file.seek(sample_chunk_len + BIGINT_SIZE, 1)
        num_samples = self.read_big_int()
        for _ in range(num_samples):
            self.offsets.append(self.read_big_int())

    def read_int(self):
        return int.from_bytes(self.sdb_file.read(INT_SIZE), BIG_ENDIAN)

    def read_big_int(self):
        return int.from_bytes(self.sdb_file.read(BIGINT_SIZE), BIG_ENDIAN)

    def find_columns(self, content=None, mime_type=None):
        criteria = []
        if content is not None:
            criteria.append((CONTENT_KEY, content))
        if mime_type is not None:
            criteria.append((MIME_TYPE_KEY, mime_type))
        if len(criteria) == 0:
            raise ValueError('At least one of "content" or "mime-type" has to be provided')
        matches = []
        for index, column in enumerate(self.schema):
            matched = 0
            for field, value in criteria:
                if column[field] == value or (isinstance(value, list) and column[field] in value):
                    matched += 1
            if matched == len(criteria):
                matches.append(index)
        return matches

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
        sample_id = '{}:{}'.format(self.id_prefix, i)
        if self.transcript_index is None:
            [audio_data] = self.read_row(i, self.speech_index)
            return Sample(self.audio_type, audio_data, sample_id=sample_id)
        audio_data, transcript = self.read_row(i, self.speech_index, self.transcript_index)
        transcript = transcript.decode()
        return LabeledSample(self.audio_type, audio_data, transcript, sample_id=sample_id)

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
    """Sample collection reader for reading a DeepSpeech CSV file
    Automatically orders samples by CSV column wav_filesize (if available)."""
    def __init__(self, csv_filename, labeled=None):
        """
        Parameters
        ----------
        csv_filename : str
            Path to the CSV file containing sample audio paths and transcripts
        labeled : bool or None
            If True: Reads LabeledSample instances. Fails, if CSV file has no transcript column.
            If False: Ignores transcripts (if available) and reads (unlabeled) util.audio.Sample instances.
            If None: Automatically determines if CSV file has a transcript column
            (reading util.sample_collections.LabeledSample instances) or not (reading util.audio.Sample instances).
        """
        self.csv_filename = csv_filename
        self.labeled = labeled
        self.rows = []
        csv_dir = Path(csv_filename).parent
        with open(csv_filename, 'r', encoding='utf8') as csv_file:
            reader = csv.DictReader(csv_file)
            if 'transcript' in reader.fieldnames:
                if self.labeled is None:
                    self.labeled = True
            elif self.labeled:
                raise RuntimeError('No transcript data (missing CSV column)')
            for row in reader:
                wav_filename = Path(row['wav_filename'])
                if not wav_filename.is_absolute():
                    wav_filename = csv_dir / wav_filename
                wav_filename = str(wav_filename)
                wav_filesize = int(row['wav_filesize']) if 'wav_filesize' in row else 0
                if self.labeled:
                    self.rows.append((wav_filename, wav_filesize, row['transcript']))
                else:
                    self.rows.append((wav_filename, wav_filesize))
        self.rows.sort(key=lambda r: r[1])

    def __getitem__(self, i):
        row = self.rows[i]
        wav_filename = row[0]
        with open(wav_filename, 'rb') as wav_file:
            if self.labeled:
                return LabeledSample(AUDIO_TYPE_WAV, wav_file.read(), row[2], sample_id=wav_filename)
            return Sample(AUDIO_TYPE_WAV, wav_file.read(), sample_id=wav_filename)

    def __iter__(self):
        for i in range(len(self.rows)):
            yield self[i]

    def __len__(self):
        return len(self.rows)


def samples_from_source(sample_source, buffering=BUFFER_SIZE, labeled=None):
    """
    Returns an iterable of util.sample_collections.LabeledSample or util.audio.Sample instances
    loaded from a sample source file.

    Parameters
    ----------
    sample_source : str
        Path to the sample source file (SDB or CSV)
    buffering : int
        Read-buffer size to use while reading files
    labeled : bool or None
        If True: Reads LabeledSample instances. Fails, if source provides no transcripts.
        If False: Ignores transcripts (if available) and reads (unlabeled) util.audio.Sample instances.
        If None: Automatically determines if source provides transcripts
        (reading util.sample_collections.LabeledSample instances) or not (reading util.audio.Sample instances).
    """
    ext = os.path.splitext(sample_source)[1].lower()
    if ext == '.sdb':
        return SDB(sample_source, buffering=buffering, labeled=labeled)
    if ext == '.csv':
        return CSV(sample_source, labeled=labeled)
    raise ValueError('Unknown file type: "{}"'.format(ext))


def samples_from_sources(sample_sources, buffering=BUFFER_SIZE, labeled=None):
    """
    Returns an iterable of util.sample_collections.LabeledSample or util.audio.Sample instances
    loaded from a collection of sample source files.

    Parameters
    ----------
    sample_sources : list of str
        Paths to sample source files (SDBs or CSVs)
    buffering : int
        Read-buffer size to use while reading files
    labeled : bool or None
        If True: Reads LabeledSample instances. Fails, if not all sources provide transcripts.
        If False: Ignores transcripts (if available) and always reads (unlabeled) util.audio.Sample instances.
        If None: Reads util.sample_collections.LabeledSample instances from sources with transcripts and
        util.audio.Sample instances from sources with no transcripts.
    """
    sample_sources = list(sample_sources)
    if len(sample_sources) == 0:
        raise ValueError('No files')
    if len(sample_sources) == 1:
        return samples_from_source(sample_sources[0], buffering=buffering, labeled=labeled)
    cols = list(map(partial(samples_from_source, buffering=buffering, labeled=labeled), sample_sources))
    return Interleaved(*cols, key=lambda s: s.duration)


def get_overlay_from_spec(overlay_spec):
    match = OVERLAY_PARSER.match(overlay_spec)
    if not match:
        raise ValueError('Overlay specification has wrong format')
    overlay = Overlay(match[1])
    parameters = match[3]
    parameters = [] if parameters is None else parameters.split(',')
    for parameter in parameters:
        key, value = tuple(list(map(str.strip, (parameter.split('=') + ['', ''])))[:2])
        if hasattr(overlay, key):
            default_value = getattr(overlay, key)
            try:
                if isinstance(default_value, int):
                    value = int(value)
                elif isinstance(default_value, float):
                    value = float(value)
            except ValueError as ve:
                raise ValueError('Problem parsing value of overlay parameter "{}": {}'.format(key, ve))
            setattr(overlay, key, value)
        else:
            raise ValueError('Unknown overlay parameter "{}"'.format(key))
    return overlay


def enqueue_overlay_samples(sample_source, queue, buffering=BUFFER_SIZE):
    samples = samples_from_source(sample_source, buffering=buffering, labeled=False)
    while True:
        for sample in samples:
            queue.put(sample)


def dbfs(sample_data):
    return 20.0 * math.log10(math.sqrt(np.mean(sample_data**2).item(0))) + 3.0103


def gain_db_to_ratio(gain_db):
    return math.pow(10.0, gain_db / 20.0)


OVERLAYS = []
OVERLAY_PICK_RANGE = 0.0


def init_preparation_worker(overlays):
    global OVERLAYS, OVERLAY_PICK_RANGE  # pylint: disable=global-statement
    OVERLAYS = overlays
    for overlay in overlays:
        OVERLAY_PICK_RANGE += overlay.p_factor
        overlay.current_sample = None
    OVERLAY_PICK_RANGE = max(1.0, OVERLAY_PICK_RANGE)


def prepare_sample(sample):
    sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
    pick_value = random.uniform(0.0, OVERLAY_PICK_RANGE)
    overlay = None
    for o in OVERLAYS:  # randomly selecting an overlay or none at all (allowed if sum of p_factors < 1)
        if pick_value <= o.p_factor:
            overlay = o
            break
        pick_value -= o.p_factor
    if overlay is None:
        return sample
    n_layers = random.randint(overlay.n_layers_min, overlay.n_layers_max)
    sample_data = sample.audio
    n_samples = len(sample_data)
    overlay_data = np.zeros_like(sample_data)
    for _ in range(n_layers):
        overlay_offset = 0
        while overlay_offset < n_samples:
            if overlay.current_sample is None:
                overlay_sample = overlay.queue.get()
                overlay_sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
                overlay.current_sample = overlay_sample.audio
            n_required = n_samples - overlay_offset
            n_current = len(overlay.current_sample)
            if n_required >= n_current:  # take it completely
                overlay_data[overlay_offset:overlay_offset + n_current] += overlay.current_sample
                overlay_offset += n_current
                overlay.current_sample = None
            else:  # take required slice from head and keep tail for next layer or sample
                overlay_data[overlay_offset:overlay_offset + n_required] += overlay.current_sample[0:n_required]
                overlay_offset += n_required
                overlay.current_sample = overlay.current_sample[n_required:]
    snr_db = random.uniform(overlay.snr_db_min, overlay.snr_db_max)
    overlay_gain = dbfs(sample_data) - dbfs(overlay_data) - snr_db
    sample_data += overlay_data * gain_db_to_ratio(overlay_gain)
    sample_data = np.maximum(np.minimum(sample_data, 1.0), -1.0)
    sample.audio = sample_data
    return sample


def prepare_training_samples(sources, overlay_specs=None, buffering=BUFFER_SIZE, process_ahead=None):
    samples = samples_from_sources(sources, buffering=buffering, labeled=True)
    overlays = [] if overlay_specs is None else list(map(get_overlay_from_spec, overlay_specs))
    enqueue_processes = []
    try:
        for overlay in overlays:
            overlay.queue = Queue(OVERLAY_QUEUE_SIZE)
            enqueue_process = Process(target=enqueue_overlay_samples,
                                      args=(overlay.sample_source, overlay.queue),
                                      kwargs={'buffering': buffering})
            enqueue_processes.append(enqueue_process)
            enqueue_process.start()
        with LimitingPool(process_ahead=process_ahead,
                          initializer=init_preparation_worker,
                          initargs=(overlays,)) as pool:
            yield from pool.imap(prepare_sample, samples)
    finally:
        for enqueue_process in enqueue_processes:
            enqueue_process.terminate()
        for enqueue_process in enqueue_processes:
            enqueue_process.join()
