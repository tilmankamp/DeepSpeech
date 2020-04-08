
import os
import re
import math
import random
import numpy as np

from librosa.core import resample
from multiprocessing import Queue, Process
from .audio import gain_db_to_ratio, dbfs, AUDIO_TYPE_NP, AUDIO_TYPE_PCM, AUDIO_TYPE_OPUS
from .helpers import min_max_int, min_max_float, MEGABYTE

SPEC_PARSER = re.compile(r'^([a-z]+)(\[(.*)\])?$')
BUFFER_SIZE = 1 * MEGABYTE


def enqueue_overlay_samples(sample_source, queue, buffering=BUFFER_SIZE):
    from .sample_collections import samples_from_source  # preventing cyclic import problems
    samples = samples_from_source(sample_source, buffering=buffering, labeled=False)
    while True:
        for sample in samples:
            queue.put(sample)


class Overlay:
    def __init__(self, source, snr=3.0, layers=1, p=1.0):
        self.source = source
        self.snr = min_max_float(snr)
        self.layers = min_max_int(layers)
        self.p = float(p)
        self.queue = Queue(max(1, math.floor(self.p * self.layers[1] * os.cpu_count())))
        self.current_sample = None
        self.enqueue_process = None

    def start(self, buffering=BUFFER_SIZE):
        self.enqueue_process = Process(target=enqueue_overlay_samples,
                                       args=(self.source, self.queue),
                                       kwargs={'buffering': buffering})
        self.enqueue_process.start()

    def apply(self, sample):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        n_layers = random.randint(self.layers[0], self.layers[1])
        audio = sample.audio
        n_samples = len(audio)
        overlay_data = np.zeros_like(audio)
        for _ in range(n_layers):
            overlay_offset = 0
            while overlay_offset < n_samples:
                if self.current_sample is None:
                    next_overlay_sample = self.queue.get()
                    next_overlay_sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
                    self.current_sample = next_overlay_sample.audio
                n_required = n_samples - overlay_offset
                n_current = len(self.current_sample)
                if n_required >= n_current:  # take it completely
                    overlay_data[overlay_offset:overlay_offset + n_current] += self.current_sample
                    overlay_offset += n_current
                    self.current_sample = None
                else:  # take required slice from head and keep tail for next layer or sample
                    overlay_data[overlay_offset:overlay_offset + n_required] += self.current_sample[0:n_required]
                    overlay_offset += n_required
                    self.current_sample = self.current_sample[n_required:]
        snr_db = random.uniform(self.snr[0], self.snr[1])
        overlay_gain = dbfs(audio).max - dbfs(overlay_data).max - snr_db
        audio += overlay_data * gain_db_to_ratio(overlay_gain)
        audio = np.maximum(np.minimum(audio, 1.0), -1.0)
        sample.audio = audio

    def stop(self):
        if self.enqueue_process is not None:
            self.enqueue_process.terminate()


class Reverb:
    def __init__(self, delay=5.0, decay=(0.4, 0.9), p=1.0):
        self.delay = min_max_float(delay)
        self.decay = min_max_float(decay)
        self.p = float(p)

    def apply(self, sample):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        audio = sample.audio
        orig_dbfs = dbfs(audio)
        rate = sample.audio_format.rate
        delay_factor = random.uniform(self.delay[0], self.delay[1])
        decay_factor = random.uniform(self.decay[0], self.decay[1])
        result = np.copy(audio)
        for delay in [0.0281, 0.0317, 0.0407, 0.0134]:
            decay = random.uniform(0.2, 0.8) * decay_factor
            layer = np.copy(audio)
            n_delay = math.floor(rate * delay * delay_factor)
            for w_index in range(0, math.floor(len(audio) / n_delay)):
                w1 = w_index * n_delay
                w2 = (w_index + 1) * n_delay
                width = min(len(audio) - w2, n_delay)
                layer[w2:w2 + width] += decay * decay_factor * layer[w1:w1 + width]
            result += layer
        audio = result
        audio *= gain_db_to_ratio(orig_dbfs.max - dbfs(audio).max)
        audio = np.maximum(np.minimum(audio, 1.0), -1.0)
        sample.audio = audio


class Resample:
    def __init__(self, rate=8000, p=1.0):
        self.rate = min_max_int(rate)
        self.p = float(p)

    def apply(self, sample):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        rate = random.randint(self.rate[0], self.rate[1])
        audio = sample.audio
        audio = np.swapaxes(audio, 0, 1)
        audio = resample(audio, sample.audio_format.rate, rate)
        audio = resample(audio, rate, sample.audio_format.rate)
        audio = np.swapaxes(audio, 0, 1)
        sample.audio = audio


class Compress:
    def __init__(self, bitrate=3200, p=1.0):
        self.bitrate = min_max_int(bitrate)
        self.p = float(p)

    def apply(self, sample):
        bitrate = random.randint(self.bitrate[0], self.bitrate[1])
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_PCM)
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_OPUS, bitrate=bitrate)


class Gaps:
    def __init__(self, n=(1, 4), size=(100, 1000), p=1.0):
        self.n = min_max_int(n)
        self.size = min_max_int(size)
        self.p = float(p)

    def apply(self, sample):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        audio = sample.audio
        n = random.randint(self.n[0], self.n[1])
        for _ in range(n):
            size = random.randint(self.size[0], self.size[1])
            offset = max(0, random.randint(0, len(audio) - size - 1))
            audio[offset:offset + size] = 0
        sample.audio = audio


class Amplify:
    def __init__(self, db=-10.0, p=1.0):
        self.db = min_max_float(db)
        self.p = float(p)

    def apply(self, sample):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        audio = sample.audio
        db = random.uniform(self.db[0], self.db[1])
        audio *= gain_db_to_ratio(db)
        audio = np.maximum(np.minimum(audio, 1.0), -1.0)
        sample.audio = audio


class Volume:
    def __init__(self, dbfs=(-20.0, 0.0), p=1.0):
        self.target_dbfs = min_max_float(dbfs)
        self.p = float(p)

    def apply(self, sample):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        audio = sample.audio
        target_dbfs = random.uniform(self.target_dbfs[0], self.target_dbfs[1])
        current_dbfs = dbfs(audio).max
        audio *= gain_db_to_ratio(target_dbfs - current_dbfs)
        audio = np.maximum(np.minimum(audio, 1.0), -1.0)
        sample.audio = audio


def parse_augmentation(augmentation_spec):
    match = SPEC_PARSER.match(augmentation_spec)
    if not match:
        raise ValueError('Augmentation specification has wrong format')
    cls_name = match[1][0].upper() + match[1][1:]
    if cls_name not in globals():
        raise ValueError('Unknown augmentation: {}'.format(cls_name))
    augmentation_cls = globals()[cls_name]
    parameters = [] if match[3] is None else match[3].split(',')
    args = []
    kwargs = {}
    for parameter in parameters:
        pair = tuple(list(map(str.strip, (parameter.split('=')))))
        if len(pair) == 1:
            args.append(pair)
        elif len(pair) == 2:
            kwargs[pair[0]] = pair[1]
        else:
            raise ValueError('Unable to parse augmentation value assignment')
    return augmentation_cls(*args, **kwargs)
