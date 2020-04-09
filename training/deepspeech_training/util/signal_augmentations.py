
import os
import re
import math
import random
import numpy as np

from librosa.core import resample
from multiprocessing import Queue, Process
from .audio import gain_db_to_ratio, compute_dbfs, AUDIO_TYPE_NP, AUDIO_TYPE_PCM, AUDIO_TYPE_OPUS
from .helpers import int_range, float_range, pick_value_from_range, MEGABYTE

SPEC_PARSER = re.compile(r'^([a-z]+)(\[(.*)\])?$')
BUFFER_SIZE = 1 * MEGABYTE


def enqueue_overlay_samples(sample_source, queue, buffering=BUFFER_SIZE):
    from .sample_collections import samples_from_source  # preventing cyclic import problems
    samples = samples_from_source(sample_source, buffering=buffering, labeled=False)
    while True:
        for sample in samples:
            queue.put(sample)


class Overlay:
    def __init__(self, source, p=1.0, snr=3.0, layers=1):
        self.source = source
        self.p = float(p)
        self.snr = float_range(snr)
        self.layers = int_range(layers)
        self.queue = Queue(max(1, math.floor(self.p * self.layers[1] * os.cpu_count())))
        self.current_sample = None
        self.enqueue_process = None

    def start(self, buffering=BUFFER_SIZE):
        self.enqueue_process = Process(target=enqueue_overlay_samples,
                                       args=(self.source, self.queue),
                                       kwargs={'buffering': buffering})
        self.enqueue_process.start()

    def apply(self, sample, clock):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        n_layers = pick_value_from_range(self.layers, clock=clock)
        audio = sample.audio
        overlay_data = np.zeros_like(audio)
        for _ in range(n_layers):
            overlay_offset = 0
            while overlay_offset < len(audio):
                if self.current_sample is None:
                    next_overlay_sample = self.queue.get()
                    next_overlay_sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
                    self.current_sample = next_overlay_sample.audio
                n_required = len(audio) - overlay_offset
                n_current = len(self.current_sample)
                if n_required >= n_current:  # take it completely
                    overlay_data[overlay_offset:overlay_offset + n_current] += self.current_sample
                    overlay_offset += n_current
                    self.current_sample = None
                else:  # take required slice from head and keep tail for next layer or sample
                    overlay_data[overlay_offset:overlay_offset + n_required] += self.current_sample[0:n_required]
                    overlay_offset += n_required
                    self.current_sample = self.current_sample[n_required:]
        snr_db = pick_value_from_range(self.snr, clock=clock)
        overlay_gain = compute_dbfs(audio).max - compute_dbfs(overlay_data).max - snr_db
        audio += overlay_data * gain_db_to_ratio(overlay_gain)
        audio = np.maximum(np.minimum(audio, 1.0), -1.0)
        sample.audio = audio

    def stop(self):
        if self.enqueue_process is not None:
            self.enqueue_process.terminate()


class Reverb:
    def __init__(self, p=1.0, delay=(0.5, 0.5, 0.2), decay=(0.6, 0.6, 0.25)):
        self.p = float(p)
        self.delay = float_range(delay)
        self.decay = float_range(decay)

    def apply(self, sample, clock):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        audio = sample.audio
        orig_dbfs = compute_dbfs(audio)
        rate = sample.audio_format.rate
        delay_factor = pick_value_from_range(self.delay, clock=clock)
        decay_factor = pick_value_from_range(self.decay, clock=clock)
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
        audio *= gain_db_to_ratio(orig_dbfs.max - compute_dbfs(audio).max)
        audio = np.maximum(np.minimum(audio, 1.0), -1.0)
        sample.audio = audio


class Resample:
    def __init__(self, p=1.0, rate=8000):
        self.p = float(p)
        self.rate = int_range(rate)

    def apply(self, sample, clock):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        rate = pick_value_from_range(self.rate, clock=clock)
        audio = sample.audio
        audio = np.swapaxes(audio, 0, 1)
        audio = resample(audio, sample.audio_format.rate, rate)
        audio = resample(audio, rate, sample.audio_format.rate)
        audio = np.swapaxes(audio, 0, 1)
        sample.audio = audio


class Codec:
    def __init__(self, p=1.0, bitrate=3200):
        self.p = float(p)
        self.bitrate = int_range(bitrate)

    def apply(self, sample, clock):
        bitrate = pick_value_from_range(self.bitrate, clock=clock)
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_PCM)  # decoding to ensure it has to get encoded again
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_OPUS, bitrate=bitrate)  # will get decoded again downstream


class Gaps:
    def __init__(self, p=1.0, n=(3, 3, 2), size=(1000, 1000, 900)):
        self.p = float(p)
        self.n = int_range(n)
        self.size = int_range(size)

    def apply(self, sample, clock):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        audio = sample.audio
        n = pick_value_from_range(self.n, clock=clock)
        for _ in range(n):
            size = pick_value_from_range(self.size, clock=clock)
            offset = max(0, random.randint(0, len(audio) - size - 1))
            audio[offset:offset + size] = 0
        sample.audio = audio


class Volume:
    def __init__(self, p=1.0, dbfs=0.0):
        self.p = float(p)
        self.target_dbfs = float_range(dbfs)

    def apply(self, sample, clock):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        audio = sample.audio
        target_dbfs = pick_value_from_range(self.target_dbfs, clock=clock)
        current_dbfs = compute_dbfs(audio).max
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
