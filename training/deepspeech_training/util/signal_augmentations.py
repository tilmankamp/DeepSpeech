
import os
import re
import math
import random
import numpy as np

from multiprocessing import Queue, Process
from .audio import gain_db_to_ratio, dbfs, AUDIO_TYPE_NP
from .helpers import min_max_int, min_max_float, MEGABYTE

SPEC_PARSER = re.compile(r'^([a-z]+)(\[([^\[]*)\])?$')
BUFFER_SIZE = 1 * MEGABYTE


def enqueue_overlay_samples(sample_source, queue, buffering=BUFFER_SIZE):
    from .sample_collections import samples_from_source  # preventing cyclic import problems
    samples = samples_from_source(sample_source, buffering=buffering, labeled=False)
    while True:
        for sample in samples:
            queue.put(sample)


class Overlay:
    def __init__(self, source, snr=(3.0, 3.0), layers=(1, 1), p=1.0):
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
        sample_data = sample.audio
        n_samples = len(sample_data)
        overlay_data = np.zeros_like(sample_data)
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
        overlay_gain = dbfs(sample_data) - dbfs(overlay_data) - snr_db
        sample_data += overlay_data * gain_db_to_ratio(overlay_gain)
        sample_data = np.maximum(np.minimum(sample_data, 1.0), -1.0)
        sample.audio = sample_data

    def stop(self):
        if self.enqueue_process is not None:
            self.enqueue_process.terminate()


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
