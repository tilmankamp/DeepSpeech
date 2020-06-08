
import os
import re
import math
import random
import numpy as np

from multiprocessing import Queue, Process
from .audio import gain_db_to_ratio, max_dbfs, normalize_audio, AUDIO_TYPE_NP, AUDIO_TYPE_PCM, AUDIO_TYPE_OPUS
from .helpers import LimitingPool, int_range, float_range, pick_value_from_range, tf_pick_value_from_range, MEGABYTE

BUFFER_SIZE = 1 * MEGABYTE
SPEC_PARSER = re.compile(r'^(?P<cls>[a-z_]+)(\[(?P<params>.*)\])?$')


class Augmentation:
    def __init__(self, p=1.0):
        self.probability = float(p)


class SampleAugmentation(Augmentation):
    def start(self, buffering=BUFFER_SIZE):
        pass

    def apply(self, sample, clock=0.0):
        pass

    def stop(self):
        pass


class GraphAugmentation(Augmentation):
    def __init__(self, p=1.0, domain='spectrogram'):
        super(GraphAugmentation, self).__init__(p)
        if domain not in ['signal', 'spectrogram', 'features']:
            raise ValueError('Unsupported augmentation domain: {}'.format(domain))
        self.domain = domain

    def apply(self, tensor, clock=0.0):
        return tensor

    def apply_with_probability(self, tensor, clock=0.0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        rv = tf.random.stateless_uniform([], seed=(clock * tf.int32.min, clock * tf.int32.max))
        return tf.cond(tf.less(rv, self.probability),
                       lambda: self.apply(tensor, clock=clock),
                       lambda: tensor)

    def maybe_apply(self, domain, tensor, clock=0.0):
        if domain == self.domain:
            return self.apply_with_probability(tensor, clock=clock)
        return tensor


def parse_augmentation(augmentation_spec):
    """
    Parses an augmentation specification.

    Parameters
    ----------
    augmentation_spec : str
        Augmentation specification like "reverb[delay=20.0,decay=1.0]".

    Returns
    -------
    Instance of an augmentation class from util.signal_augmentations.*.
    """
    match = SPEC_PARSER.match(augmentation_spec)
    if not match:
        raise ValueError('Augmentation specification has wrong format')
    cls_name = ''.join(map(lambda p: p[0].upper() + p[1:], match.group('cls').split('_')))
    augmentation_cls = globals()[cls_name] if cls_name in globals() else None
    if not issubclass(augmentation_cls, Augmentation) or augmentation_cls == Augmentation:
        raise ValueError('Unknown augmentation: {}'.format(cls_name))
    parameters = match.group('params')
    parameters = [] if parameters is None else parameters.split(',')
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


def parse_augmentations(augmentation_specs):
    """
    Parses an augmentation specification.

    Parameters
    ----------
    augmentation_specs : list of str
        List of augmentation specifications like ["reverb[delay=20.0,decay=1.0]", "volume"].

    Returns
    -------
    List of augmentation class instances from util.signal_augmentations.*.
    """
    return [] if augmentation_specs is None else list(map(parse_augmentation, augmentation_specs))


def apply_graph_augmentations(domain, tensor, augmentations, clock=0.0):
    """
    Augments training sample tensor of a certain domain with matching augmentations of passed list.

    Parameters
    ----------
    domain : str
        Domain of the tensor to apply augmentations to. One of "signal", "spectrogram" or "features"
    tensor : Tensor of type float32
        Tensor to apply augmentations to.
    augmentations : list of augmentation class instances from util.signal_augmentations.*.
        List of augmentations of which only the spectrogram ones will get applied to the samples.
    clock : Tensor of type float32
        Time indicator for augmentation value-ranges. Running from 0.0 (start of training) to 1.0 (end of training).

    Returns
    -------
    Tensor of type float32
        The augmented spectrogram
    """
    if augmentations is not None:
        # Warp has to come before any masking
        for augmentation in augmentations:
            if isinstance(augmentation, Warp):
                tensor = augmentation.maybe_apply(domain, tensor, clock=clock)
        for augmentation in augmentations:
            if isinstance(augmentation, GraphAugmentation) and not isinstance(augmentation, Warp):
                tensor = augmentation.maybe_apply(domain, tensor, clock=clock)
    return tensor


class AugmentationContext:
    def __init__(self, target_audio_type, augmentations):
        self.target_audio_type = target_audio_type
        self.augmentations = augmentations


AUGMENTATION_CONTEXT = None


def _init_augmentation_worker(preparation_context):
    global AUGMENTATION_CONTEXT  # pylint: disable=global-statement
    AUGMENTATION_CONTEXT = preparation_context


def _augment_sample(timed_sample, context=None):
    context = AUGMENTATION_CONTEXT if context is None else context
    sample, clock = timed_sample
    for augmentation in context.augmentations:
        if random.random() < augmentation.probability:
            augmentation.apply(sample, clock)
    sample.change_audio_type(new_audio_type=context.target_audio_type)
    return sample


def apply_sample_augmentations(samples,
                               augmentations,
                               audio_type=AUDIO_TYPE_NP,
                               buffering=BUFFER_SIZE,
                               process_ahead=None,
                               clock_from=0.0,
                               clock_to=1.0):
    """
    Prepares samples for being used during training.
    This includes parallel and buffered application of augmentations and a conversion to a specified audio-type.

    Parameters
    ----------
    samples : Sample enumeration
        Typically produced by util.sample_collections.samples_from_sources.
    augmentations : list of augmentation class instances from util.signal_augmentations.*.
        List of augmentations of which only the signal ones will get applied to the samples.
    audio_type : str
        Target audio-type to convert samples to. See util.audio.Sample.__init__ .
    buffering : int
        Read-buffer size to use while reading files.
    process_ahead : int
        Number of samples to pre-process ahead of time.
    clock_from : float
        Start clock value between 0.0 and 1.0 for the first sample. Has to be <= than clock_to.
    clock_to : float
        Final clock value between 0.0 and 1.0 for the last sample. Has to be >= than clock_from.

    Returns
    -------
    iterable of util.sample_collections.LabeledSample or util.audio.Sample
    """
    def timed_samples():
        for sample_index, sample in enumerate(samples):
            sample_clock = clock_from + (clock_to - clock_from) * (sample_index / len(samples))
            yield sample, sample_clock

    assert 0.0 <= clock_from <= 1.0
    assert 0.0 <= clock_to <= 1.0
    assert clock_from <= clock_to
    augmentations = [] if augmentations is None else list(filter(lambda aug: isinstance(aug, SampleAugmentation),
                                                                 augmentations))
    try:
        for augmentation in augmentations:
            augmentation.start(buffering=buffering)
        context = AugmentationContext(audio_type, augmentations)
        if process_ahead == 0:
            for timed_sample in timed_samples():
                yield _augment_sample(timed_sample, context=context)
        else:
            with LimitingPool(process_ahead=process_ahead,
                              initializer=_init_augmentation_worker,
                              initargs=(context,)) as pool:
                yield from pool.imap(_augment_sample, timed_samples())
    finally:
        for augmentation in augmentations:
            augmentation.stop()


def _enqueue_overlay_samples(sample_source, queue, buffering=BUFFER_SIZE):
    """
    As the central distribution point for overlay samples this function is supposed to run in one process only.
    This ensures that samples are not used twice if not required.
    It loads the (raw and still compressed) data and provides it to the actual augmentation workers.
    These are then doing decompression, potential conversion and overlaying in parallel.
    """
    # preventing cyclic import problems
    from .sample_collections import samples_from_source  # pylint: disable=import-outside-toplevel
    samples = samples_from_source(sample_source, buffering=buffering, labeled=False)
    while True:
        for sample in samples:
            queue.put(sample)


class Overlay(SampleAugmentation):
    """See "Overlay augmentation" in TRAINING.rst"""
    def __init__(self, source, p=1.0, snr=3.0, layers=1):
        super(Overlay, self).__init__(p)
        self.source = source
        self.snr = float_range(snr)
        self.layers = int_range(layers)
        self.queue = Queue(max(1, math.floor(self.probability * self.layers[1] * os.cpu_count())))
        self.current_sample = None
        self.enqueue_process = None

    def start(self, buffering=BUFFER_SIZE):
        self.enqueue_process = Process(target=_enqueue_overlay_samples,
                                       args=(self.source, self.queue),
                                       kwargs={'buffering': buffering})
        self.enqueue_process.start()

    def apply(self, sample, clock=0.0):
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
        orig_dbfs = max_dbfs(audio)
        overlay_gain = orig_dbfs - max_dbfs(overlay_data) - snr_db
        audio += overlay_data * gain_db_to_ratio(overlay_gain)
        sample.audio = normalize_audio(audio, dbfs=orig_dbfs)

    def stop(self):
        if self.enqueue_process is not None:
            self.enqueue_process.terminate()


class Codec(SampleAugmentation):
    """See "Codec augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, bitrate=3200):
        super(Codec, self).__init__(p)
        self.bitrate = int_range(bitrate)

    def apply(self, sample, clock=0.0):
        bitrate = pick_value_from_range(self.bitrate, clock=clock)
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_PCM)  # decoding to ensure it has to get encoded again
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_OPUS, bitrate=bitrate)  # will get decoded again downstream


class Reverb(SampleAugmentation):
    """See "Reverb augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, delay=20.0, decay=10.0):
        super(Reverb, self).__init__(p)
        self.delay = float_range(delay)
        self.decay = float_range(decay)

    def apply(self, sample, clock=0.0):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        audio = np.array(sample.audio, dtype=np.float64)
        orig_dbfs = max_dbfs(audio)
        delay = pick_value_from_range(self.delay, clock=clock)
        decay = pick_value_from_range(self.decay, clock=clock)
        decay = gain_db_to_ratio(-decay)
        result = np.copy(audio)
        primes = [17, 19, 23, 29, 31]
        for delay_prime in primes:  # primes to minimize comb filter interference
            layer = np.copy(audio)
            n_delay = math.floor(delay * (delay_prime / primes[0]) * sample.audio_format.rate / 1000.0)
            n_delay = max(16, n_delay)  # 16 samples minimum to avoid performance trap and risk of division by zero
            for w_index in range(0, math.floor(len(audio) / n_delay)):
                w1 = w_index * n_delay
                w2 = (w_index + 1) * n_delay
                width = min(len(audio) - w2, n_delay)  # last window could be smaller
                layer[w2:w2 + width] += decay * layer[w1:w1 + width]
            result += layer
        audio = normalize_audio(result, dbfs=orig_dbfs)
        sample.audio = np.array(audio, dtype=np.float32)


class Resample(SampleAugmentation):
    """See "Resample augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, rate=8000):
        super(Resample, self).__init__(p)
        self.rate = int_range(rate)

    def apply(self, sample, clock=0.0):
        # late binding librosa and its dependencies
        from librosa.core import resample  # pylint: disable=import-outside-toplevel
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        rate = pick_value_from_range(self.rate, clock=clock)
        audio = sample.audio
        orig_len = len(audio)
        audio = np.swapaxes(audio, 0, 1)
        audio = resample(audio, sample.audio_format.rate, rate)
        audio = resample(audio, rate, sample.audio_format.rate)
        audio = np.swapaxes(audio, 0, 1)[0:orig_len]
        sample.audio = audio


class SampleVolume(SampleAugmentation):
    """See "Volume augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, dbfs=3.0103):
        super(SampleVolume, self).__init__(p)
        self.target_dbfs = float_range(dbfs)

    def apply(self, sample, clock=0.0):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        target_dbfs = pick_value_from_range(self.target_dbfs, clock=clock)
        sample.audio = normalize_audio(sample.audio, dbfs=target_dbfs)


class Volume(GraphAugmentation):
    """See "Amplify augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, dbfs=3.0103):
        super(Volume, self).__init__(p, domain='signal')
        self.target_dbfs = float_range(dbfs)

    def apply(self, signal, clock=0.0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        target_dbfs = tf_pick_value_from_range(self.target_dbfs, clock=clock)
        sample_max = tf.math.maximum(tf.math.abs(tf.math.reduce_max(signal)), tf.math.abs(tf.math.reduce_min(signal)))
        current_dbfs = 20.0 * tf.math.log(tf.math.maximum(1e-16, sample_max)) / tf.math.log(10.0) + 3.0103
        return signal * tf.math.pow(10.0, (target_dbfs - current_dbfs) / 20.0)


class PitchAndTempo(GraphAugmentation):
    """See "Pitch and tempo augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, tempo=1.2, pitch=(1.075, 1.075, 0.125)):
        super(PitchAndTempo, self).__init__(p, domain='spectrogram')
        self.tempo = float_range(tempo)
        self.pitch = float_range(pitch)

    def apply(self, spectrogram, clock=0.0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        original_shape = tf.shape(spectrogram)
        pitch = tf_pick_value_from_range(self.pitch, clock=clock)
        tempo = tf.math.maximum(1.0, tf_pick_value_from_range(self.tempo, clock=clock))
        new_freq_size = tf.cast(tf.cast(original_shape[2], tf.float32) * pitch, tf.int32)
        new_time_size = tf.cast(tf.cast(original_shape[1], tf.float32) / tempo, tf.int32)
        spectrogram_aug = tf.image.resize_bilinear(tf.expand_dims(spectrogram, -1), [new_time_size, new_freq_size])
        spectrogram_aug = tf.image.crop_to_bounding_box(spectrogram_aug,
                                                        offset_height=0,
                                                        offset_width=0,
                                                        target_height=tf.shape(spectrogram_aug)[1],
                                                        target_width=tf.math.minimum(original_shape[2], new_freq_size))
        spectrogram_aug = tf.cond(pitch < 1,
                                  lambda: tf.image.pad_to_bounding_box(spectrogram_aug,
                                                                       offset_height=0,
                                                                       offset_width=0,
                                                                       target_height=tf.shape(spectrogram_aug)[1],
                                                                       target_width=original_shape[2]),
                                  lambda: spectrogram_aug)
        return spectrogram_aug[:, :, :, 0]


class Warp(GraphAugmentation):
    """See "Warp augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, shift=20, order=3, nbp=1, ncp=1, regularization_weight=0.0):
        super(Warp, self).__init__(p, domain='spectrogram')
        self.shift = int_range(shift)
        self.order = int_range(order)
        self.nbp = int_range(nbp)
        self.ncp = int_range(ncp)
        # Making this a value-range is impossible, as it would get a tensor which would downstream be used as parameter
        # of a comparison inside tensorflow.contrib.image.python.ops.interpolate_spline. This is not supported.
        self.regularization_weight = float(regularization_weight)

    def apply(self, spectrogram, clock=0.0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        from .sparse_image_warp import sparse_image_warp  # pylint: disable=import-outside-toplevel

        # reshape to fit `sparse_image_warp`'s input shape (1, time steps, freq, 1), batch_size must be 1
        expanded_spectrogram = tf.expand_dims(spectrogram, -1)
        original_shape = tf.shape(expanded_spectrogram)
        tau, freq_size = original_shape[1], original_shape[2]
        seed = (clock * tf.int32.min, clock * tf.int32.max)

        shift = tf_pick_value_from_range(self.shift, clock=clock)
        shift = tf.math.minimum(shift, tf.math.floordiv(tau, 2) - 1)  # to protect short audio
        nbp = tf_pick_value_from_range(self.nbp, clock=clock)
        ncp = tf_pick_value_from_range(self.ncp, clock=clock)
        # workaround for missing stateless shuffle support
        frequencies = tf.random.stateless_uniform([2 * ncp], seed, minval=1, maxval=freq_size - 2, dtype=tf.int32)
        frequencies = tf.unique(tf.concat([frequencies, tf.range(1, limit=freq_size - 3)], axis=0))[0][0:ncp]
        source_max = tau - shift
        source_min = tf.math.minimum(source_max - ncp, shift)
        # workaround for missing stateless shuffle support
        src_times = tf.random.stateless_uniform([2 * ncp], seed, minval=source_min, maxval=source_max, dtype=tf.int32)
        src_times = tf.unique(tf.concat([src_times, tf.range(1, limit=source_max)], axis=0))[0][0:ncp]
        dst_times = src_times + tf.random.stateless_uniform([ncp], seed, minval=-shift, maxval=shift, dtype=tf.int32)
        scp_locations = tf.cast([tf.transpose(tf.stack([src_times, frequencies]))], dtype=tf.float32)
        dcp_locations = tf.cast([tf.transpose(tf.stack([dst_times, frequencies]))], dtype=tf.float32)

        order = tf_pick_value_from_range(self.order, clock=clock)
        order = tf.math.maximum(3, order)  # prevents "Input matrix is not invertible." exception
        order = tf.cast(order, tf.float32)

        spectrogram_aug, _ = sparse_image_warp(expanded_spectrogram,
                                               source_control_point_locations=scp_locations,
                                               dest_control_point_locations=dcp_locations,
                                               interpolation_order=order,
                                               regularization_weight=self.regularization_weight,
                                               num_boundary_points=nbp)
        return tf.reshape(spectrogram_aug, shape=(1, -1, freq_size))


class Speed(GraphAugmentation):
    """See "Speed augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, factor=1.1):
        super(Speed, self).__init__(p, domain='spectrogram')
        self.factor = float_range(factor)

    def apply(self, spectrogram, clock=0.0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        factor = tf_pick_value_from_range(self.factor, clock=clock)
        original_shape = tf.shape(spectrogram)
        new_time_size = tf.cast(tf.cast(original_shape[1], tf.float32) / factor, tf.int32)
        spectrogram_aug = tf.image.resize_bilinear(tf.expand_dims(spectrogram, -1), [new_time_size, original_shape[2]])
        return spectrogram_aug[:, :, :, 0]


class FrequencyMask(GraphAugmentation):
    """See "Frequency mask augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, n=3, size=2):
        super(FrequencyMask, self).__init__(p, domain='spectrogram')
        self.n = int_range(n)
        self.size = int_range(size)

    def apply(self, spectrogram, clock=0.0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        time_max = tf.shape(spectrogram)[1]
        freq_max = tf.shape(spectrogram)[2]
        n = tf_pick_value_from_range(self.n, clock=clock)

        def body(i, spectrogram_aug):
            size = tf_pick_value_from_range(self.size, clock=clock)
            size = tf.math.maximum(1, tf.math.minimum(freq_max - 1, size))
            seed = tf.cast(clock * tf.int32.max, tf.int32) - i
            f0 = tf.random.stateless_uniform((), (-seed, seed), minval=0, maxval=freq_max - size, dtype=tf.dtypes.int32)
            freq_mask = tf.concat([tf.ones([1, time_max, f0]),
                                   tf.zeros([1, time_max, size]),
                                   tf.ones([1, time_max, freq_max - f0 - size])], axis=2)
            return i + 1, spectrogram_aug * freq_mask

        return tf.while_loop(lambda i, spectrogram_aug: i < n, body, (0, spectrogram))[1]


class TimeMask(GraphAugmentation):
    """See "Time mask augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, domain='spectrogram', n=3, size=2):
        super(TimeMask, self).__init__(p, domain=domain)
        self.n = int_range(n)
        self.size = int_range(size)

    def apply(self, tensor, clock=0.0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        time_max = tf.shape(tensor)[0] if self.domain == 'signal' else tf.shape(tensor)[1]
        n = tf_pick_value_from_range(self.n, clock=clock)

        def body(i, augmented):
            size = tf_pick_value_from_range(self.size, clock=clock)
            size = tf.math.maximum(1, tf.math.minimum(time_max - 1, size))
            seed = tf.cast(clock * tf.int32.max, tf.int32) - i
            t0 = tf.random.stateless_uniform((), (-seed, seed), minval=0, maxval=time_max - size, dtype=tf.dtypes.int32)
            rest = time_max - t0 - size
            if self.domain == 'spectrogram':
                fm = tf.shape(tensor)[2]
                time_mask = tf.concat([tf.ones([1, t0, fm]), tf.zeros([1, size, fm]), tf.ones([1, rest, fm])], axis=1)
            elif self.domain == 'signal':
                time_mask = tf.concat([tf.ones([t0, 1]), tf.zeros([size, 1]), tf.ones([rest, 1])], axis=0)
            else:
                time_mask = tf.concat([tf.ones([1, t0]), tf.zeros([1, size]), tf.ones([1, rest])], axis=1)
            return i + 1, augmented * time_mask

        return tf.while_loop(lambda i, augmented: i < n, body, (0, tensor))[1]


class Dropout(GraphAugmentation):
    """See "Dropout augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, domain='spectrogram', rate=0.05):
        super(Dropout, self).__init__(p, domain=domain)
        self.rate = float_range(rate)

    def apply(self, tensor, clock=0.0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        rate = tf_pick_value_from_range(self.rate, clock=clock)
        rate = tf.math.maximum(0.0, rate)

        def drop():
            factors = tf.random.stateless_uniform(tf.shape(tensor),
                                                  (clock * tf.int32.min, clock * tf.int32.max),
                                                  minval=0.0,
                                                  maxval=1.0,
                                                  dtype=tf.float32)
            return tensor * tf.math.sign(tf.math.floor(factors / rate))

        return tf.cond(rate > 0.0, drop, lambda: tensor)


class Add(GraphAugmentation):
    """See "Add augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, domain='features', stddev=5):
        super(Add, self).__init__(p, domain=domain)
        self.stddev = float_range(stddev)

    def apply(self, tensor, clock=0.0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        stddev = tf_pick_value_from_range(self.stddev, clock=clock)
        seed = (clock * tf.int32.min, clock * tf.int32.max)
        return tensor + tf.random.stateless_normal(tf.shape(tensor), seed, mean=0.0, stddev=stddev)


class Multiply(GraphAugmentation):
    """See "Multiply augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, domain='features', stddev=5):
        super(Multiply, self).__init__(p, domain=domain)
        self.stddev = float_range(stddev)

    def apply(self, tensor, clock=0.0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        stddev = tf_pick_value_from_range(self.stddev, clock=clock)
        seed = (clock * tf.int32.min, clock * tf.int32.max)
        return tensor * tf.random.stateless_normal(tf.shape(tensor), seed, mean=1.0, stddev=stddev)
