
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


class SignalAugmentation(Augmentation):
    def start(self, buffering=BUFFER_SIZE):
        pass

    def apply(self, sample, clock=0.0):
        pass

    def stop(self):
        pass


class GraphAugmentation(Augmentation):
    def apply(self, tensor, clock=0, seed=0):
        return tensor

    def apply_with_probability(self, tensor, clock=0, seed=0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        return tf.cond(tf.less(tf.random.stateless_uniform([], (0, seed)), self.probability),
                       lambda: self.apply(tensor, clock=clock, seed=seed),
                       lambda: tensor)


class SpectrogramAugmentation(GraphAugmentation):
    def __init__(self, p=1.0, priority=1):
        super(SpectrogramAugmentation, self).__init__(p)
        self.priority = priority


class FeaturesAugmentation(GraphAugmentation):
    pass


def parse_augmentation(augmentation_spec):
    """
    Parses an augmentation specification.

    Parameters
    ----------
    augmentation_spec : str
        Augmentation specification like "reverb[delay=20.0,decay=1.0]".
    modules : list of Python modules
        Modules to look-up for augmentation classes

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
    modules : list of Python modules
        Modules to look-up for augmentation classes

    Returns
    -------
    List of augmentation class instances from util.signal_augmentations.*.
    """
    augmentations = [] if augmentation_specs is None else list(map(parse_augmentation, augmentation_specs))
    for augmentation in augmentations:
        if isinstance(augmentation, GraphAugmentation):
            for other_augmentation in augmentations:
                if augmentation != other_augmentation and type(augmentation) == type(other_augmentation):
                    raise ValueError('Augmentation {} can only be specified once'.format(type(augmentation).__name__))
    return augmentations


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


class Overlay(SignalAugmentation):
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
        orig_dbfs = max_dbfs(audio)
        overlay_gain = orig_dbfs - max_dbfs(overlay_data) - snr_db
        audio += overlay_data * gain_db_to_ratio(overlay_gain)
        sample.audio = normalize_audio(audio, dbfs=orig_dbfs)

    def stop(self):
        if self.enqueue_process is not None:
            self.enqueue_process.terminate()


class Reverb(SignalAugmentation):
    """See "Reverb augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, delay=20.0, decay=10.0):
        super(Reverb, self).__init__(p)
        self.delay = float_range(delay)
        self.decay = float_range(decay)

    def apply(self, sample, clock):
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


class Resample(SignalAugmentation):
    """See "Resample augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, rate=8000):
        super(Resample, self).__init__(p)
        self.rate = int_range(rate)

    def apply(self, sample, clock):
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


class Codec(SignalAugmentation):
    """See "Codec augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, bitrate=3200):
        super(Codec, self).__init__(p)
        self.bitrate = int_range(bitrate)

    def apply(self, sample, clock):
        bitrate = pick_value_from_range(self.bitrate, clock=clock)
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_PCM)  # decoding to ensure it has to get encoded again
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_OPUS, bitrate=bitrate)  # will get decoded again downstream


class Gaps(SignalAugmentation):
    """See "Gaps augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, n=1, size=50.0):
        super(Gaps, self).__init__(p)
        self.n_gaps = int_range(n)
        self.size = float_range(size)

    def apply(self, sample, clock):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        audio = sample.audio
        n_gaps = pick_value_from_range(self.n_gaps, clock=clock)
        for _ in range(n_gaps):
            size = pick_value_from_range(self.size, clock=clock)
            size = int(size * sample.audio_format.rate / 1000.0)
            size = min(size, len(audio) // 10)  # a gap should never exceed 10 percent of the audio
            offset = random.randint(0, max(0, len(audio) - size - 1))
            audio[offset:offset + size] = 0
        sample.audio = audio


class Volume(SignalAugmentation):
    """See "Volume augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, dbfs=3.0103):
        super(Volume, self).__init__(p)
        self.target_dbfs = float_range(dbfs)

    def apply(self, sample, clock):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        target_dbfs = pick_value_from_range(self.target_dbfs, clock=clock)
        sample.audio = normalize_audio(sample.audio, dbfs=target_dbfs)


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


def apply_signal_augmentations(samples,
                               augmentations,
                               audio_type=AUDIO_TYPE_NP,
                               buffering=BUFFER_SIZE,
                               process_ahead=None,
                               repetitions=1,
                               fixed_clock=None):
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
    repetitions : int
        How often the input sample enumeration should get repeated for being re-augmented.
    fixed_clock : float
        Sets the internal clock to a value between 0.0 (beginning of epoch) and 1.0 (end of epoch).
        Setting this to a number is used for simulating augmentations at a certain epoch-time.
        If kept at None (default), the internal clock will run regularly from 0.0 to 1.0,
        hence preparing them for training.

    Returns
    -------
    iterable of util.sample_collections.LabeledSample or util.audio.Sample
    """
    def timed_samples():
        for repetition in range(repetitions):
            for sample_index, sample in enumerate(samples):
                if fixed_clock is None:
                    yield sample, (repetition * len(samples) + sample_index) / (repetitions * len(samples))
                else:
                    yield sample, fixed_clock

    augmentations = [] if augmentations is None else list(filter(lambda aug: isinstance(aug, SignalAugmentation),
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


class TimeMask(SpectrogramAugmentation):
    """See "Time mask augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, n=3, max_size=2):
        super(TimeMask, self).__init__(p)
        self.n = int(n)
        self.max_size = int(max_size)

    def apply(self, spectrogram, clock=0, seed=0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        time_max = tf.shape(spectrogram)[1]
        freq_max = tf.shape(spectrogram)[2]
        spectrogram_aug = spectrogram
        for _ in range(self.n):
            f = tf.random.uniform(shape=(), minval=0, maxval=self.max_size, dtype=tf.dtypes.int32)
            f0 = tf.random.uniform(shape=(), minval=0, maxval=freq_max - f, dtype=tf.dtypes.int32)
            value_ones_freq_prev = tf.ones(shape=[1, time_max, f0])
            value_zeros_freq = tf.zeros(shape=[1, time_max, f])
            value_ones_freq_next = tf.ones(shape=[1, time_max, freq_max - (f0 + f)])
            freq_mask = tf.concat([value_ones_freq_prev, value_zeros_freq, value_ones_freq_next], axis=2)
            spectrogram_aug = spectrogram_aug * freq_mask
        return spectrogram_aug


class FrequencyMask(SpectrogramAugmentation):
    """See "Frequency mask augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, n=3, max_size=5):
        super(FrequencyMask, self).__init__(p)
        self.n = int(n)
        self.max_size = int(max_size)

    def apply(self, spectrogram, clock=0, seed=0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        time_max = tf.shape(spectrogram)[1]
        freq_max = tf.shape(spectrogram)[2]
        spectrogram_aug = spectrogram
        for _ in range(self.n):
            t = tf.random.uniform(shape=(), minval=0, maxval=self.max_size, dtype=tf.dtypes.int32)
            t0 = tf.random.uniform(shape=(), minval=0, maxval=time_max - t, dtype=tf.dtypes.int32)
            value_zeros_time_prev = tf.ones(shape=[1, t0, freq_max])
            value_zeros_time = tf.zeros(shape=[1, t, freq_max])
            value_zeros_time_next = tf.ones(shape=[1, time_max - (t0 + t), freq_max])
            time_mask = tf.concat([value_zeros_time_prev, value_zeros_time, value_zeros_time_next], axis=1)
            spectrogram_aug = spectrogram_aug * time_mask
        return spectrogram_aug


class PitchAndTempo(SpectrogramAugmentation):
    """See "Pitch and tempo augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, max_tempo=1.2, min_pitch=0.95, max_pitch=1.2):
        super(PitchAndTempo, self).__init__(p)
        self.max_tempo = float(max_tempo)
        self.min_pitch = float(min_pitch)
        self.max_pitch = float(max_pitch)

    def apply(self, spectrogram, clock=0, seed=0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        original_shape = tf.shape(spectrogram)
        chosen_pitch = tf.random.uniform(shape=(), minval=self.min_pitch, maxval=self.max_pitch)
        chosen_tempo = tf.random.uniform(shape=(), minval=1, maxval=self.max_tempo)
        new_freq_size = tf.cast(tf.cast(original_shape[2], tf.float32) * chosen_pitch, tf.int32)
        new_time_size = tf.cast(tf.cast(original_shape[1], tf.float32) / chosen_tempo, tf.int32)
        spectrogram_aug = tf.image.resize_bilinear(tf.expand_dims(spectrogram, -1), [new_time_size, new_freq_size])
        spectrogram_aug = tf.image.crop_to_bounding_box(spectrogram_aug,
                                                        offset_height=0,
                                                        offset_width=0,
                                                        target_height=tf.shape(spectrogram_aug)[1],
                                                        target_width=tf.minimum(original_shape[2], new_freq_size))
        spectrogram_aug = tf.cond(chosen_pitch < 1,
                                  lambda: tf.image.pad_to_bounding_box(spectrogram_aug,
                                                                       offset_height=0,
                                                                       offset_width=0,
                                                                       target_height=tf.shape(spectrogram_aug)[1],
                                                                       target_width=original_shape[2]),
                                  lambda: spectrogram_aug)
        return spectrogram_aug[:, :, :, 0]


class SpeedUp(SpectrogramAugmentation):
    """See "Speed-up augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, std=0.1):
        super(SpeedUp, self).__init__(p)
        self.std = float(std)

    def apply(self, spectrogram, clock=0, seed=0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        original_shape = tf.shape(spectrogram)
        # abs makes sure the augmentation will only speed up
        chosen_speed = 1 + tf.math.abs(tf.random.normal(shape=(), stddev=self.std))
        new_freq_size = tf.cast(tf.cast(original_shape[2], tf.float32), tf.int32)
        new_time_size = tf.cast(tf.cast(original_shape[1], tf.float32) / chosen_speed, tf.int32)
        spectrogram_aug = tf.image.resize_bilinear(tf.expand_dims(spectrogram, -1), [new_time_size, new_freq_size])
        return spectrogram_aug[:, :, :, 0]


class Dropout(SpectrogramAugmentation):
    """See "Dropout augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, rate=0.05):
        super(Dropout, self).__init__(p)
        self.rate = float(rate)

    def apply(self, spectrogram, clock=0, seed=0):
        import tensorflow as tf
        return tf.nn.dropout(spectrogram, rate=self.rate)


class SparseWarp(SpectrogramAugmentation):
    """See "Sparse-warp augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, warp=20, interpolation_order=2, regularization_weight=0.0, nbp=1, ncp=1):
        super(SparseWarp, self).__init__(p)
        self.warp = int(warp)
        self.interpolation_order = int(interpolation_order)
        self.regularization_weight = float(regularization_weight)
        self.nbp = int(nbp)
        self.ncp = int(ncp)

    def apply(self, spectrogram, clock=0, seed=0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        import tensorflow.compat.v1 as tfv1  # pylint: disable=import-outside-toplevel
        from .sparse_image_warp import sparse_image_warp  # pylint: disable=import-outside-toplevel

        # reshape to fit `sparse_image_warp`'s input shape (1, time steps, freq, 1), batch_size must be 1
        expanded_spectrogram = tf.expand_dims(spectrogram, -1)

        original_shape = tf.shape(expanded_spectrogram)
        tau, freq_size = original_shape[1], original_shape[2]

        # to protect short audio
        warp = tf.math.minimum(self.warp, tf.math.subtract(tf.math.floordiv(tau, 2), 1))

        # don't choose boundary frequency
        chosen_frequencies = tf.random.shuffle(tf.add(tf.range(freq_size - 3), 1))[0: self.ncp]

        source_max = tau - warp
        source_min = tf.math.minimum(source_max - self.ncp, warp)
        chosen_times = tf.random.shuffle(tf.range(source_min, limit=source_max))[0: self.ncp]
        dest_time_widths = tfv1.random_uniform([self.ncp], tf.negative(warp), warp, tf.int32)

        sources = []
        dests = []
        for i in range(self.ncp):
            # generate source points `t` of time axis between (W, tau-W)
            rand_source_time = chosen_times[i]
            rand_dest_time = rand_source_time + dest_time_widths[i]
            chosen_frequency = chosen_frequencies[i]
            sources.append([rand_source_time, chosen_frequency])
            dests.append([rand_dest_time, chosen_frequency])

        spectrogram_aug, _ = sparse_image_warp(expanded_spectrogram,
                                               source_control_point_locations=tf.cast([sources], tf.float32),
                                               dest_control_point_locations=tf.cast([dests], tf.float32),
                                               interpolation_order=self.interpolation_order,
                                               regularization_weight=self.regularization_weight,
                                               num_boundary_points=self.nbp)
        return tf.reshape(spectrogram_aug, shape=(1, -1, freq_size))


def apply_spectrogram_augmentations(spectrogram, augmentations, clock=0, seed=0):
    """
    Augments training sample spectrograms with spectrogram augmentations from passed augmentation list.

    Parameters
    ----------
    spectrogram : Tensor of type float32
        Spectrogram to apply augmentations to.
    augmentations : list of augmentation class instances from util.signal_augmentations.*.
        List of augmentations of which only the spectrogram ones will get applied to the samples.

    Returns
    -------
    Tensor of type float32
        The augmented spectrogram
    """
    if augmentations is not None:
        for augmentation in augmentations:
            if isinstance(augmentation, SpectrogramAugmentation):
                spectrogram = augmentation.apply_with_probability(spectrogram, clock=clock, seed=seed)
    return spectrogram


class Add(FeaturesAugmentation):
    """See "Add augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, stddev=0.0):
        super(Add, self).__init__(p)
        self.stddev = float_range(stddev)

    def apply(self, features, clock=0, seed=0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        stddev = tf_pick_value_from_range(self.stddev, clock=clock, seed=seed)
        return features + tf.random.normal(mean=0.0, stddev=self.stddev, shape=tf.shape(features))


class Multiply(FeaturesAugmentation):
    """See "Multiply augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, stddev=0.0):
        super(Multiply, self).__init__(p)
        self.stddev = float_range(stddev)

    def apply(self, features, clock=0, seed=0):
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        stddev = tf_pick_value_from_range(self.stddev, clock=clock, seed=seed)
        return features * tf.random.normal(mean=1.0, stddev=stddev, shape=tf.shape(features))


def apply_feature_augmentations(features, augmentations, clock=0, seed=0):
    """
    Augments training sample features with feature augmentations from passed augmentation list.

    Parameters
    ----------
    features : Tensor of type float32
        Features to apply augmentations to.
    augmentations : list of augmentation class instances from util.signal_augmentations.*.
        List of augmentations of which only the feature ones will get applied to the samples.

    Returns
    -------
    Tensor of type float32
        The augmented features
    """
    if augmentations is not None:
        for augmentation in augmentations:
            if isinstance(augmentation, FeaturesAugmentation):
                features = augmentation.apply_with_probability(features, clock=clock, seed=seed)
    return features
