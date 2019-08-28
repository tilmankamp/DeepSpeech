#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import json
import errno
import progressbar
import tensorflow as tf

from multiprocessing import cpu_count
from ds_ctcdecoder import ctc_beam_search_decoder_batch, Scorer
from util.config import Config, initialize_globals
from util.feeding import split_audio_file
from util.flags import create_flags, FLAGS
from util.logging import log_error, log_info, log_progress, create_progressbar

def split_audio_file_flags(audio_file):
    return split_audio_file(audio_file,
                            batch_size=FLAGS.test_batch_size,
                            aggressiveness=FLAGS.vad_aggressiveness,
                            outlier_fraction=FLAGS.outlier_fraction,
                            outlier_batch_size=FLAGS.outlier_batch_size,
                            cache_path=FLAGS.feature_cache)


def transcribe(path_pairs, create_model, try_loading):
    scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta, FLAGS.lm_binary_path, FLAGS.lm_trie_path, Config.alphabet)

    audio_path, log_path = next(path_pairs, (None, None))
    if audio_path is None:
        return
    data_set, number_of_samples = split_audio_file_flags(audio_path)
    iterator = tf.data.Iterator.from_structure(data_set.output_types,
                                               data_set.output_shapes,
                                               output_classes=data_set.output_classes)

    batch_time_start, batch_time_end, batch_x, batch_x_len = iterator.get_next()

    # One rate per layer
    no_dropout = [None] * 6
    logits, _ = create_model(batch_x=batch_x, seq_length=batch_x_len, dropout=no_dropout)

    # Transpose to batch major and apply softmax for decoder
    transposed = tf.nn.softmax(tf.transpose(logits, [1, 0, 2]))

    tf.train.get_or_create_global_step()

    # Get number of accessible CPU cores for this process
    try:
        num_processes = cpu_count()
    except NotImplementedError:
        num_processes = 1

    # Create a saver using variables from the above newly created graph
    saver = tf.train.Saver()

    with tf.Session(config=Config.session_config) as session:
        # Restore variables from training checkpoint
        loaded = try_loading(session, saver, 'best_dev_checkpoint', 'best validation')
        if not loaded:
            loaded = try_loading(session, saver, 'checkpoint', 'most recent')
        if not loaded:
            log_error('Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(FLAGS.checkpoint_dir))
            exit(1)

        def run_transcription(data_set, audio_path, log_path, num_samples):
            transcriptions = []
            bar = create_progressbar(prefix='Transcribing file "{}" | '.format(audio_path),
                                     max_value=num_samples,
                                     widgets=[progressbar.AdaptiveETA()]).start()
            log_progress('Transcribing file "{}"...'.format(audio_path))

            # Initialize iterator to the appropriate dataset
            session.run(iterator.make_initializer(data_set))

            while True:
                try:
                    starts, ends, batch_logits, batch_lengths = session.run([batch_time_start,
                                                                             batch_time_end,
                                                                             transposed,
                                                                             batch_x_len])
                except tf.errors.OutOfRangeError:
                    break
                decoded = ctc_beam_search_decoder_batch(batch_logits,
                                                        batch_lengths,
                                                        Config.alphabet,
                                                        FLAGS.beam_width,
                                                        num_processes=num_processes,
                                                        scorer=scorer)
                decoded = list(d[0][1] for d in decoded)
                transcriptions.extend(zip(starts, ends, decoded))
                bar.update(len(transcriptions))

            bar.finish()
            transcriptions.sort(key=lambda t: t[0])
            transcriptions = list(map(lambda t: {
                'start': int(t[0]),
                'end': int(t[1]),
                'transcript': t[2]
            }, transcriptions))
            log_info('Writing transcription log to "{}"...'.format(log_path))
            json.dump(transcriptions, open(log_path, 'w'), default=float)

        while audio_path is not None:
            if data_set is None:
                data_set, number_of_samples = split_audio_file_flags(audio_path)
            run_transcription(data_set, audio_path, log_path, number_of_samples)
            data_set = None
            audio_path, log_path = next(path_pairs, (None, None))


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as ex:
        if not (ex.errno == errno.EEXIST and os.path.isdir(path)):
            log_error('Cannot create directory at ' + path)
            exit(1)


def main(_):
    initialize_globals()

    if not FLAGS.src:
        log_error('You need to specify which file or directory to transcribe via the --src flag.')
        exit(1)

    from DeepSpeech import create_model, try_loading # pylint: disable=cyclic-import

    src_path = os.path.abspath(FLAGS.src)
    dst_path = None
    formats = FLAGS.formats.split(',')

    def scan():
        for root, dirs, files in os.walk(src_path):
            created = False
            dst_parent = os.path.join(dst_path, os.path.relpath(root, src_path))
            for file in files:
                base, ext = os.path.splitext(file)
                if ext in formats:
                    tlog = os.path.join(dst_parent, base + '.tlog')
                    if not FLAGS.force and os.path.exists(tlog):
                        log_error('Transcription log "{}" already existing - not transcribing'.format(tlog))
                        continue
                    if not created:
                        if not os.path.exists(dst_parent):
                            mkdirs(dst_parent)
                        created = True
                    yield os.path.join(root, file), tlog

    def one_pair():
        yield src_path, dst_path

    if os.path.isdir(src_path):
        dst_path = os.path.abspath(FLAGS.dst) if FLAGS.dst else src_path
        if os.path.isdir(dst_path):
            transcribe(scan(), create_model, try_loading)
        else:
            log_error('Path in --src is directory, but path in --dst not')
            exit(1)
    elif os.path.isfile(src_path):
        if FLAGS.dst:
            dst_path = os.path.abspath(FLAGS.dst)
        else:
            base, ext = os.path.splitext(src_path)
            dst_path = base + '.tlog'
        if os.path.isfile(dst_path):
            if FLAGS.force:
                transcribe(one_pair(), create_model, try_loading)
            else:
                log_error('File in --dst already existing, requires --force for overwriting')
                exit(1)
        elif os.path.isdir(os.path.dirname(dst_path)):
            transcribe(one_pair(), create_model, try_loading)
        else:
            log_error('Cannot write to path in --dst')
            exit(1)
    else:
        log_error('Path in --src not existing')
        exit(1)


if __name__ == '__main__':
    create_flags()
    tf.app.flags.DEFINE_string('src', '', 'source path to an audio file or directory to recursively scan '
                                          'for audio files. If --dst not set, transcription logs (.tlog) will be '
                                          'written in-place using the source filenames with '
                                          'suffix ".tlog" instead of ".wav".')
    tf.app.flags.DEFINE_string('dst', '', 'path for writing the transcription log or logs (.tlog). '
                                          'If --src is a directory, this one also has to be a directory '
                                          'and the required sub-dir tree of --src will get replicated.')
    tf.app.flags.DEFINE_string('formats', '.wav', 'Comma separated list of audio file suffixes to scan for')
    tf.app.flags.DEFINE_boolean('force', False, 'Forces re-transcribing and overwriting of already existing '
                                                'transcription logs (.tlog)')
    tf.app.flags.DEFINE_integer('vad_aggressiveness', 3, 'How aggressive (0=lowest, 3=highest) the VAD should '
                                                         'split audio')
    tf.app.flags.DEFINE_float('outlier_fraction', 0, 'Fraction of samples per file that are to be considered '
                                                     'duration outliers')
    tf.app.flags.DEFINE_integer('outlier_batch_size', None, 'Batch size for duration outliers '
                                                            '(defaults to 50% of the normal batch size)')
    tf.app.run(main)
