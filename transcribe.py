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
from util.feeding import split_to_dataset
from util.flags import create_flags, FLAGS
from util.logging import log_error, log_progress, create_progressbar


def transcribe(path_pairs, create_model, try_loading):
    scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta, FLAGS.lm_binary_path, FLAGS.lm_trie_path, Config.alphabet)

    wav_path, log_path = next(path_pairs)
    wav_set, audio_length = split_to_dataset(wav_path, batch_size=FLAGS.test_batch_size)
    iterator = tf.data.Iterator.from_structure(wav_set.output_types,
                                               wav_set.output_shapes,
                                               output_classes=wav_set.output_classes)

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

        def run_transcription(data_set, wav_path, log_path, audio_length):
            transcriptions = []
            bar = create_progressbar(prefix='Transcribing file "{}" | '.format(wav_path),
                                     max_value=audio_length,
                                     widgets=[progressbar.ETA()]).start()
            log_progress('Transcribing...')
            audio_offset = 0

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

                decoded = ctc_beam_search_decoder_batch(batch_logits, batch_lengths, Config.alphabet, FLAGS.beam_width,
                                                        num_processes=num_processes, scorer=scorer)
                decoded = list(d[0][1] for d in decoded)
                transcriptions.extend(zip(starts, ends, decoded))
                audio_offset = max(ends.max(), audio_offset)
                bar.update(audio_offset)

            bar.finish()
            transcriptions.sort(key=lambda t: t[0])
            json.dump(transcriptions, open(log_path, 'w'), default=float)

        while wav_path is not None:
            if wav_set is None:
                wav_set, audio_length = split_to_dataset(wav_path, batch_size=FLAGS.test_batch_size)
            run_transcription(wav_set, wav_path, log_path, audio_length)
            wav_set = None
            wav_path, log_path = next(path_pairs, (None, None))


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as ex:
        if not (ex.errno == errno.EEXIST and os.path.isdir(path)):
            raise ex


def main(_):
    initialize_globals()

    if not FLAGS.src:
        log_error('You need to specify which files to transcribe via the --src flag.')
        exit(1)

    from DeepSpeech import create_model, try_loading # pylint: disable=cyclic-import

    src_path = os.path.abspath(FLAGS.src)
    dst_path = None

    def scan():
        for root, dirs, files in os.walk(src_path):
            created = False
            dst_parent = os.path.join(dst_path, os.path.relpath(root, src_path))
            for file in files:
                base, ext = os.path.splitext(file)
                if ext == '.wav':
                    if not created:
                        if not os.path.exists(dst_parent):
                            mkdirs(dst_parent)
                        created = True
                    yield os.path.join(root, file), os.path.join(dst_parent, base + '.tlog')

    def one_pair():
        yield src_path, dst_path

    if os.path.isdir(src_path):
        dst_path = os.path.abspath(FLAGS.dst) if FLAGS.dst else src_path
        if os.path.isdir(dst_path):
            transcribe(scan(), create_model, try_loading)
        else:
            log_error('If --src specifies a directory to scan, --dst also has to point to an existing directory.')
            exit(1)
    elif os.path.isfile(src_path):
        if FLAGS.dst:
            dst_path = os.path.abspath(FLAGS.dst)
        else:
            base, ext = os.path.splitext(src_path)
            dst_path = base + '.tlog'
        if os.path.isfile(dst_path) or os.path.isdir(os.path.dirname(dst_path)):
            transcribe(one_pair(), create_model, try_loading)
        else:
            log_error('Cannot write to path in --dst')
            exit(1)
    else:
        log_error('--src neither pointing to an existing file nor to an existing directory')
        exit(1)


if __name__ == '__main__':
    create_flags()
    tf.app.flags.DEFINE_string('src', '', 'source path to a WAV file or directory to scan for WAV files. '
                                          'If --dst not set, transcripts will be written in-place '
                                          'using the source filenames with suffix ".tlog" instead of ".wav".')
    tf.app.flags.DEFINE_string('dst', '', 'target path to a transcript file or a directory to store transcripts. '
                                          'If --src is a directory, this one also has to be a directory '
                                          'and the required sub-dir tree of --src will get replicated.')
    tf.app.flags.DEFINE_boolean('force', False, 'Forces re-transcribing and overwriting '
                                                'of already existing transcription files')
    tf.app.run(main)
