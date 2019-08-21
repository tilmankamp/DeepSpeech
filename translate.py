#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import itertools
import json

from multiprocessing import cpu_count

import numpy as np
import progressbar
import tensorflow as tf

from ds_ctcdecoder import ctc_beam_search_decoder_batch, Scorer
from six.moves import zip

from util.config import Config, initialize_globals
from util.feeding import split_to_dataset
from util.flags import create_flags, FLAGS
from util.logging import log_error, log_progress, create_progressbar


def translate(speech_file, create_model, try_loading):
    scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta, FLAGS.lm_binary_path, FLAGS.lm_trie_path, Config.alphabet)

    split_set = split_to_dataset(speech_file, batch_size=FLAGS.test_batch_size)
    iterator = tf.data.Iterator.from_structure(split_set.output_types,
                                               split_set.output_shapes,
                                               output_classes=split_set.output_classes)
    init_op = iterator.make_initializer(split_set)

    batch_time_start, batch_time_end, batch = iterator.get_next()
    batch_x, batch_x_len = batch

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

        predictions = []

        bar = create_progressbar(prefix='Translating | ',
                                 widgets=['Steps: ', progressbar.Counter(), ' | ', progressbar.Timer()]).start()
        log_progress('Translating...')

        step_count = 0

        # Initialize iterator to the appropriate dataset
        session.run(init_op)

        # First pass, compute losses and transposed logits for decoding
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
            predictions.extend(decoded)
            for d in decoded:
                print('', d)

            step_count += 1
            bar.update(step_count)

        bar.finish()

        return predictions


def main(_):
    initialize_globals()

    if not FLAGS.speech_file:
        log_error('You need to specify what files to use for evaluation via '
                  'the --speech_file flag.')
        exit(1)

    from DeepSpeech import create_model, try_loading # pylint: disable=cyclic-import
    samples = translate(FLAGS.speech_file, create_model, try_loading)

    if FLAGS.output_file:
        # Save decoded tuples as JSON, converting NumPy floats to Python floats
        json.dump(samples, open(FLAGS.output_file, 'w'), default=float)


if __name__ == '__main__':
    create_flags()
    tf.app.flags.DEFINE_string('speech_file', '', 'path to a WAV file with speech to translate')
    tf.app.flags.DEFINE_string('output_file', '', 'path to a JSON file to save all translated phrases')
    tf.app.run(main)
