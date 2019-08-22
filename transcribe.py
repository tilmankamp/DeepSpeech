#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import json
import progressbar
import tensorflow as tf

from threading import Thread
from multiprocessing import cpu_count
from ds_ctcdecoder import ctc_beam_search_decoder_batch, Scorer
from util.config import Config, initialize_globals
from util.feeding import split_to_dataset
from util.flags import create_flags, FLAGS
from util.logging import log_error, log_progress, create_progressbar


def transcribe(wav_files, create_model, try_loading):
    scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta, FLAGS.lm_binary_path, FLAGS.lm_trie_path, Config.alphabet)

    current_file = wav_files[0]
    current_set, current_audio_length = split_to_dataset(current_file, batch_size=FLAGS.test_batch_size)
    iterator = tf.data.Iterator.from_structure(current_set.output_types,
                                               current_set.output_shapes,
                                               output_classes=current_set.output_classes)

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

        file_results = []

        def run_transcription(data_set, wav_file, audio_length):
            transcriptions = []
            bar = create_progressbar(prefix='Transcribing file "{}" | '.format(wav_file),
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
            transcriptions.sort(key=lambda t: t[1])
            return transcriptions

        run_transcription(current_set, current_file, current_audio_length)
        for current_file in wav_files[1:]:
            current_set, current_audio_length = split_to_dataset(current_file, batch_size=FLAGS.test_batch_size)
            file_transcriptions = run_transcription(current_set, current_file, current_audio_length)
            file_results.extend(file_transcriptions)
        return file_results

def main(_):
    initialize_globals()

    if not FLAGS.transcribe:
        log_error('You need to specify what files to use for transcription via '
                  'the --transcribe flag.')
        exit(1)

    from DeepSpeech import create_model, try_loading # pylint: disable=cyclic-import
    file_results = transcribe(FLAGS.transcribe.split(','), create_model, try_loading)

    if FLAGS.output_file:
        # Save decoded tuples as JSON, converting NumPy floats to Python floats
        json.dump(file_results, open(FLAGS.output_file, 'w'), default=float)


if __name__ == '__main__':
    create_flags()
    tf.app.flags.DEFINE_string('transcribe', '', 'path to a WAV or text file with speech '
                                                 'or paths to speech files to transcribe')
    tf.app.flags.DEFINE_string('output_file', '', 'path to a JSON file to save all transcribed phrases')
    tf.app.run(main)
