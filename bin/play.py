#!/usr/bin/env python
"""
Tool for playing samples from Sample Databases (SDB files) and DeepSpeech CSV files
Use "python3 build_sdb.py -h" for help
"""

import argparse
import random
import sys

from deepspeech_training.util.audio import AUDIO_TYPE_PCM
from deepspeech_training.util.sample_collections import LabeledSample, samples_from_source, prepare_samples


def get_samples_in_play_order():
    samples = samples_from_source(CLI_ARGS.collection, buffering=0)
    played = 0
    index = CLI_ARGS.start
    while True:
        if 0 <= CLI_ARGS.number <= played:
            return
        if CLI_ARGS.random:
            yield samples[random.randint(0, len(samples) - 1)]
        elif index < 0:
            yield samples[len(samples) + index]
        elif index >= len(samples):
            print("No sample with index {}".format(CLI_ARGS.start))
            sys.exit(1)
        else:
            yield samples[index]
        played += 1
        index = (index + 1) % len(samples)


def play_collection():
    samples = get_samples_in_play_order()
    samples = prepare_samples(samples,
                              audio_type=AUDIO_TYPE_PCM,
                              augmentation_specs=CLI_ARGS.augment,
                              process_ahead=0,
                              fixed_clock=CLI_ARGS.clock)
    for sample in samples:
        print('Sample "{}"'.format(sample.sample_id))
        if isinstance(sample, LabeledSample):
            print('  "{}"'.format(sample.transcript))
        rate, channels, width = sample.audio_format
        wave_obj = simpleaudio.WaveObject(sample.audio, channels, width, rate)
        play_obj = wave_obj.play()
        play_obj.wait_done()


def handle_args():
    parser = argparse.ArgumentParser(
        description="Tool for playing samples from Sample Databases (SDB files) "
        "and DeepSpeech CSV files"
    )
    parser.add_argument("collection", help="Sample DB or CSV file to play samples from")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Sample index to start at (negative numbers are relative to the end of the collection)",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=-1,
        help="Number of samples to play (-1 for endless)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="If samples should be played in random order",
    )
    parser.add_argument(
        "--augment",
        action='append',
        help="Add an augmentation operation",
    )
    parser.add_argument(
        "--clock",
        type=float,
        default=0.5,
        help="Simulates clock value used for augmentations during training."
             "Ranges from 0.0 (representing parameter start values) to"
             "1.0 (representing parameter end values)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        import simpleaudio
    except ModuleNotFoundError:
        print('play.py requires Python package "simpleaudio"')
        sys.exit(1)
    CLI_ARGS = handle_args()
    try:
        play_collection()
    except KeyboardInterrupt:
        print(" Stopped")
        sys.exit(0)
