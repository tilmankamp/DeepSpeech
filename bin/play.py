#!/usr/bin/env python
'''
Builds Sample Databases (.sdb files)
Use "python3 build_sdb.py -h" for help
'''
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import random
import argparse
import simpleaudio

from util.collections import samples_from_file
from util.audio import AUDIO_TYPE_PCM


def play_sample(samples, index):
    if index < 0:
        index = len(samples) + index
    if CLI_ARGS.random:
        index = random.randint(0, len(samples))
    elif index >= len(samples):
        print('No sample with index {}'.format(CLI_ARGS.start))
        sys.exit(1)
    sample = samples[index]
    print('Sample "{}"'.format(sample.id))
    print('  "{}"'.format(sample.transcript))
    sample.convert(AUDIO_TYPE_PCM)
    rate, channels, width = sample.audio_format
    wave_obj = simpleaudio.WaveObject(sample.audio, channels, width, rate)
    play_obj = wave_obj.play()
    play_obj.wait_done()


def play_collection():
    samples = samples_from_file(CLI_ARGS.collection, buffering=0)
    played = 0
    index = CLI_ARGS.start
    while True:
        if 0 <= CLI_ARGS.number <= played:
            return
        play_sample(samples, index)
        played += 1
        index = (index + 1) % len(samples)


def handle_args():
    parser = argparse.ArgumentParser(description='Tool for playing samples from Sample Databases (.sdb files) '
                                                 'and DeepSpeech .csv files')
    parser.add_argument('collection', help='Sample DB or CSV file to play samples from')
    parser.add_argument('--start', type=int, default=0,
                        help='Sample index to start at (negative numbers are relative to the end of the collection)')
    parser.add_argument('--number', type=int, default=-1, help='Number of samples to play (-1 for endless)')
    parser.add_argument('--random', action='store_true', help='If samples should be played in random order')
    return parser.parse_args()


if __name__ == "__main__":
    CLI_ARGS = handle_args()
    try:
        play_collection()
    except KeyboardInterrupt:
        print(' Stopped')
        sys.exit(0)
    except RuntimeError as runtime_error:
        print(runtime_error)
        sys.exit(1)
