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

from util.collections import collection_from_file


def play_collection():
    col = collection_from_file(CLI_ARGS.collection)
    while True:
        index = random.randrange(0, len(col))
        sample = col[index]
        print('Sample index: {} - transcript: "{}"'.format(index, sample.transcript))
        rate, channels, width = sample.audio_format
        wave_obj = simpleaudio.WaveObject(sample.audio_data, channels, width, rate)
        play_obj = wave_obj.play()
        play_obj.wait_done()


def handle_args():
    parser = argparse.ArgumentParser(description='Tool for playing samples from Sample Databases (.sdb files) '
                                                 'and DeepSpeech .csv files')
    parser.add_argument('collection', help='Sample DB or CSV file to play samples from')
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
