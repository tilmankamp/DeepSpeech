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

import argparse
import progressbar

from util.downloader import SIMPLE_BAR
from util.helpers import parse_file_size
from util.audio import convert_samples, AUDIO_TYPE_OPUS
from util.collections import samples_from_files, DirectSDBWriter, SortingSDBWriter


def add_samples(sample_sink):
    samples = samples_from_files(CLI_ARGS.sources)
    num_workers = os.cpu_count() * CLI_ARGS.load_factor if CLI_ARGS.load_factor > 0 else 1
    bar = progressbar.ProgressBar(max_value=len(samples), widgets=SIMPLE_BAR)
    for sample in bar(convert_samples(samples, audio_type=AUDIO_TYPE_OPUS, processes=num_workers)):
        sample_sink.add(sample)


def build_sdb():
    if CLI_ARGS.sort:
        with SortingSDBWriter(CLI_ARGS.target,
                              tmp_sdb_filename=CLI_ARGS.sort_tmp_file,
                              cache_size=parse_file_size(CLI_ARGS.sort_cache_size)) as sdb_writer:
            add_samples(sdb_writer)
    else:
        with DirectSDBWriter(CLI_ARGS.target) as sdb_writer:
            add_samples(sdb_writer)


def handle_args():
    parser = argparse.ArgumentParser(description='Tool for building Sample Databases (.sdb files) '
                                                 'from DeepSpeech CSV files')
    parser.add_argument('--load_factor', type=int, default=1,
                        help='CPU-multiplier for the number of parallel workers - 0 for 1 worker')
    parser.add_argument('--sort', action='store_true', help='Force sample sorting by durations '
                                                            '(assumes SDB sources unsorted)')
    parser.add_argument('--sort_tmp_file', default=None, help='Overrides default tmp_file (target + ".tmp") '
                                                              'for sorting through --sort option')
    parser.add_argument('--sort_cache_size', default='1GB', help='Cache (bucket) size for binary audio data '
                                                                 'for sorting through --sort option')
    parser.add_argument('sources', nargs='+', help='Source collections (.csv and Sample DB files)')
    parser.add_argument('target', help='Sample DB to create')
    return parser.parse_args()


if __name__ == "__main__":
    CLI_ARGS = handle_args()
    build_sdb()
