
import sys

from collections import Counter
from multiprocessing import Process, Queue

STOP_TOKEN = False
ALPHABET = 'abcdefghijklmnopqrstuvwxyzäöüß'
NUM_WORKERS = 10
MAX_CHUNKS = 10 * NUM_WORKERS
TOP_WORDS = 500000
PRUNE_FACTOR = 10
LINES_PER_CHUNK = 100000
MAX_KEYS = 100000


def count_words(cid, input_lines, resulting_lines, counters):
    counter = Counter()
    while True:
        lines = input_lines.get()
        if len(counter.keys()) > MAX_KEYS or lines == STOP_TOKEN:
            counters.put(counter)
            counter = Counter()
        if lines == STOP_TOKEN:
            return
        new_lines = []
        for line in lines:
            line_lower = line.lower()
            new_lines.append(line_lower)
            for w in line_lower.split():
                cw = ''
                for c in w:
                    if c in ALPHABET:
                        cw += c
                if len(cw) > 0:
                    counter[cw] += 1
        resulting_lines.put(new_lines)


def aggregate_counters(vocab_filename, counters):
    overall_counter = Counter()
    while True:
        counter = counters.get()
        if counter == STOP_TOKEN:
            with open(sys.argv[1], 'w') as vocab_file:
                vocab_file.write('\n'.join(str(word) for word, count in overall_counter.most_common(TOP_WORDS)))
            return
        overall_counter += counter
        if len(overall_counter.keys()) > PRUNE_FACTOR * TOP_WORDS:
            overall_counter = Counter(overall_counter.most_common(TOP_WORDS))


def write_lines(resulting_lines):
    while True:
        lines = resulting_lines.get()
        if lines == STOP_TOKEN:
            return
        print('\n'.join(lines))


def main():
    vocab_filename = sys.argv[1]
    input_lines = Queue(MAX_CHUNKS)
    resulting_lines = Queue(MAX_CHUNKS)
    counters = Queue(NUM_WORKERS)

    writer_process = Process(target=write_lines, args=(resulting_lines,))
    writer_process.start()

    aggregator_process = Process(target=aggregate_counters, args=(vocab_filename, counters))
    aggregator_process.start()

    counter_processes = map(lambda index: Process(target=count_words,
                                                  args=(vocab_filename + '_' + str(index),
                                                        input_lines,
                                                        resulting_lines,
                                                        counters)),
                            range(NUM_WORKERS))
    for p in counter_processes:
        p.start()

    lines = []
    for line in sys.stdin:
        lines.append(line)
        if len(lines) >= LINES_PER_CHUNK:
            input_lines.put(lines)
            lines = []
    input_lines.put(lines)

    for _ in counter_processes:
        input_lines.put(STOP_TOKEN)
    for p in counter_processes:
        p.join()

    counters.put(STOP_TOKEN)
    aggregator_process.join()

    resulting_lines.put(STOP_TOKEN)
    writer_process.join()


if __name__ == '__main__':
    main()
