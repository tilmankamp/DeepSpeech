
import threading
import numpy as np
import tensorflow as tf

from util.text import ctc_label_dense_to_sparse

class SwitchableDataSet(object):
    def __init__(self, data_sets):
        self._data_sets = data_sets
        self._sets = [data_sets.train, data_sets.dev, data_sets.test]
        assert len(set([s._batch_size for s in self._sets])) == 1
        self._batch_size = data_sets.train._batch_size
        self._queues = [s._example_queue for s in self._sets]
        self._queue_selector = tf.placeholder(tf.int32)
        self._queue = tf.QueueBase.from_list(self._queue_selector, self._queues)
        self._close_op = self._queue.close(cancel_pending_enqueues=True)

    def set_data_set(self, feed_dict, data_set):
        index = self._sets.index(data_set)
        assert index >= 0
        feed_dict[self._queue_selector] = index

    def start_queue_threads(self, session, coord):
        batch_threads = []
        for s in self._sets:
            batch_threads += s.start_queue_threads(session, coord)
        return batch_threads

    def close_queue(self, session):
        session.run(self._close_op, feed_dict={ self._queue_selector: 0 })
        for s in self._sets:
            s.close_queue(session)

    def next_batch(self):
        source, source_lengths, target, target_lengths = self._queue.dequeue_many(self._batch_size)
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, self._batch_size)
        return source, source_lengths, sparse_labels
