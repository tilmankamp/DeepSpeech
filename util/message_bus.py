import tensorflow as tf
import pickle
from threading import Thread, Lock

class MessageBusClient():
    def __init__(self, cluster, job, task):
        self.cluster = cluster
        self.job = job
        self.task = task
        self.id = 'MBQ-%s-%d' % (job, task)

        self._ph_content = tf.placeholder(tf.string, [])

        inbound_queue = tf.FIFOQueue(1000, [tf.string], name=self.id, shared_name=self.id)
        self._inbound_dequeue = inbound_queue.dequeue()
        self._inbound_close = inbound_queue.close(cancel_pending_enqueues=True)

        self._outbound = {}
        for j in cluster.jobs:
            for t in cluster.task_indices(j):
                id = 'MBQ-%s-%d' % (j, t)
                if id != self.id:
                    queue = tf.FIFOQueue(1000, [tf.string], name=id, shared_name=id)
                    enqueue = queue.enqueue([self._ph_content])
                    close = queue.close(cancel_pending_enqueues=True)
                    self._outbound[id] = (enqueue, close)

        self._waiting_for_response = {}
        self._response_id_counter = 0
        self._lock = Lock()
        self._to_send = []

    def send(self, session):
        while self._run:
            with self._lock:
                if len(self._to_send) > 0:
                    enqueue, feed_dict = self._to_send.pop(0)
                    content = session.run(enqueue, feed_dict=feed_dict)

    def _receive(self, session, caller, is_response, response_id, function, items):
        if is_response:
            print('[%s] Response From: %s' % (self.id, caller))
            callback = self._waiting_for_response[response_id]
            if callback:
                if not callback(caller, function, *items):
                    del self._waiting_for_response[response_id]
        else:
            try:
                method = getattr(self, function)
            except AttributeError:
                return
            results = method(caller, *items)
            if response_id > 0:
                if not isinstance(results, tuple):
                    results = (results,)
                content = pickle.dumps((caller, True, response_id, function, results))
                queue = self._outbound[caller]
                self._to_send.append((queue[0], { self._ph_content: content }))

    def receive(self, session):
        while self._run:
            content = session.run(self._inbound_dequeue)
            self._receive(session, *pickle.loads(content))

    def start_listening(self, session):
        self._run = True
        for routine in [self.send, self.receive]:
            self._thread = Thread(target=routine, args=(session,))
            self._thread.daemon = True
            self._thread.start()

    def stop_listening(self):
        self._run = False
        self._thread.join()

    def call_function(self, session, job, task, function, callback, *args):
        response_id = 0
        if callback:
            with self._lock:
                response_id = self._response_id_counter = self._response_id_counter + 1
                self._waiting_for_response[response_id] = callback
        content = pickle.dumps((self.id, False, response_id, function, args))
        queue = self._outbound['MBQ-%s-%d' % (job, task)]
        self._to_send.append((queue[0], { self._ph_content: content }))

    def call_method(self, session, job, task, method, *args):
        self.call_function(session, job, task, method, None, *args)

