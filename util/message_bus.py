import tensorflow as tf
import pickle
from threading import Thread, Lock, Event
import Queue

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
        self._to_send = Queue.Queue()

    def _call(self, function, args):
        try:
            fun = getattr(self, function, *args)
        except AttributeError:
            return
        return fun(*args)

    def send(self, session):
        while self._run:
            enqueue, feed_dict = self._to_send.get()
            content = session.run(enqueue, feed_dict=feed_dict)

    def _receive(self, session, caller, is_response, response_id, function, items):
        if is_response:
            print('[%s] Response From: %s' % (self.id, caller))
            callback = self._waiting_for_response[response_id]
            if callable(callback):
                callback(items)
                del self._waiting_for_response[response_id]
            else:
                self._waiting_for_response[response_id] = items
                callback.set()
        else:
            results = self._call(function, items)
            if response_id > 0:
                content = pickle.dumps((self.id, True, response_id, function, results))
                queue = self._outbound[caller]
                self._to_send.put((queue[0], { self._ph_content: content }))

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

    def call_async(self, job, task, function, callback, *args):
        if job == self.job and task == self.task:
            results = self._call(function, args)
            callback(results)
            return -1
        response_id = 0
        if callback:
            with self._lock:
                response_id = self._response_id_counter = self._response_id_counter + 1
                self._waiting_for_response[response_id] = callback
        content = pickle.dumps((self.id, False, response_id, function, args))
        queue = self._outbound['MBQ-%s-%d' % (job, task)]
        self._to_send.put((queue[0], { self._ph_content: content }))
        return response_id

    def call(self, job, task, function, *args):
        if job == self.job and task == self.task:
            return self._call(function, args)
        event = Event()
        response_id = self.call_async(job, task, function, event, *args)
        event.wait()
        result = self._waiting_for_response[response_id]
        del self._waiting_for_response[response_id]
        return result
