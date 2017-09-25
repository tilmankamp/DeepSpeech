import tensorflow as tf
import pickle
from threading import Thread, Lock, Event
import Queue
from util.log import Logger

log = Logger('Message Bus')

class MessageBusClient(object):
    def __init__(self, cluster_spec, job, task):
        self.cluster_spec = cluster_spec
        self.job = job
        self.task = task
        self.id = 'MBQ-%s-%d' % (job, task)

        self._ph_content = tf.placeholder(tf.string, [])

        inbound_queue = tf.FIFOQueue(1000, [tf.string], name=self.id, shared_name=self.id)
        self._inbound_dequeue = inbound_queue.dequeue()
        self._inbound_close = inbound_queue.close(cancel_pending_enqueues=True)

        self._outbound_enqueues = {}
        self._outbound_closes = []
        for j in cluster_spec.jobs:
            for t in cluster_spec.task_indices(j):
                id = 'MBQ-%s-%d' % (j, t)
                if id != self.id:
                    queue = tf.FIFOQueue(1000, [tf.string], name=id, shared_name=id)
                    self._outbound_enqueues[id] = queue.enqueue([self._ph_content])
                    self._outbound_closes.append(queue.close(cancel_pending_enqueues=True))

        self._waiting_for_response = {}
        self._response_id_counter = 0
        self._response_id_lock = Lock()
        self._to_send = Queue.Queue()

    def _call(self, function, args):
        try:
            fun = getattr(self, function, *args)
        except AttributeError:
            log.warn('Function not found: %s, arguments: %r' % (function, args))
            return
        return fun(*args)

    def send(self, session, coord):
        while not coord.should_stop():
            enqueue, feed_dict = self._to_send.get()
            try:
                content = session.run(enqueue, feed_dict=feed_dict)
            except (tf.errors.CancelledError, tf.errors.OutOfRangeError):
                return

    def _receive(self, session, caller, is_response, response_id, function, items):
        if is_response:
            log.traffic('Response from: %s, function: %s, return values: %r' % (caller, function, items))
            callback = self._waiting_for_response[response_id]
            if callable(callback):
                callback(items)
                del self._waiting_for_response[response_id]
            else:
                self._waiting_for_response[response_id] = items
                callback.set()
        else:
            log.traffic('Call from: %s, function: %s, arguments: %r' % (caller, function, items))
            results = self._call(function, items)
            if response_id > 0:
                content = pickle.dumps((self.id, True, response_id, function, results))
                enqueue = self._outbound_enqueues[caller]
                self._to_send.put((enqueue, { self._ph_content: content }))

    def receive(self, session, coord):
        while not coord.should_stop():
            try:
                content = session.run(self._inbound_dequeue)
            except (tf.errors.CancelledError, tf.errors.OutOfRangeError):
                return
            args = (session,) + pickle.loads(content)
            Thread(target=self._receive, args=args).start()

    def start_queue_threads(self, session, coord):
        threads = []
        for routine in [self.send, self.receive]:
            thread = Thread(target=routine, args=(session,coord))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        log.debug('Started queue threads')
        return threads

    def close_queues(self, session):
        session.run(self._inbound_close)
        for close in self._outbound_closes:
            session.run(close)
        log.debug('Closed queue threads')

    def call_async(self, job, task, function, callback, *args):
        id = 'MBQ-%s-%d' % (job, task)
        if self.id == id:
            log.traffic('Local call of function: %s, arguments: %r' % (function, args))
            results = self._call(function, args)
            if callback:
                callback(results)
            return -1
        log.traffic('Calling worker: %s, function: %s, arguments: %r' % (id, function, args))
        response_id = 0
        if callback:
            with self._response_id_lock:
                response_id = self._response_id_counter = self._response_id_counter + 1
                self._waiting_for_response[response_id] = callback
        content = pickle.dumps((self.id, False, response_id, function, args))
        enqueue = self._outbound_enqueues[id]
        self._to_send.put((enqueue, { self._ph_content: content }))
        return response_id

    def call(self, job, task, function, *args):
        if job == self.job and task == self.task:
            log.traffic('Local call of function: %s, arguments: %r' % (function, args))
            return self._call(function, args)
        event = Event()
        response_id = self.call_async(job, task, function, event, *args)
        event.wait()
        result = self._waiting_for_response[response_id]
        del self._waiting_for_response[response_id]
        return result
