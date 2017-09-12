import time
import sys
import tensorflow as tf
from threading import Thread

# start this script two times as "python rq.py 0" and "python rq.py 1"
worker_index = int(sys.argv[1])
cluster = tf.train.ClusterSpec({ 'worker': ['localhost:5656', 'localhost:5757'] })
server = tf.train.Server(cluster, job_name='worker', task_index=worker_index)

# two queues to be fed by one worker each
q0 = tf.FIFOQueue(1, [tf.int32], name='Q0', shared_name='Q0')
q1 = tf.FIFOQueue(1, [tf.int32], name='Q1', shared_name='Q1')
ph_number = tf.placeholder(tf.int32, [])
enqueues = [q0.enqueue([ph_number]), q1.enqueue([ph_number])]
with tf.device('/job:worker/task:0/cpu:0'):
    q0p = q0.dequeue() * 2
with tf.device('/job:worker/task:1/cpu:0'):
    q1p = q1.dequeue() * 200
result = q0p + q1p

def _feeding(session):
    counter = 0
    # pick enqueue op for our worker/queue
    enqueue = enqueues[worker_index]
    while True:
        counter += 1
        session.run(enqueue, feed_dict={ ph_number: counter })
        print('Enqueued %d' % counter)

with tf.Session(server.target) as session:
    # feeding two queues - one per worker
    Thread(target=_feeding, args=(session,)).start()
    while True:
        time.sleep(1)
        # in-graph approach - all evaluation only by chief worker
        if worker_index == 0:
            print(session.run(result))