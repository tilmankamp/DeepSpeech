from util.message_bus import MessageBusClient
import time
import sys
import tensorflow as tf

class MBTestClient(MessageBusClient):
    def __init__(self, cluster, job, task):
        MessageBusClient.__init__(self, cluster, job, task)

    def test(self, data):
        print('>>>>> %r <<<<<' % data)
        return 'hola - I am worker %d' % self.task

task_index = int(sys.argv[1])
cluster = tf.train.ClusterSpec({'worker': ['localhost:5656', 'localhost:5757']})
server = tf.train.Server(cluster, job_name='worker', task_index=task_index)

with tf.Session(server.target) as session:
    mbc = MBTestClient(cluster, 'worker', task_index)
    coord = tf.train.Coordinator()
    mbc.start_queue_threads(session, coord)
    counter = 0
    while True:
        time.sleep(0)
        counter = counter + 1
        print('Calling! %d' % counter)
        result = mbc.call('worker', int(sys.argv[2]), 'test', sys.argv[3])
        print(result)
