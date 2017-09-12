#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys
from util.gpu import get_available_gpus

log_level_index = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[log_level_index] if log_level_index > 0 and log_level_index < len(sys.argv) else '3'

# Determining memory state of each GPU before anything is loaded
memory_limits = [gpu.memory_limit for gpu in get_available_gpus()]
if len(memory_limits) == 0:
    memory_limits = [1000000000]

import datetime
import pickle
import shutil
import subprocess
import tensorflow as tf
import time
import inspect
import multiprocessing

from six.moves import zip, range, filter, urllib, BaseHTTPServer
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.tools import freeze_graph
from threading import Thread, Lock
from util.feeding import DataSet, ModelFeeder
from util.shared_lib import check_cupti
from util.spell import correction
from util.text import sparse_tensor_value_to_texts, wer
from util.message_bus import MessageBusClient
from urlparse import parse_qs
from xdg import BaseDirectory as xdg
import numpy as np


# Importer
# ========

tf.app.flags.DEFINE_integer ('threads_per_set',  0,           'concurrent sample loader threads per data set (train, dev, test) - default (0) is equal to the number of CPU cores (but at least 2)')
tf.app.flags.DEFINE_integer ('loader_buffer',    0,           'number of samples in the buffer that is used to pick batches from - default (0) is GPU memory of the biggest GPU divided by 10 million (but at least 100)')
tf.app.flags.DEFINE_integer ('queue_capacity',   100,         'capacity of feeding queues (number of samples) - defaults to 100')

# Files

tf.app.flags.DEFINE_string  ('train_files',      '',          'comma separated list of files specifying the dataset used for training. multiple files will get merged')
tf.app.flags.DEFINE_string  ('dev_files',        '',          'comma separated list of files specifying the dataset used for validation. multiple files will get merged')
tf.app.flags.DEFINE_string  ('test_files',       '',          'comma separated list of files specifying the dataset used for testing. multiple files will get merged')

# Sample window

tf.app.flags.DEFINE_integer ('limit_train',      0,           'maximum number of elements to use from train set - 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_dev',        0,           'maximum number of elements to use from validation set- 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_test',       0,           'maximum number of elements to use from test set- 0 means no limit')
tf.app.flags.DEFINE_integer ('skip_train',       0,           'number of elements to skip from the beginning of the train set')
tf.app.flags.DEFINE_integer ('skip_dev',         0,           'number of elements to skip from the beginning of the validation set')
tf.app.flags.DEFINE_integer ('skip_test',        0,           'number of elements to skip from the beginning of the test set')
tf.app.flags.DEFINE_boolean ('train_ascending',  True,        'process samples in train set in ascending (True) or descending (False) order - default True')
tf.app.flags.DEFINE_boolean ('dev_ascending',    True,        'process samples in validation set in ascending (True) or descending (False) order - default True')
tf.app.flags.DEFINE_boolean ('test_ascending',   True,        'process samples in test set in ascending (True) or descending (False) order - default True')

# Cluster configuration
# =====================

tf.app.flags.DEFINE_string  ('nodes',            '',          'comma separated list of hostname:port pairs of cluster worker nodes')
tf.app.flags.DEFINE_integer ('task_index',       0,           'index of this worker within the cluster - worker with index 0 will be the chief')

# Global Constants
# ================

tf.app.flags.DEFINE_integer ('gpu_allocation',   100,         'how much GPU memory should be allocated in percent')

tf.app.flags.DEFINE_boolean ('train',            True,        'wether to train the network')
tf.app.flags.DEFINE_boolean ('test',             True,        'wether to test the network')
tf.app.flags.DEFINE_integer ('epoch',            75,          'target epoch to train - if negative, the absolute number of additional epochs will be trained')

tf.app.flags.DEFINE_boolean ('use_warpctc',      False,       'wether to use GPU bound Warp-CTC')

tf.app.flags.DEFINE_float   ('dropout_rate',     0.05,        'dropout rate for feedforward layers')
tf.app.flags.DEFINE_float   ('dropout_rate2',    -1.0,        'dropout rate for layer 2 - defaults to dropout_rate')
tf.app.flags.DEFINE_float   ('dropout_rate3',    -1.0,        'dropout rate for layer 3 - defaults to dropout_rate')
tf.app.flags.DEFINE_float   ('dropout_rate4',    0.0,         'dropout rate for layer 4 - defaults to 0.0')
tf.app.flags.DEFINE_float   ('dropout_rate5',    0.0,         'dropout rate for layer 5 - defaults to 0.0')
tf.app.flags.DEFINE_float   ('dropout_rate6',    -1.0,        'dropout rate for layer 6 - defaults to dropout_rate')

tf.app.flags.DEFINE_float   ('relu_clip',        20.0,        'ReLU clipping value for non-recurrant layers')

# Adam optimizer (http://arxiv.org/abs/1412.6980) parameters

tf.app.flags.DEFINE_float   ('beta1',            0.9,         'beta 1 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('beta2',            0.999,       'beta 2 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('epsilon',          1e-8,        'epsilon parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('learning_rate',    0.001,       'learning rate of Adam optimizer')

# Step widths

tf.app.flags.DEFINE_integer ('display_step',     0,           'number of epochs we cycle through before displaying detailed progress - 0 means no progress display')
tf.app.flags.DEFINE_integer ('validation_step',  0,           'number of epochs we cycle through before validating the model - a detailed progress report is dependent on "--display_step" - 0 means no validation steps')

# Checkpointing

tf.app.flags.DEFINE_string  ('checkpoint_dir',   '',          'directory in which checkpoints are stored - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('checkpoint_secs',  600,         'checkpoint saving interval in seconds')
tf.app.flags.DEFINE_integer ('max_to_keep',      5,           'number of checkpoint files to keep - default value is 5')

# Exporting

tf.app.flags.DEFINE_string  ('export_dir',       '',          'directory in which exported models are stored - if omitted, the model won\'t get exported')
tf.app.flags.DEFINE_integer ('export_version',   1,           'version number of the exported model')
tf.app.flags.DEFINE_boolean ('remove_export',    False,       'wether to remove old exported models')

# Reporting

tf.app.flags.DEFINE_integer ('log_level',        1,           'log level for console logs - 0: INFO, 1: WARN, 2: ERROR, 3: FATAL')
tf.app.flags.DEFINE_boolean ('log_traffic',      False,       'log cluster transaction and traffic information during debug logging')

tf.app.flags.DEFINE_string  ('wer_log_pattern',  '',          'pattern for machine readable global logging of WER progress; has to contain %%s, %%s and %%f for the set name, the date and the float respectively; example: "GLOBAL LOG: logwer(\'12ade231\', %%s, %%s, %%f)" would result in some entry like "GLOBAL LOG: logwer(\'12ade231\', \'train\', \'2017-05-18T03:09:48-0700\', 0.05)"; if omitted (default), there will be no logging')

tf.app.flags.DEFINE_boolean ('log_placement',    False,       'wether to log device placement of the operators to the console')
tf.app.flags.DEFINE_integer ('report_count',     10,          'number of phrases with lowest WER (best matching) to print out during a WER report')

tf.app.flags.DEFINE_string  ('summary_dir',      '',          'target directory for TensorBoard summaries - defaults to directory "deepspeech/summaries" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('summary_secs',     0,           'interval in seconds for saving TensorBoard summaries - if 0, no summaries will be written')

# Geometry

tf.app.flags.DEFINE_integer ('n_hidden',         2048,        'layer width to use when initialising layers')

# Initialization

tf.app.flags.DEFINE_integer ('random_seed',      4567,        'default random seed that is used to initialize variables')
tf.app.flags.DEFINE_float   ('default_stddev',   0.046875,    'default standard deviation to use when initialising weights and biases')

# Early Stopping

tf.app.flags.DEFINE_boolean ('early_stop',       True,        'enable early stopping mechanism over validation dataset. Make sure that dev FLAG is enabled for this to work')

# This parameter is irrespective of the time taken by single epoch to complete and checkpoint saving intervals.
# It is possible that early stopping is triggered far after the best checkpoint is already replaced by checkpoint saving interval mechanism.
# One has to align the parameters (earlystop_nsteps, checkpoint_secs) accordingly as per the time taken by an epoch on different datasets.

tf.app.flags.DEFINE_integer ('earlystop_nsteps',  4,          'number of steps to consider for early stopping. Loss is not stored in the checkpoint so when checkpoint is revived it starts the loss calculation from start at that point')
tf.app.flags.DEFINE_float   ('estop_mean_thresh', 0.5,        'mean threshold for loss to determine the condition if early stopping is required')
tf.app.flags.DEFINE_float   ('estop_std_thresh',  0.5,        'standard deviation threshold for loss to determine the condition if early stopping is required')

for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
    tf.app.flags.DEFINE_float('%s_stddev' % var, None, 'standard deviation to use when initialising %s' % var)

FLAGS = tf.app.flags.FLAGS

def initialize_globals():

    # ps and worker hosts required for p2p cluster setup
    FLAGS.nodes = list(filter(len, FLAGS.nodes.split(',')))
    if len(FLAGS.nodes) == 0:
        FLAGS.nodes.append('localhost:3987')

    # Determine, if we are the chief worker
    global is_chief
    is_chief = FLAGS.task_index == 0

    # The absolute number of computing nodes - regardless of cluster or single mode
    global num_workers
    num_workers = max(1, len(FLAGS.nodes))

    # Create a cluster from the parameter server and worker hosts.
    global cluster
    cluster = tf.train.ClusterSpec({ 'worker': FLAGS.nodes })
    FLAGS.nodes

    # The device path base for this node
    global worker_device
    worker_device = '/job:worker/task:%d' % FLAGS.task_index

    # This node's CPU device
    global cpu_device
    cpu_device = worker_device + '/cpu:0'

    # This node's available GPU devices
    global available_devices
    available_devices = [worker_device + gpu.name for gpu in get_available_gpus()]

    # If there is no GPU available, we fall back to CPU based operation
    if 0 == len(available_devices):
        available_devices = [cpu_device]

    # By default we run as many sample loading threads per set as CPU cores
    # (as there is only one set active at a time)
    cpu_count = multiprocessing.cpu_count()
    if FLAGS.threads_per_set <= 0:
        FLAGS.threads_per_set = max(2, cpu_count)
    log_debug('Number of loader threads per data set (%d CPUs): %d' % (cpu_count, FLAGS.threads_per_set))

    # By default the loader buffer is the 10 million-th part of the biggest GPU's memory in bytes
    if FLAGS.loader_buffer <= 0:
        FLAGS.loader_buffer = max(100, max(memory_limits) // 10000000) if len(memory_limits) > 0 else 100
    log_debug('Number of samples in loader buffer (%d bytes GPU memory): %d' % (max(memory_limits), FLAGS.loader_buffer))

    # Set default dropout rates
    if FLAGS.dropout_rate2 < 0:
        FLAGS.dropout_rate2 = FLAGS.dropout_rate
    if FLAGS.dropout_rate3 < 0:
        FLAGS.dropout_rate3 = FLAGS.dropout_rate
    if FLAGS.dropout_rate6 < 0:
        FLAGS.dropout_rate6 = FLAGS.dropout_rate

    global dropout_rates
    dropout_rates = [ FLAGS.dropout_rate,
                      FLAGS.dropout_rate2,
                      FLAGS.dropout_rate3,
                      FLAGS.dropout_rate4,
                      FLAGS.dropout_rate5,
                      FLAGS.dropout_rate6 ]

    global no_dropout
    no_dropout = [ 0.0 ] * 6

    # Set default checkpoint dir
    if len(FLAGS.checkpoint_dir) == 0:
        FLAGS.checkpoint_dir = xdg.save_data_path(os.path.join('deepspeech','checkpoints'))

    # Set default summary dir
    if len(FLAGS.summary_dir) == 0:
        FLAGS.summary_dir = xdg.save_data_path(os.path.join('deepspeech','summaries'))

    # Standard session configuration that'll be used for all new sessions.
    global session_config
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=float(FLAGS.gpu_allocation) / 100.0)
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=FLAGS.log_placement,
                                    gpu_options=gpu_options)

    # Geometric Constants
    # ===================

    # For an explanation of the meaning of the geometric constants, please refer to
    # doc/Geometry.md

    # Number of MFCC features
    global n_input
    n_input = 26 # TODO: Determine this programatically from the sample rate

    # The number of frames in the context
    global n_context
    n_context = 9 # TODO: Determine the optimal value using a validation data set

    # Number of units in hidden layers
    global n_hidden
    n_hidden = FLAGS.n_hidden

    global n_hidden_1
    n_hidden_1 = n_hidden

    global n_hidden_2
    n_hidden_2 = n_hidden

    global n_hidden_5
    n_hidden_5 = n_hidden

    # LSTM cell state dimension
    global n_cell_dim
    n_cell_dim = n_hidden

    # The number of units in the third layer, which feeds in to the LSTM
    global n_hidden_3
    n_hidden_3 = 2 * n_cell_dim

    # The number of characters in the target language plus one
    global n_character
    n_character = 29 # TODO: Determine if this should be extended with other punctuation

    # The number of units in the sixth layer
    global n_hidden_6
    n_hidden_6 = n_character

    # Assign default values for standard deviation
    for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
        val = getattr(FLAGS, '%s_stddev' % var)
        if val is None:
            setattr(FLAGS, '%s_stddev' % var, FLAGS.default_stddev)


# Logging functions
# =================

def prefix_print(prefix, message):
    print(prefix + ('\n' + prefix).join(message.split('\n')))

def log_debug(message):
    if FLAGS.log_level == 0:
        prefix_print('D ', str(message))

def log_traffic(message):
    if FLAGS.log_traffic:
        log_debug(message)

def log_info(message):
    if FLAGS.log_level <= 1:
        prefix_print('I ', str(message))

def log_warn(message):
    if FLAGS.log_level <= 2:
        prefix_print('W ', str(message))

def log_error(message):
    if FLAGS.log_level <= 3:
        prefix_print('E ', str(message))


# Graph Creation
# ==============

def variable_on_ps_level(name, shape, initializer, trainable=True):
    r'''
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_ps_level()``
    used to create a variable in CPU memory.
    '''
    # Use the /cpu:0 device on worker_device for scoped operations
    with tf.device(worker_device):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer, trainable=trainable)
    return var


def BiRNN(batch_x, seq_length, dropout):
    r'''
    That done, we will define the learned variables, the weights and biases,
    within the method ``BiRNN()`` which also constructs the neural network.
    The variables named ``hn``, where ``n`` is an integer, hold the learned weight variables.
    The variables named ``bn``, where ``n`` is an integer, hold the learned bias variables.
    In particular, the first variable ``h1`` holds the learned weight matrix that
    converts an input vector of dimension ``n_input + 2*n_input*n_context``
    to a vector of dimension ``n_hidden_1``.
    Similarly, the second variable ``h2`` holds the weight matrix converting
    an input vector of dimension ``n_hidden_1`` to one of dimension ``n_hidden_2``.
    The variables ``h3``, ``h5``, and ``h6`` are similar.
    Likewise, the biases, ``b1``, ``b2``..., hold the biases for the various layers.
    '''

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    batch_x_shape = tf.shape(batch_x)

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x, [-1, n_input + 2*n_input*n_context]) # (n_steps*batch_size, n_input + 2*n_input*n_context)

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.

    # 1st layer
    b1 = variable_on_ps_level('b1', [n_hidden_1], tf.random_normal_initializer(stddev=FLAGS.b1_stddev))
    h1 = variable_on_ps_level('h1', [n_input + 2*n_input*n_context, n_hidden_1], tf.contrib.layers.xavier_initializer(uniform=False))
    layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), FLAGS.relu_clip)
    layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout[0]))

    # 2nd layer
    b2 = variable_on_ps_level('b2', [n_hidden_2], tf.random_normal_initializer(stddev=FLAGS.b2_stddev))
    h2 = variable_on_ps_level('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(stddev=FLAGS.h2_stddev))
    layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), FLAGS.relu_clip)
    layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[1]))

    # 3rd layer
    b3 = variable_on_ps_level('b3', [n_hidden_3], tf.random_normal_initializer(stddev=FLAGS.b3_stddev))
    h3 = variable_on_ps_level('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=FLAGS.h3_stddev))
    layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), FLAGS.relu_clip)
    layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[2]))

    # Now we create the forward and backward LSTM units.
    # Both of which have inputs of length `n_cell_dim` and bias `1.0` for the forget gate of the LSTM.

    # Forward direction cell: (if else required for TF 1.0 and 1.1 compat)
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True) \
                   if 'reuse' not in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args else \
                   tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                                                input_keep_prob=1.0 - dropout[3],
                                                output_keep_prob=1.0 - dropout[3],
                                                seed=FLAGS.random_seed)
    # Backward direction cell: (if else required for TF 1.0 and 1.1 compat)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True) \
                   if 'reuse' not in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args else \
                   tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                input_keep_prob=1.0 - dropout[4],
                                                output_keep_prob=1.0 - dropout[4],
                                                seed=FLAGS.random_seed)

    # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
    # as the LSTM BRNN expects its input to be of shape `[max_time, batch_size, input_size]`.
    layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])

    # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                             cell_bw=lstm_bw_cell,
                                                             inputs=layer_3,
                                                             dtype=tf.float32,
                                                             time_major=True,
                                                             sequence_length=seq_length)

    # Reshape outputs from two tensors each of shape [n_steps, batch_size, n_cell_dim]
    # to a single tensor of shape [n_steps*batch_size, 2*n_cell_dim]
    outputs = tf.concat(outputs, 2)
    outputs = tf.reshape(outputs, [-1, 2*n_cell_dim])

    # Now we feed `outputs` to the fifth hidden layer with clipped RELU activation and dropout
    b5 = variable_on_ps_level('b5', [n_hidden_5], tf.random_normal_initializer(stddev=FLAGS.b5_stddev))
    h5 = variable_on_ps_level('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=FLAGS.h5_stddev))
    layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), FLAGS.relu_clip)
    layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

    # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
    # creating `n_classes` dimensional vectors, the logits.
    b6 = variable_on_ps_level('b6', [n_hidden_6], tf.random_normal_initializer(stddev=FLAGS.b6_stddev))
    h6 = variable_on_ps_level('h6', [n_hidden_5, n_hidden_6], tf.contrib.layers.xavier_initializer(uniform=False))
    layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_hidden_6])

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6


# Accuracy and Loss
# =================

# In accord with 'Deep Speech: Scaling up end-to-end speech recognition'
# (http://arxiv.org/abs/1412.5567),
# the loss function used by our network should be the CTC loss function
# (http://www.cs.toronto.edu/~graves/preprint.pdf).
# Conveniently, this loss function is implemented in TensorFlow.
# Thus, we can simply make use of this implementation to define our loss.

def calculate_mean_edit_distance_and_loss(tower, model_feeder, dropout):
    r'''
    This routine beam search decodes a mini-batch and calculates the loss and mean edit distance.
    Next to batch size, total and average loss it returns the mean edit distance,
    the decoded result and the batch's original Y.
    '''
    # Obtain the next batch of data
    batch_size, batch_x, batch_seq_len, batch_y = model_feeder.next_batch(tower)

    # Calculate the logits of the batch using BiRNN
    logits = BiRNN(batch_x, tf.to_int64(batch_seq_len), dropout)

    # Compute the CTC loss using either TensorFlow's `ctc_loss` or Baidu's `warp_ctc_loss`.
    if FLAGS.use_warpctc:
        total_loss = tf.contrib.warpctc.warp_ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)
    else:
        total_loss = tf.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)

    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(total_loss)

    # Beam search decode the batch
    decoded, _ = tf.nn.ctc_beam_search_decoder(logits, batch_seq_len, merge_repeated=False)

    # Compute the edit (Levenshtein) distance
    distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), batch_y)

    # Compute the mean edit distance
    mean_edit_distance = tf.reduce_mean(distance)

    # Finally we return the
    # - calculated total and
    # - average losses,
    # - the Levenshtein distance,
    # - the recognition mean edit distance,
    # - the decoded batch and
    # - the original batch_y (which contains the verified transcriptions).
    return batch_size, total_loss, avg_loss, distance, mean_edit_distance, decoded, batch_y


# Adam Optimization
# =================

# In constrast to 'Deep Speech: Scaling up end-to-end speech recognition'
# (http://arxiv.org/abs/1412.5567),
# in which 'Nesterov's Accelerated Gradient Descent'
# (www.cs.toronto.edu/~fritz/absps/momentum.pdf) was used,
# we will use the Adam method for optimization (http://arxiv.org/abs/1412.6980),
# because, generally, it requires less fine-tuning.
def create_optimizer(weight):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate * weight,
                                       beta1=FLAGS.beta1,
                                       beta2=FLAGS.beta2,
                                       epsilon=FLAGS.epsilon)
    return optimizer


# Towers
# ======

# In order to properly make use of multiple GPU's, one must introduce new abstractions,
# not present when using a single GPU, that facilitate the multi-GPU use case.
# In particular, one must introduce a means to isolate the inference and gradient
# calculations on the various GPU's.
# The abstraction we intoduce for this purpose is called a 'tower'.
# A tower is specified by two properties:
# * **Scope** - A scope, as provided by `tf.name_scope()`,
# is a means to isolate the operations within a tower.
# For example, all operations within 'tower 0' could have their name prefixed with `tower_0/`.
# * **Device** - A hardware device, as provided by `tf.device()`,
# on which all operations within the tower execute.
# For example, all operations of 'tower 0' could execute on the first GPU `tf.device('/gpu:0')`.

def for_each_tower(callback, params):
    results = []
    with tf.variable_scope(tf.get_variable_scope()):
        # Loop over available_devices
        for i in range(len(available_devices)):
            # Execute operations of tower i on device i
            device = available_devices[i]
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i) as scope:
                    results.append(callback(i, *params[i]))
                    # Allow for variables to be re-used by the next tower
                    tf.get_variable_scope().reuse_variables()
    return results

def get_tower_results(model_feeder):
    r'''
    With this preliminary step out of the way, we can for each GPU introduce a
    tower for which's batch we calculate

    * The CTC decodings ``decoded``,
    * The (total) loss against the outcome (Y) ``total_loss``,
    * The loss averaged over the whole batch ``avg_loss``,
    * The optimization gradient (computed based on the averaged loss),
    * The Levenshtein distances between the decodings and their transcriptions ``distance``,
    * The mean edit distance of the outcome averaged over the whole batch ``mean_edit_distance``
    '''

    results = for_each_tower(calculate_mean_edit_distance_and_loss, \
                             [[model_feeder, dropout_rates]] * len(available_devices))

    # Compute sum of all tower batch sizes
    batch_sizes = [result[0] for result in results]
    batch_size_sum = tf.to_float(tf.maximum(1, tf.reduce_sum(batch_sizes)))

    optimizer = create_optimizer(batch_size_sum)

    def weight_and_compute_gradients(index, batch_size, total_loss, avg_loss, distance, \
                                     mean_edit_distance, decoded, labels):
        batch_size_float = tf.to_float(batch_size)
        return batch_size, \
               total_loss, \
               avg_loss * batch_size_float, \
               distance, \
               mean_edit_distance * batch_size_float, \
               decoded, \
               labels, \
               optimizer.compute_gradients(avg_loss)

    results = for_each_tower(weight_and_compute_gradients, results)

    batch_sizes, total_losses, avg_losses, distances, mean_edit_distances, decodings, labels, gradients = zip(*results)

    # Return the optimizer, the results tuple, batch sizes, the gradients, and the means of mean edit distances and average losses
    return optimizer, \
           (labels, decodings, distances, total_losses), \
           batch_sizes, \
           gradients, \
           tf.reduce_mean(mean_edit_distances, 0) / batch_size_sum, \
           tf.reduce_mean(avg_losses, 0) / batch_size_sum


def average_gradients(batch_sizes, tower_gradients):
    r'''
    A routine for computing each variable's average of the gradients obtained from the GPUs.
    Note also that this code acts as a syncronization point as it requires all
    GPUs to be finished with their mini-batch before it can run to completion.
    '''
    # List of average gradients to return to the caller
    average_grads = []

    # Run this on cpu_device to conserve GPU memory
    with tf.device(cpu_device):
        # Compute sum of all tower batch sizes
        batch_size_sum = tf.reduce_sum(batch_sizes, 0)
        sample_number = tf.maximum(1, batch_size_sum)

        # Loop over gradient/variable pairs from all towers
        for grad_and_vars in zip(*tower_gradients):
            # Introduce grads to store the gradients for the current variable
            grads = []

            # Loop over the gradients for the current variable
            for gv, batch_size in zip(grad_and_vars, batch_sizes):
                # Weighted gradient - batch size is 0 for dummy sample
                g = gv[0] * tf.to_float(batch_size)
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            grad = grad / tf.to_float(sample_number)

            # Create a gradient/variable tuple for the current variable with its average gradient
            grad_and_var = (grad, grad_and_vars[0][1])

            # Add the current tuple to average_grads
            average_grads.append(grad_and_var)

    # Return result to caller
    return average_grads, batch_size_sum



# Logging
# =======

def log_variable(variable, gradient=None):
    r'''
    We introduce a function for logging a tensor variable's current state.
    It logs scalar values for the mean, standard deviation, minimum and maximum.
    Furthermore it logs a histogram of its state and (if given) of an optimization gradient.
    '''
    name = variable.name
    mean = tf.reduce_mean(variable)
    tf.summary.scalar(name='%s/mean'   % name, tensor=mean)
    tf.summary.scalar(name='%s/sttdev' % name, tensor=tf.sqrt(tf.reduce_mean(tf.square(variable - mean))))
    tf.summary.scalar(name='%s/max'    % name, tensor=tf.reduce_max(variable))
    tf.summary.scalar(name='%s/min'    % name, tensor=tf.reduce_min(variable))
    tf.summary.histogram(name=name, values=variable)
    if gradient is not None:
        if isinstance(gradient, tf.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is not None:
            tf.summary.histogram(name='%s/gradients' % name, values=grad_values)

def log_grads_and_vars(grads_and_vars):
    r'''
    Let's also introduce a helper function for logging collections of gradient/variable tuples.
    '''
    for gradient, variable in grads_and_vars:
        log_variable(variable, gradient=gradient)

# Helpers
# =======

def calculate_report(results_tuple):
    r'''
    This routine will calculate a WER report.
    It'll compute the `mean` WER and create ``Sample`` objects of the ``report_count`` top lowest
    loss items from the provided WER results tuple (only items with WER!=0 and ordered by their WER).
    '''
    samples = []
    items = list(zip(*results_tuple))
    mean_wer = 0.0
    for label, decoding, distance, loss in items:
        if len(label) == 1 and label[0] == 0:
            # skip dummy sample
            continue
        corrected = correction(decoding)
        sample_wer = wer(label, corrected)
        sample = Sample(label, corrected, loss, distance, sample_wer)
        samples.append(sample)
        mean_wer += sample_wer

    # Getting the mean WER from the accumulated one
    mean_wer = mean_wer / len(items)
    # Filter out all items with WER=0
    samples = [s for s in samples if s.wer > 0]
    # Order the remaining items by their loss (lowest loss on top)
    samples.sort(key=lambda s: s.loss)
    # Take only the first report_count items
    samples = samples[:FLAGS.report_count]
    # Order this top FLAGS.report_count items by their WER (lowest WER on top)
    samples.sort(key=lambda s: s.wer)
    return mean_wer, samples

def collect_results(results_tuple, returns):
    r'''
    This routine will help collecting partial results for the WER reports.
    The ``results_tuple`` is composed of an array of the original labels,
    an array of the corresponding decodings, an array of the corrsponding
    distances and an array of the corresponding losses. ``returns`` is built up
    in a similar way, containing just the unprocessed results of one
    ``session.run`` call (effectively of one batch).
    Labels and decodings are converted to text before splicing them into their
    corresponding results_tuple lists. In the case of decodings,
    for now we just pick the first available path.
    '''
    # Each of the arrays within results_tuple will get extended by a batch of each available device
    for i in range(len(available_devices)):
        # Collect the labels
        results_tuple[0].extend(sparse_tensor_value_to_texts(returns[0][i]))
        # Collect the decodings - at the moment we default to the first one
        results_tuple[1].extend(sparse_tensor_value_to_texts(returns[1][i][0]))
        # Collect the distances
        results_tuple[2].extend(returns[2][i])
        # Collect the losses
        results_tuple[3].extend(returns[3][i])

def export():
    r'''
    Restores the trained variables into a simpler graph that will be exported for serving.
    '''
    log_info('Exporting the model...')
    with tf.device('/cpu:0'):

        tf.reset_default_graph()
        session = tf.Session(config=session_config)

        # Run inference

        # Input tensor will be of shape [batch_size, n_steps, n_input + 2*n_input*n_context]
        input_tensor = tf.placeholder(tf.float32, [None, None, n_input + 2*n_input*n_context], name='input_node')

        seq_length = tf.placeholder(tf.int32, [None], name='input_lengths')

        # Calculate the logits of the batch using BiRNN
        logits = BiRNN(input_tensor, tf.to_int64(seq_length), no_dropout)

        # Beam search decode the batch
        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, seq_length, merge_repeated=False)
        decoded = tf.convert_to_tensor(
            [tf.sparse_tensor_to_dense(sparse_tensor) for sparse_tensor in decoded], name='output_node')

        # TODO: Transform the decoded output to a string

        # Create a saver and exporter using variables from the above newly created graph
        saver = tf.train.Saver(tf.global_variables())
        model_exporter = exporter.Exporter(saver)

        # Restore variables from training checkpoint
        # TODO: This restores the most recent checkpoint, but if we use validation to counterract
        #       over-fitting, we may want to restore an earlier checkpoint.
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        checkpoint_path = checkpoint.model_checkpoint_path
        saver.restore(session, checkpoint_path)
        log_info('Restored checkpoint at training epoch %d' % (int(checkpoint_path.split('-')[-1]) + 1))

        # Initialise the model exporter and export the model
        model_exporter.init(session.graph.as_graph_def(),
                            named_graph_signatures = {
                                'inputs': exporter.generic_signature(
                                    { 'input': input_tensor,
                                      'input_lengths': seq_length}),
                                'outputs': exporter.generic_signature(
                                    { 'outputs': decoded})})
        if FLAGS.remove_export:
            actual_export_dir = os.path.join(FLAGS.export_dir, '%08d' % FLAGS.export_version)
            if os.path.isdir(actual_export_dir):
                log_info('Removing old export')
                shutil.rmtree(actual_FLAGS.export_dir)
        try:
            # Export serving model
            model_exporter.export(FLAGS.export_dir, tf.constant(FLAGS.export_version), session)

            # Export graph
            input_graph_name = 'input_graph.pb'
            tf.train.write_graph(session.graph, FLAGS.export_dir, input_graph_name, as_text=False)

            # Freeze graph
            input_graph_path = os.path.join(FLAGS.export_dir, input_graph_name)
            input_saver_def_path = ''
            input_binary = True
            output_node_names = 'output_node'
            restore_op_name = 'save/restore_all'
            filename_tensor_name = 'save/Const:0'
            output_graph_path = os.path.join(FLAGS.export_dir, 'output_graph.pb')
            clear_devices = False
            freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                                      input_binary, checkpoint_path, output_node_names,
                                      restore_op_name, filename_tensor_name,
                                      output_graph_path, clear_devices, '')

            log_info('Models exported at %s' % (FLAGS.export_dir))
        except RuntimeError:
            log_error(sys.exc_info()[1])


class MBC(MessageBusClient):
    def __init__(self, cluster, task):
        MessageBusClient.__init__(self, cluster, 'worker', task)
        self.index_lock = Lock()
        self.index = 0

    def chief_allocate_indices(self, number):
        with self.index_lock:
            value = self.index
            self.index = self.index + number
            #print ('Allocated %d indices.' % number)
            return value
        
    def allocate_indices(self, number):
        return self.call('worker', 0, 'chief_allocate_indices', number)

def main(_) :

    initialize_globals()

    is_chief = FLAGS.task_index == 0

    server = tf.train.Server(cluster, job_name='worker', task_index=FLAGS.task_index)

    mbc = MBC(cluster, FLAGS.task_index)

    # Reading training set
    train_set = DataSet(FLAGS.train_files.split(','),
                        limit=FLAGS.limit_train,
                        skip=FLAGS.skip_train,
                        ascending=FLAGS.train_ascending)

    # Reading validation set
    dev_set = DataSet(FLAGS.dev_files.split(','),
                      limit=FLAGS.limit_dev,
                      skip=FLAGS.skip_dev,
                      ascending=FLAGS.dev_ascending)

    # Reading test set
    test_set = DataSet(FLAGS.test_files.split(','),
                       limit=FLAGS.limit_test,
                       skip=FLAGS.skip_test,
                       ascending=FLAGS.test_ascending)

    # Combining all sets to a multi set model feeder
    model_feeder = ModelFeeder(n_input,
                               n_context,
                               len(available_devices),
                               FLAGS.threads_per_set,
                               FLAGS.loader_buffer,
                               min(memory_limits),
                               FLAGS.queue_capacity,
                               allocate_indices=mbc.allocate_indices)

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    try:
        with tf.Session(server.target, config=session_config) as session:

            if is_chief:
                # Get the data_set specific graph end-points
                optimizer, results_tuple, batch_sizes, gradients, mean_edit_distance, loss = get_tower_results(model_feeder)

                # Average tower gradients across GPUs
                avg_tower_gradients, sample_number = average_gradients(batch_sizes, gradients)

                # Add summaries of all variables and gradients to log
                log_grads_and_vars(avg_tower_gradients)

                # Op to merge all summaries for the summary hook
                merge_all_summaries_op = tf.summary.merge_all()

                # Apply gradients to modify the model
                apply_gradient_op = optimizer.apply_gradients(avg_tower_gradients)

                # Increment global sample counter
                sample_counter = variable_on_ps_level('sample_counter', None, 0, trainable=False)
                sample_inc_op = tf.assign_add(sample_counter, sample_number)

                # Add an op to initialize the variables.
                init_op = tf.global_variables_initializer()

                # Central thread coordinator
                coord = tf.train.Coordinator()

                # Starting threads for feeding
                model_feeder.start_queue_threads(session, coord)

                # Initialize all variables
                session.run(init_op)

                # Retrieving global_step from the (potentially restored) model
                current_sample_number = session.run(sample_counter)

                # Number of samples per epoch - to be at least 1
                samples_train = max(1, len(train_set.files))
                amples_dev =    max(1, len(dev_set.files))
                amples_test =   max(1, len(test_set.files))

                # The start epoch of our training
                start_epoch = current_sample_number // samples_train

                # Number of additional 'jobs' trained already 'on top of' our start epoch
                samples_trained = (current_sample_number % samples_train)

                # A negative epoch means to add its absolute number to the epochs already computed
                target_epoch = (start_epoch + abs(FLAGS.epoch)) if FLAGS.epoch < 0 else start_epoch

                # Important for debugging
                log_debug('start epoch: %d' % start_epoch)
                log_debug('target epoch: %d' % target_epoch)
                log_debug('samples per epoch: %d' % samples_train)
                log_debug('number of samples already trained in first epoch: %d' % samples_trained)

                def apply_set(data_set, should_train, should_report):
                    # Sets the current data_set for the respective placeholder in feed_dict
                    model_feeder.start_data_set(data_set)
                    # Sets the range allocator back to 0
                    mbc.index = 0
                    # Setting the training operation in case of training requested
                    train_ops = [apply_gradient_op, sample_inc_op] if should_train else []
                    # Requirements to display a WER report
                    if should_report:
                        # Reset mean edit distance
                        total_mean_edit_distance = 0.0
                        # Create report results tuple
                        report_results = ([],[],[],[])
                        # Extend the session.run parameters
                        report_params = [results_tuple, mean_edit_distance]
                    else:
                        report_params = []

                    # Initializing all aggregators
                    total_loss = 0
                    total_sample_number = 0
                    total_mean_edit_distance = 0

                    # Loop over the batches till sample number is 0 (the epoch is over)
                    target_sample_number = len(data_set.files)
                    while total_sample_number < target_sample_number:
                        log_debug('Total sample number: %d, Target sample number: %d' % (total_sample_number, target_sample_number))
                        log_debug(model_feeder.get_state_report())
                        # Compute the batch
                        _, current_sample_number, current_loss, current_report = \
                            session.run([train_ops, sample_number, loss, report_params])
                        log_debug('Finished batch with %d samples.' % current_sample_number)
                        # Collect results
                        total_sample_number += current_sample_number
                        total_loss += current_loss * current_sample_number
                        if should_report:
                            # Collect individual sample results
                            collect_results(report_results, current_report[0])
                            # Add batch to total_mean_edit_distance
                            total_mean_edit_distance += current_report[1] * current_sample_number

                    # Gathering job results
                    total_loss = total_loss / total_sample_number
                    if should_report:
                        total_mean_edit_distance = total_mean_edit_distance / total_sample_number
                        wer, samples = calculate_report(report_results)

                if FLAGS.train and target_epoch > start_epoch:
                    log_info('STARTING Optimization')
                    for epoch in range(start_epoch, target_epoch):
                        print('=' * 100)
                        log_info('Training epoch %d...' % epoch)
                        apply_set(train_set, True, FLAGS.display_step > 0 and (epoch % FLAGS.display_step) > 0)

                        if FLAGS.validation_step > 0 and epoch % FLAGS.validation_step > 0:
                            log_info('Validating epoch %d...' % epoch)
                            apply_set(dev_set, False, epoch % FLAGS.display_step > 0)
                
                        log_info('Finished epoch %d.' % epoch)
                    print('=' * 100)
                    log_info('FINISHED Optimization')

                if FLAGS.test:
                    log_info('Testing epoch %d...' % target_epoch)
                    apply_set(train_set, False, True)
                    log_info('Finished testing epoch %d.' % target_epoch)

                coord.request_stop()
                coord.join()

        log_debug('Session closed.')
    except Exception as ex:
        print(ex)

    # Are we the main process?
    if is_chief:
        # Doing solo/post-processing work just on the main process...
        # Exporting the model
        if FLAGS.export_dir:
            export()

if __name__ == '__main__' :
    tf.app.run()
