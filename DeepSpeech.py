#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import datetime
import json
import numpy as np
import shutil
import subprocess
import sys
import tempfile
import tensorflow as tf
import time
import importlib

from collections import OrderedDict
from math import ceil, floor
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.tools import freeze_graph
from tensorflow.python.ops import ctc_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from util.gpu import get_available_gpus
from util.spell import correction
from util.shared_lib import check_cupti
from util.text import sparse_tensor_value_to_texts, wer
from util.data_set_helpers import SwitchableDataSet
from xdg import BaseDirectory as xdg


# Importer
# ========

tf.app.flags.DEFINE_string  ('importer',         'ldc93s1',   'importer module - one of ldc93s1, LDC97S62, ted, librivox, fischer')
tf.app.flags.DEFINE_string  ('dataset_path',     '',          'data set path for the importer - defaults to .data/<importer>')
tf.app.flags.DEFINE_boolean ('fulltrace',        False,       'if full trace debug info should be generated during training')

# Cluster configuration
# =====================

tf.app.flags.DEFINE_string  ('ps_hosts',         '',          'parameter servers - comma separated list of hostname:port pairs')
tf.app.flags.DEFINE_string  ('worker_hosts',     '',          'workers - comma separated list of hostname:port pairs')
tf.app.flags.DEFINE_string  ('job_name',         'localhost', 'job name - one of localhost (default), worker, ps')
tf.app.flags.DEFINE_integer ('task_index',       0,           'index of task within the job - worker with index 0 will be the chief')

# Global Constants
# ================

tf.app.flags.DEFINE_integer ('epochs',           75,          'number of iterations to train')
tf.app.flags.DEFINE_boolean ('use_warpctc',      False,       'weather to use GPU bound Warp-CTC')

tf.app.flags.DEFINE_float   ('dropout_rate',     0.05,        'dropout rate for feedforward layers')
tf.app.flags.DEFINE_float   ('dropout_rate2',    -1.0,        'dropout rate for layer 2 - defaults to dropout_rate')
tf.app.flags.DEFINE_float   ('dropout_rate3',    -1.0,        'dropout rate for layer 3 - defaults to dropout_rate')
tf.app.flags.DEFINE_float   ('dropout_rate4',    0.0,         'dropout rate for layer 4 - defaults to 0.0')
tf.app.flags.DEFINE_float   ('dropout_rate5',    0.0,         'dropout rate for layer 5 - defaults to 0.0')
tf.app.flags.DEFINE_float   ('dropout_rate6',    -1.0,        'dropout rate for layer 6 - defaults to dropout_rate')

tf.app.flags.DEFINE_float   ('relu_clip',        20.0,        'ReLU clipping value for non-recurrant layers')

# Adam optimizer (http://arxiv.org/abs/1412.6980) parameters

tf.app.flags.DEFINE_float   ('beta1',            0.9,         'Beta 1 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('beta2',            0.999,       'Beta 2 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('epsilon',          1e-8,        'Epsilon parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('learning_rate',    0.001,       'Learning Rate of Adam optimizer')

# Batch sizes

tf.app.flags.DEFINE_integer ('train_batch_size', 1,           'number of elements in a training batch')
tf.app.flags.DEFINE_integer ('dev_batch_size',   1,           'number of elements in a validation batch')
tf.app.flags.DEFINE_integer ('test_batch_size',  1,           'number of elements in a test batch')

# Sample limits

tf.app.flags.DEFINE_integer ('limit_train',      0,           'maximum number of elements to use from train set')
tf.app.flags.DEFINE_integer ('limit_dev',        0,           'maximum number of elements to use from validation set')
tf.app.flags.DEFINE_integer ('limit_test',       0,           'maximum number of elements to use from test set')

# Step widths

tf.app.flags.DEFINE_integer ('display_step',     1,           'number of epochs we cycle through before displaying progress')
tf.app.flags.DEFINE_integer ('validation_step',  0,           'number of epochs we cycle through before validating the model')

# Checkpointing

tf.app.flags.DEFINE_string  ('checkpoint_dir',   '',          'directory in which checkpoints are stored')
tf.app.flags.DEFINE_integer ('checkpoint_secs',  600,         'checkpoint saving interval in seconds')

# Exporting

tf.app.flags.DEFINE_string  ('export_dir',       '',          'directory in which exported models are stored')
tf.app.flags.DEFINE_integer ('export_version',   1,           'version number of the exported model')
tf.app.flags.DEFINE_boolean ('remove_export',    False,       'weather to remove old exported models')

# Reporting

tf.app.flags.DEFINE_boolean ('publish',          False,       'weather to publish hyper parameters')
tf.app.flags.DEFINE_integer ('report_count',     10,          'number of phrases to print out during a WER report')
tf.app.flags.DEFINE_boolean ('log_placement',    False,       'weather to log device placement of the operators to the console')
tf.app.flags.DEFINE_boolean ('log_variables',    False,       'weather to log gradients and variables summaries to TensorBoard during training')
tf.app.flags.DEFINE_integer ('summaries_steps',  1,           'number of global training steps we cycle through before saving a summary')
tf.app.flags.DEFINE_string  ('logs_dir',         'logs',      'directory in which checkpoints are stored')
tf.app.flags.DEFINE_string  ('wer_log_file',     'werlog.js', 'log-file for keeping track of WER progress')

# Initialization

tf.app.flags.DEFINE_integer ('random_seed',      4567,        'default random seed that is used to initialize variables')
tf.app.flags.DEFINE_float   ('default_stddev',   0.046875,    'default standard deviation to use when initialising weights and biases')

for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
    tf.app.flags.DEFINE_float('%s_stddev' % var, None, 'standard deviation to use when initialising %s' % var)


FLAGS = tf.app.flags.FLAGS


# Geometric Constants
# ===================

# For an explanation of the meaning of the geometric constants, please refer to
# doc/Geometry.md

# Number of MFCC features
n_input = 26 # TODO: Determine this programatically from the sample rate

# The number of frames in the context
n_context = 9 # TODO: Determine the optimal value using a validation data set

# Number of units in hidden layers
n_hidden_1 = 494
n_hidden_2 = 494
n_hidden_5 = 494

# LSTM cell state dimension
n_cell_dim = 494

# The number of units in the third layer, which feeds in to the LSTM
n_hidden_3 = 2 * n_cell_dim

# The number of characters in the target language plus one
n_character = 29 # TODO: Determine if this should be extended with other punctuation

# The number of units in the sixth layer
n_hidden_6 = n_character


# Graph Creation
# ==============

def variable_on_cpu(name, shape, initializer):
    r"""
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_cpu()``
    used to create a variable in CPU memory.
    """
    # Use the /cpu:0 device on worker_device for scoped operations
    if len(FLAGS.ps_hosts) == 0:
        device = worker_device
    else:
        device = tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster)

    with tf.device(device):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def BiRNN(batch_x, seq_length, dropout):
    r"""
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
    """
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
    b1 = variable_on_cpu('b1', [n_hidden_1], tf.random_normal_initializer(stddev=FLAGS.b1_stddev))
    h1 = variable_on_cpu('h1', [n_input + 2*n_input*n_context, n_hidden_1], tf.random_normal_initializer(stddev=FLAGS.h1_stddev))
    layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), FLAGS.relu_clip)
    layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout[0]))

    # 2nd layer
    b2 = variable_on_cpu('b2', [n_hidden_2], tf.random_normal_initializer(stddev=FLAGS.b2_stddev))
    h2 = variable_on_cpu('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(stddev=FLAGS.h2_stddev))
    layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), FLAGS.relu_clip)
    layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[1]))

    # 3rd layer
    b3 = variable_on_cpu('b3', [n_hidden_3], tf.random_normal_initializer(stddev=FLAGS.b3_stddev))
    h3 = variable_on_cpu('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=FLAGS.h3_stddev))
    layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), FLAGS.relu_clip)
    layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[2]))

    # Now we create the forward and backward LSTM units.
    # Both of which have inputs of length `n_cell_dim` and bias `1.0` for the forget gate of the LSTM.

    # Forward direction cell:
    lstm_fw_cell = core_rnn_cell.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
    lstm_fw_cell = core_rnn_cell.DropoutWrapper(lstm_fw_cell,
                                                input_keep_prob=1.0 - dropout[3],
                                                output_keep_prob=1.0 - dropout[3],
                                                seed=FLAGS.random_seed)
    # Backward direction cell:
    lstm_bw_cell = core_rnn_cell.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = core_rnn_cell.DropoutWrapper(lstm_bw_cell,
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
    b5 = variable_on_cpu('b5', [n_hidden_5], tf.random_normal_initializer(stddev=FLAGS.b5_stddev))
    h5 = variable_on_cpu('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=FLAGS.h5_stddev))
    layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), FLAGS.relu_clip)
    layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

    # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
    # creating `n_classes` dimensional vectors, the logits.
    b6 = variable_on_cpu('b6', [n_hidden_6], tf.random_normal_initializer(stddev=FLAGS.b6_stddev))
    h6 = variable_on_cpu('h6', [n_hidden_5, n_hidden_6], tf.random_normal_initializer(stddev=FLAGS.h6_stddev))
    layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_hidden_6])

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6


# Accuracy and Loss
# =================

# In accord with "Deep Speech: Scaling up end-to-end speech recognition"
# (http://arxiv.org/abs/1412.5567),
# the loss function used by our network should be the CTC loss function
# (http://www.cs.toronto.edu/~graves/preprint.pdf).
# Conveniently, this loss function is implemented in TensorFlow.
# Thus, we can simply make use of this implementation to define our loss.

def calculate_accuracy_and_loss(batch_set, dropout):
    r"""
    This routine beam search decodes a mini-batch and calculates the loss and accuracy.
    Next to total and average loss it returns the accuracy,
    the decoded result and the batch's original Y.
    """
    # Obtain the next batch of data
    batch_x, batch_seq_len, batch_y = batch_set.next_batch()

    # Calculate the logits of the batch using BiRNN
    logits = BiRNN(batch_x, tf.to_int64(batch_seq_len), dropout)

    # Compute the CTC loss using either TensorFlow's `ctc_loss` or Baidu's `warp_ctc_loss`.
    if FLAGS.use_warpctc:
        total_loss = tf.contrib.warpctc.warp_ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)
    else:
        total_loss = ctc_ops.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)

    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(total_loss)

    # Beam search decode the batch
    decoded, _ = ctc_ops.ctc_beam_search_decoder(logits, batch_seq_len, merge_repeated=False)

    # Compute the edit (Levenshtein) distance
    distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), batch_y)

    # Compute the accuracy
    accuracy = tf.reduce_mean(distance)

    # Finally we return the
    # - calculated total and
    # - average losses,
    # - the Levenshtein distance,
    # - the recognition accuracy,
    # - the decoded batch and
    # - the original batch_y (which contains the verified transcriptions).
    return total_loss, avg_loss, distance, accuracy, decoded, batch_y


# Adam Optimization
# =================

# In constrast to "Deep Speech: Scaling up end-to-end speech recognition"
# (http://arxiv.org/abs/1412.5567),
# in which "Nesterov's Accelerated Gradient Descent"
# (www.cs.toronto.edu/~fritz/absps/momentum.pdf) was used,
# we will use the Adam method for optimization (http://arxiv.org/abs/1412.6980),
# because, generally, it requires less fine-tuning.
def create_optimizer():
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate,
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
# For example, all operations within "tower 0" could have their name prefixed with `tower_0/`.
# * **Device** - A hardware device, as provided by `tf.device()`,
# on which all operations within the tower execute.
# For example, all operations of "tower 0" could execute on the first GPU `tf.device('/gpu:0')`.

def get_tower_results(batch_set, optimizer):
    r"""
    With this preliminary step out of the way, we can for each GPU introduce a
    tower for which's batch we calculate

    * The CTC decodings ``decoded``,
    * The (total) loss against the outcome (Y) ``total_loss``,
    * The loss averaged over the whole batch ``avg_loss``,
    * The optimization gradient (computed based on the averaged loss),
    * The Levenshtein distances between the decodings and their transcriptions ``distance``,
    * The accuracy of the outcome averaged over the whole batch ``accuracy``

    and retain the original ``labels`` (Y).
    ``decoded``, ``labels``, the optimization gradient, ``distance``, ``accuracy``,
    ``total_loss`` and ``avg_loss`` are collected into the corresponding arrays
    ``tower_decodings``, ``tower_labels``, ``tower_gradients``, ``tower_distances``,
    ``tower_accuracies``, ``tower_total_losses``, ``tower_avg_losses`` (dimension 0 being the tower).
    Finally this new method ``get_tower_results()`` will return those tower arrays.
    In case of ``tower_accuracies`` and ``tower_avg_losses``, it will return the
    averaged values instead of the arrays.
    """
    # Tower labels to return
    tower_labels = []

    # Tower decodings to return
    tower_decodings = []

    # Tower distances to return
    tower_distances = []

    # Tower total batch losses to return
    tower_total_losses = []

    # Tower gradients to return
    tower_gradients = []

    # To calculate the mean of the accuracies
    tower_accuracies = []

    # To calculate the mean of the losses
    tower_avg_losses = []

    with tf.variable_scope(tf.get_variable_scope()):
        # Loop over available_devices
        for i in xrange(len(available_devices)):
            # Execute operations of tower i on device i
            if len(FLAGS.ps_hosts) == 0:
                device = available_devices[i]
            else:
                device = tf.train.replica_device_setter(worker_device=available_devices[i], cluster=cluster)
            with tf.device(device):
                # Create a scope for all operations of tower i
                with tf.name_scope('tower_%d' % i) as scope:
                    # Calculate the avg_loss and accuracy and retrieve the decoded
                    # batch along with the original batch's labels (Y) of this tower
                    total_loss, avg_loss, distance, accuracy, decoded, labels = \
                        calculate_accuracy_and_loss(batch_set, no_dropout if optimizer is None else dropout_rates)

                    # Allow for variables to be re-used by the next tower
                    tf.get_variable_scope().reuse_variables()

                    # Retain tower's labels (Y)
                    tower_labels.append(labels)

                    # Retain tower's decoded batch
                    tower_decodings.append(decoded)

                    # Retain tower's distances
                    tower_distances.append(distance)

                    # Retain tower's total losses
                    tower_total_losses.append(total_loss)

                    # Compute gradients for model parameters using tower's mini-batch
                    gradients = optimizer.compute_gradients(avg_loss)

                    # Retain tower's gradients
                    tower_gradients.append(gradients)

                    # Retain tower's accuracy
                    tower_accuracies.append(accuracy)

                    # Retain tower's avg losses
                    tower_avg_losses.append(avg_loss)

    # Return the results tuple, the gradients, and the means of accuracies and losses
    return (tower_labels, tower_decodings, tower_distances, tower_total_losses), \
           tower_gradients, \
           tf.reduce_mean(tower_accuracies, 0), \
           tf.reduce_mean(tower_avg_losses, 0)


def average_gradients(tower_gradients):
    r"""
    A routine for computing each variable's average of the gradients obtained from the GPUs.
    Note also that this code acts as a syncronization point as it requires all
    GPUs to be finished with their mini-batch before it can run to completion.
    """
    # List of average gradients to return to the caller
    average_grads = []

    # Loop over gradient/variable pairs from all towers
    for grad_and_vars in zip(*tower_gradients):
        # Introduce grads to store the gradients for the current variable
        grads = []

        # Loop over the gradients for the current variable
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Create a gradient/variable tuple for the current variable with its average gradient
        grad_and_var = (grad, grad_and_vars[0][1])

        # Add the current tuple to average_grads
        average_grads.append(grad_and_var)

    # Return result to caller
    return average_grads



# Logging
# =======

def log_variable(variable, gradient=None):
    r"""
    We introduce a function for logging a tensor variable's current state.
    It logs scalar values for the mean, standard deviation, minimum and maximum.
    Furthermore it logs a histogram of its state and (if given) of an optimization gradient.
    """
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
    r"""
    Let's also introduce a helper function for logging collections of gradient/variable tuples.
    """
    for gradient, variable in grads_and_vars:
        log_variable(variable, gradient=gradient)

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

def get_git_branch():
    return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip()


# Helpers
# =======

def calculate_and_print_wer_report(caption, results_tuple):
    r"""
    This routine will print a WER report with a given caption.
    It'll print the `mean` WER plus summaries of the ``report_count`` top lowest
    loss items from the provided WER results tuple
    (only items with WER!=0 and ordered by their WER).
    """
    items = zip(*results_tuple)

    count = len(items)
    mean_wer = 0.0
    for i in xrange(count):
        item = items[i]
        # If distance > 0 we know that there is a WER > 0 and have to calculate it
        if item[2] > 0:
            # Replace result by language model corrected result
            item = (item[0], correction(item[1]), item[2], item[3])
            # Replacing accuracy tuple entry by the WER
            item = (item[0], item[1], wer(item[0], item[1]), item[3])
            # Replace items[i] with new item
            items[i] = item
            mean_wer = mean_wer + item[2]

    # Getting the mean WER from the accumulated one
    mean_wer = mean_wer / float(count)

    # Filter out all items with WER=0
    items = [a for a in items if a[2] > 0]

    # Order the remaining items by their loss (lowest loss on top)
    items.sort(key=lambda a: a[3])

    # Take only the first report_count items
    items = items[:FLAGS.report_count]

    # Order this top ten items by their WER (lowest WER on top)
    items.sort(key=lambda a: a[2])

    print "%s WER: %f" % (caption, mean_wer)
    for a in items:
        print "-" * 80
        print "    WER:    %f" % a[2]
        print "    loss:   %f" % a[3]
        print "    source: \"%s\"" % a[0]
        print "    result: \"%s\"" % a[1]

    return mean_wer

def print_report(caption, batch_set_result, no_wer=None):
    r"""
    This routine will print a report from a provided batch set result tuple.
    It takes a caption for titling the output plus the batch set result tuple.
    If the batch set result tuple contains accuracy and a report results tuple,
    a complete WER report will be calculated, printed and its mean WER returned.
    Otherwise it will just print the loss and return ``no_wer``.
    """
    # Unpacking batch set result tuple
    loss, accuracy, results_tuple = batch_set_result

    # We always have a loss value
    title = caption + " loss=" + "{:.9f}".format(loss)

    mean_wer = no_wer
    if accuracy is not None and results_tuple is not None:
        title += " avg_cer=" + "{:.9f}".format(accuracy)
        mean_wer = calculate_and_print_wer_report(title, results_tuple)
    else:
        print title

    return mean_wer


def collect_results(results_tuple, returns):
    r"""
    This routine will help collecting partial results for the WER reports.
    The ``results_tuple`` is composed of an array of the original labels,
    an array of the corresponding decodings, an array of the corrsponding
    distances and an array of the corresponding losses. ``returns`` is built up
    in a similar way, containing just the unprocessed results of one
    ``session.run`` call (effectively of one batch).
    Labels and decodings are converted to text before splicing them into their
    corresponding results_tuple lists. In the case of decodings,
    for now we just pick the first available path.
    """
    # Each of the arrays within results_tuple will get extended by a batch of each available device
    for i in xrange(len(available_devices)):
        # Collect the labels
        results_tuple[0].extend(sparse_tensor_value_to_texts(returns[0][i]))

        # Collect the decodings - at the moment we default to the first one
        results_tuple[1].extend(sparse_tensor_value_to_texts(returns[1][i][0]))

        # Collect the distances
        results_tuple[2].extend(returns[2][i])

        # Collect the losses
        results_tuple[3].extend(returns[3][i])


# For reporting we also need a standard way to do time measurements.
def stopwatch(start_duration=0):
    r"""
    This function will toggle a stopwatch.
    The first call starts it, second call stops it, third call continues it etc.
    So if you want to measure the accumulated time spent in a certain area of the code,
    you can surround that code by stopwatch-calls like this:

    .. code:: python

        fun_time = 0 # initializes a stopwatch
        [...]
        for i in xrange(10):
          [...]
          # Starts/continues the stopwatch - fun_time is now a point in time (again)
          fun_time = stopwatch(fun_time)
          fun()
          # Pauses the stopwatch - fun_time is now a duration
          fun_time = stopwatch(fun_time)
        [...]
        # The following line only makes sense after an even call of :code:`fun_time = stopwatch(fun_time)`.
        print "Time spent in fun():", format_duration(fun_time)

    """
    if start_duration == 0:
        return datetime.datetime.utcnow()
    else:
        return datetime.datetime.utcnow() - start_duration

def format_duration(duration):
    """Formats the result of an even stopwatch call as hours:minutes:seconds"""
    duration = duration if isinstance(duration, int) else duration.seconds
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


# Execution
# =========

def read_data_sets(set_names=['train', 'dev', 'test']):
    r"""
    Returns a :class:`DataSets` object of the selected importer, containing all available/selected sets.
    """
    return importer_module.read_data_sets(FLAGS.dataset_path,
                                          FLAGS.train_batch_size,
                                          FLAGS.dev_batch_size,
                                          FLAGS.test_batch_size,
                                          n_input,
                                          n_context,
                                          stride=max(1, len(FLAGS.worker_hosts)),
                                          offset=FLAGS.task_index,
                                          limit_dev=FLAGS.limit_dev,
                                          limit_test=FLAGS.limit_test,
                                          limit_train=FLAGS.limit_train,
                                          sets=set_names)

def train(server=None):
    r"""
    Trains the network on a given server of a cluster.
    If no server provided, it performs single process training.
    """

    # Create a variable to hold the global_step.
    # It will automgically get incremented by the optimizer.
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Read all data sets
    data_sets = read_data_sets()

    if data_sets.train.total_batches == 0:
        print 'Zero batches in train data set - stopping...'
        return

    # Calculates the epoch from a given global ``step``
    def get_epoch_from_step(step):
        # Uncomment the next line for debugging distributed TF
        # print ('step: %d, devices: %d, tau: %d, batches: %d' % (step, len(available_devices), 3, data_sets.train.total_batches))
        return int(ceil(float(step * len(available_devices) * 3) / float(data_sets.train.total_batches)))

    def log_wer(wer_type, wer):
        hash = get_git_revision_hash()
        time = datetime.datetime.utcnow().isoformat()
        with open(FLAGS.wer_log_file, 'a') as wer_log_file:
            if wer_log_file.tell() > 0:
                wer_log_file.write('\n')
            wer_log_file.write('logwer("%s", "%s", "%s", %f)' % (hash, time, wer_type, wer))
        # Publish to web server
        if FLAGS.publish:
            maybe_publish()

    # Get the data sets
    switchable_data_set = SwitchableDataSet(data_sets)

    # Create the optimizer
    optimizer = create_optimizer()

    # Synchronous training
    if not server is None:
        optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                   replicas_to_aggregate=3*len(FLAGS.worker_hosts),
                                                   total_num_replicas=3*len(FLAGS.worker_hosts))

    # Get the data_set specific graph end-points
    tower_results = get_tower_results(switchable_data_set, optimizer)
    results_tuple, gradients, accuracy, loss = tower_results

    # Average tower gradients across GPUs
    avg_tower_gradients = average_gradients(gradients)

    # Add variable summaries to log
    log_grads_and_vars(avg_tower_gradients)

    # Apply gradients to modify the model
    apply_gradient_op = optimizer.apply_gradients(avg_tower_gradients, global_step=global_step)


    def run_batches(session, data_set, train=False, query_report=False):
        r"""
        It iterates over all batches ``data_set`` and
        calculate the mean loss using the provided ``session``.
        If ``train`` is set to ``True``, the data set will get trained.
        If ``query_report`` is set to ``True``, data for a WER report will be gathered.
        Returns batch set result tuple consisting of the current global step, the average loss, and,
        if WER report data should be gathered, the average accuracy and a report results tuple.
        During operation this method is using variables of the encapsulating Python context.
        """

        # The feed_dict (mainly for switching between queues)
        feed_dict = {}

        # Sets the current data_set on SwitchableDataSet switchable_data_set
        # and the respective placeholder in feed_dict
        switchable_data_set.set_data_set(feed_dict, data_set)

        # The amount of batches per device (GPU) == ceiled number of loops in batch loop
        batches_per_device = ceil(float(data_set.total_batches) / len(available_devices))

        # Initialize loss aggregator
        total_loss = 0.0

        # Setting the training operation in case of training requested
        train_op = apply_gradient_op if train else []

        # Requirements to display a WER report
        if query_report:
            # Reset accuracy
            total_accuracy = 0.0
            # Create report results tuple
            report_results = ([],[],[],[])
            # Extend the session.run parameters
            report_params = [results_tuple, accuracy]
        else:
            report_params = []

        # So far the only extra parameter is the feed_dict
        extra_params = { 'feed_dict': feed_dict }

        # Loop over the batches
        for batch in range(int(batches_per_device)):
            # In case of the test set iteration from within the CoordHook,
            # the provided session is (for some reason) not
            # a MonitoredTrainingSession with a ``should_stop`` method.
            if 'should_stop' in dir(session):
                if session.should_stop():
                    break

            # Compute the batch
            _, current_step, batch_loss, batch_report = session.run([train_op, global_step, loss, report_params], **extra_params)

            # Uncomment the next line for debugging race conditions / distributed TF
            # print ('Batch step %d' % current_step)

            # Add batch to loss
            total_loss += batch_loss

            if query_report:
                # Collect individual sample results
                collect_results(report_results, batch_report[0])
                # Add batch to total_accuracy
                total_accuracy += batch_report[1]

        # Returning the batch set result tuple
        avg_loss = total_loss / batches_per_device
        if query_report:
            return current_step, (avg_loss, total_accuracy / batches_per_device, report_results)
        else:
            return current_step, (avg_loss, None, None)


    class CoordHook(tf.train.SessionRunHook):
        r"""
        Embedded coordination hook-class that will use variables of the
        surrounding Python context.
        It will start the importers queue threads after session creation.
        Before the session ends, it will evaluate the final test data set and
        send out this worker's stop token to all parameter servers.
        """
        def after_create_session(self, session, coord):
            print ('Starting queue runners...')
            self.threads = switchable_data_set.start_queue_threads(session, coord)
            print ('Queue runners started.')

        def end(self, session):
            # Iterate over test data set
            _, results = run_batches(session, data_sets.test, query_report=True)

            if is_chief:
                # Print Test WER report
                test_wer = print_report('Test', results)

                # Append to WER-log
                log_wer('test', test_wer)

            # Closing the data_set queues
            print ("Closing queues...")
            switchable_data_set.close_queue(session)

            # Sending our token (the task_index as a debug opportunity) to each parameter server.
            for enqueue in done_enqueues:
                print ('Sending stop token to ps...')
                session.run(enqueue, feed_dict={ token_placeholder: FLAGS.task_index })
                print ('Sent stop token to ps.')


    # Calculate last step
    last_step = FLAGS.epochs * data_sets.train.total_batches - 1

    # The StopAtStepHook handles stopping after running given steps.
    hooks = [CoordHook(),
             tf.train.StopAtStepHook(last_step=last_step)]

    # Hook to handle initialization and queues for sync replicas.
    if not server is None:
        hooks.append(optimizer.make_session_run_hook(is_chief))

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master='' if server is None else server.target,
                                           is_chief=is_chief,
                                           hooks=hooks,
                                           checkpoint_dir=FLAGS.checkpoint_dir,
                                           save_checkpoint_secs=FLAGS.checkpoint_secs,
                                           save_summaries_steps=FLAGS.summaries_steps,
                                           config=session_config) as session:

        if is_chief:
            print "STARTING Optimization\n"
            global_time = stopwatch()
            global_train_time = 0

            # Init recent word error rate levels
            train_wer = 0.0
            dev_wer = 0.0

        # Retrieving global_step from the (potentially restored) model
        feed_dict = {}
        switchable_data_set.set_data_set(feed_dict, data_sets.train)
        step = session.run(global_step, feed_dict=feed_dict)
        epoch = get_epoch_from_step(step)

        while not session.should_stop():

            if is_chief:
                print "STARTING Epoch", '%04d' % (epoch)
                overall_time = stopwatch()
                train_time = 0

                # (Re)start training stopwatches
                global_train_time = stopwatch(global_train_time)
                train_time = stopwatch(train_time)

            # As the entry check is done, we can now make the last one the current one.
            last_epoch = epoch

            # Determine if we want to display and/or validate on this iteration/worker
            is_display_step = is_chief and FLAGS.display_step > 0 and ((epoch + 1) % FLAGS.display_step == 0 or epoch == FLAGS.epochs - 1)
            is_validation_step = FLAGS.validation_step > 0 and (epoch + 1) % FLAGS.validation_step == 0

            # Train for one epoch, keep global step for next loop and report data
            step, results = run_batches(session, data_sets.train, train=True, query_report=is_display_step)

            # Calculate (potentially next) epoch on base of global step
            last_epoch = epoch
            epoch = get_epoch_from_step(step)

            if is_chief:
                # Stop training stopwatches
                global_train_time = stopwatch(global_train_time)
                train_time = stopwatch(train_time)

                # Print report and keep (potentially) calculated WER
                train_wer = print_report('Training', results)

                # If we got a Training WER, we append it to the WER log
                if train_wer:
                    log_wer('train', train_wer)

            if is_validation_step and not session.should_stop():
                _, results = run_batches(session, data_sets.dev, query_report=True)
                # Print report and keep (potentially) calculated WER
                if is_chief:
                    # Print Validation WER report
                    dev_wer = print_report('Validation', results)

                    # Append to WER log
                    log_wer('dev', dev_wer)

            if is_chief:
                # Finishing last epoch
                overall_time = stopwatch(overall_time)
                print "FINISHED Epoch", '%04d' % (last_epoch),\
                      "  Overall epoch time:", format_duration(overall_time),\
                      "  Training time:", format_duration(train_time)
                print

        if is_chief:
            # Indicate optimization has concluded
            global_time = stopwatch(global_time)
            print "FINISHED Optimization",\
                  "  Overall time:", format_duration(global_time),\
                  "  Training time:", format_duration(global_train_time)
            print


    print ('Session closed.')


def export():
    r"""
    Restores the trained variables into a simpler graph that will be exported for serving.
    """
    print ('Exporting the model...')
    with tf.device('/cpu:0'):

        tf.reset_default_graph()
        session = tf.Session(config=session_config)

        # Run inference

        # Input tensor will be of shape [batch_size, n_steps, n_input + 2*n_input*n_context]
        input_tensor = tf.placeholder(tf.float32, [None, None, n_input + 2*n_input*n_context], name='input_node')

        # Calculate input sequence length. This is done by tiling n_steps, batch_size times.
        # If there are multiple sequences, it is assumed they are padded with zeros to be of
        # the same length.
        n_items  = tf.slice(tf.shape(input_tensor), [0], [1])
        n_steps = tf.slice(tf.shape(input_tensor), [1], [1])
        seq_length = tf.tile(n_steps, n_items)

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
        print 'Restored checkpoint at training epoch %d' % (int(checkpoint_path.split('-')[-1]) + 1)

        # Initialise the model exporter and export the model
        model_exporter.init(session.graph.as_graph_def(),
                            named_graph_signatures = {
                                'inputs': exporter.generic_signature(
                                    { 'input': input_tensor }),
                                'outputs': exporter.generic_signature(
                                    { 'outputs': decoded})})
        if FLAGS.remove_export:
            actual_export_dir = os.path.join(FLAGS.export_dir, '%08d' % FLAGS.export_version)
            if os.path.isdir(actual_export_dir):
                print 'Removing old export'
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

            print 'Models exported at %s' % (FLAGS.export_dir)
        except RuntimeError:
            print sys.exc_info()[1]


def main(_) :

    # Determine, if we are the chief worker
    global is_chief
    is_chief = len(FLAGS.worker_hosts) == 0 or (FLAGS.task_index == 0 and FLAGS.job_name == 'worker')

    # Set default relative data set directory based on importer name
    if FLAGS.dataset_path == '':
        FLAGS.dataset_path = os.path.join('./data', FLAGS.importer)

    # Lazy-import data set module
    global importer_module
    importer_module = importlib.import_module('util.importers.%s' % FLAGS.importer)


    from util.website import maybe_publish
    if FLAGS.fulltrace:
        check_cupti()

    # ps and worker hosts required for p2p cluster setup
    FLAGS.ps_hosts = filter(len, FLAGS.ps_hosts.split(","))
    FLAGS.worker_hosts = filter(len, FLAGS.worker_hosts.split(","))

    # Create a cluster from the parameter server and worker hosts.
    global cluster
    cluster = tf.train.ClusterSpec({"ps": FLAGS.ps_hosts, "worker": FLAGS.worker_hosts})

    # The device path base for this node
    global worker_device
    worker_device = '/job:%s/task:%d' % (FLAGS.job_name, FLAGS.task_index)

    # This node's CPU device
    global cpu_device
    cpu_device = worker_device + '/cpu:0'

    # This node's available GPU devices
    global available_devices
    available_devices = [worker_device + gpu for gpu in get_available_gpus()]

    # If there is no GPU available, we fall back to CPU based operation
    if 0 == len(available_devices):
        available_devices = [cpu_device]

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
        FLAGS.checkpoint_dir = xdg.save_data_path('deepspeech')

    # Current log sub-directory
    global log_dir
    log_dir = os.path.join(FLAGS.logs_dir, time.strftime("%Y%m%d-%H%M%S"))

    # Standard session configuration that'll be used for all new sessions.
    global session_config
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_placement)

    # Assign default values for standard deviation
    for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
        val = getattr(FLAGS, '%s_stddev' % var)
        if val is None:
            setattr(FLAGS, '%s_stddev' % var, FLAGS.default_stddev)

    # Queues that are used to gracefully stop parameter servers.
    # Each queue stands for one ps. A finishing worker sends a token to each queue befor joining/quitting.
    # Each ps will dequeue as many tokens as there are workers before joining/quitting.
    # This ensures parameter servers won't quit, if still required by at least one worker and
    # also won't wait forever (like with a standard `server.join()`).
    done_queues = []
    for i, ps in enumerate(FLAGS.ps_hosts):
        # Queues are hosted by their respective owners
        with tf.device('/job:ps/task:%d' % i):
            done_queues.append(tf.FIFOQueue(1, tf.int32, shared_name=('queue%i' % i)))

    # Placeholder to pass in the worker's index as token
    global token_placeholder
    token_placeholder = tf.placeholder(tf.int32)

    # Enqueue operations for each parameter server
    global done_enqueues
    done_enqueues = [queue.enqueue(token_placeholder) for i, queue in enumerate(done_queues)]

    # Dequeue operations for each parameter server
    global done_dequeues
    done_dequeues = [queue.dequeue() for queue in done_queues]

    if len(FLAGS.worker_hosts) == 0:
        # Only one local task: this process (default case - no cluster)
        train()
        print "Done."

    else:
        # Create and start a server for the local task.
        server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

        if FLAGS.job_name == 'ps':
            # We are a parameter server and therefore we just wait for all workers to finish
            # by waiting for their stop tokens.
            with tf.Session(server.target) as session:
                for worker in FLAGS.worker_hosts:
                    print ('Waiting for stop token...')
                    token = session.run(done_dequeues[FLAGS.task_index])
                    print ('Got a stop token from worker %i' %token)
            print ('Session closed.')

        elif FLAGS.job_name == 'worker':
            # We are a worker and therefore we have to do some work.

            # Assigns ops to the local worker by default.
            with tf.device(tf.train.replica_device_setter(
                           worker_device=worker_device,
                           cluster=cluster)):

                # Do the training
                train(server)

        print ('Server stopped.')

    # Are we the main process?
    if is_chief:
        # Doing solo/post-processing work just on the main process...

        # Exporting the model
        if FLAGS.export_dir:
            export()


if __name__ == '__main__' :
    tf.app.run()
