#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import json
import numpy as np
import os
import shutil
import subprocess
import sys
import tempfile
import tensorflow as tf
import time

from collections import OrderedDict
from math import ceil
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.ops import ctc_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from util.gpu import get_available_gpus
from util.log import merge_logs
from util.spell import correction
from util.shared_lib import check_cupti
from util.text import sparse_tensor_value_to_texts, wer
from util.data_set_helpers import SwitchableDataSet
from xdg import BaseDirectory as xdg

ds_importer = os.environ.get('ds_importer', 'ldc93s1')
ds_dataset_path = os.environ.get('ds_dataset_path', os.path.join('./data', ds_importer))

import importlib
ds_importer_module = importlib.import_module('util.importers.%s' % ds_importer)

from util.website import maybe_publish

do_fulltrace = bool(len(os.environ.get('ds_do_fulltrace', '')))

if do_fulltrace:
    check_cupti()



# Cluster configuration
# =====================

# Comma separated list of hostname:port pairs
ps_hosts = os.environ.get('ds_ps_hosts', 'localhost:2223').split(",")

# Comma separated list of hostname:port pairs
worker_hosts = os.environ.get('ds_worker_hosts', 'localhost:2222').split(",")

# One of 'ps', 'worker', if cluster operation is desired. '' for single operation.
job_name = os.environ.get('ds_job_name', 'worker')

# Index of task within the job - worker with index 0 will be the chief
task_index = int(os.environ.get('ds_task_index', 0))

# As we are introducing one tower for each GPU, first we must determine how many GPU's are available
# Get a list of the available gpu's ['/gpu:0', '/gpu:1'...]
# available_devices = ['/job:%s/task:%d%s' % (job_name, task_index, gpu) for gpu in get_available_gpus()]

# If there are no GPU's use the CPU
# if 0 == len(available_devices):
#    available_devices = ['/job:%s/task:%d/cpu:0' % (job_name, task_index)]

# Create a cluster from the parameter server and worker hosts.
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

worker_device = '/job:%s/task:%d' % (job_name, task_index)
cpu_device = worker_device + '/cpu:0'
available_devices = [worker_device + gpu for gpu in get_available_gpus()]
if 0 == len(available_devices):
    available_devices = [cpu_device]
print(available_devices)


# Global Constants
# ================

# The number of iterations (epochs) we will train for
epochs = int(os.environ.get('ds_epochs', 75))

# Weather to use GPU bound Warp-CTC
use_warpctc = bool(len(os.environ.get('ds_use_warpctc', '')))

# As we will employ dropout on the feedforward layers of the network,
# we need to define a parameter `dropout_rate` that keeps track of the dropout rate for these layers
dropout_rate = float(os.environ.get('ds_dropout_rate', 0.05))  # TODO: Validate this is a reasonable value

# We allow for customisation of dropout per-layer
dropout_rate2 = float(os.environ.get('ds_dropout_rate2', dropout_rate))
dropout_rate3 = float(os.environ.get('ds_dropout_rate3', dropout_rate))
dropout_rate4 = float(os.environ.get('ds_dropout_rate4', 0.0))
dropout_rate5 = float(os.environ.get('ds_dropout_rate5', 0.0))
dropout_rate6 = float(os.environ.get('ds_dropout_rate6', dropout_rate))

dropout_rates = [ dropout_rate,
                  dropout_rate2,
                  dropout_rate3,
                  dropout_rate4,
                  dropout_rate5,
                  dropout_rate6 ]
no_dropout = [ 0.0 ] * 6

# One more constant required of the non-recurrant layers is the clipping value of the ReLU.
relu_clip = int(os.environ.get('ds_relu_clip', 20)) # TODO: Validate this is a reasonable value


# Adam optimizer (http://arxiv.org/abs/1412.6980) parameters

# Beta 1 parameter
beta1 = float(os.environ.get('ds_beta1', 0.9)) # TODO: Determine a reasonable value for this

# Beta 2 parameter
beta2 = float(os.environ.get('ds_beta2', 0.999)) # TODO: Determine a reasonable value for this

# Epsilon parameter
epsilon = float(os.environ.get('ds_epsilon', 1e-8)) # TODO: Determine a reasonable value for this

# Learning rate parameter
learning_rate = float(os.environ.get('ds_learning_rate', 0.001))


# Batch sizes

# The number of elements in a training batch
train_batch_size = int(os.environ.get('ds_train_batch_size', 1))

# The number of elements in a dev batch
dev_batch_size = int(os.environ.get('ds_dev_batch_size', 1))

# The number of elements in a test batch
test_batch_size = int(os.environ.get('ds_test_batch_size', 1))


# Sample limits

# The maximum amount of samples taken from (the beginning of) the train set - 0 meaning no limit
limit_train = int(os.environ.get('ds_limit_train', 0))

# The maximum amount of samples taken from (the beginning of) the validation set - 0 meaning no limit
limit_dev   = int(os.environ.get('ds_limit_dev',   0))

# The maximum amount of samples taken from (the beginning of) the test set - 0 meaning no limit
limit_test  = int(os.environ.get('ds_limit_test',  0))


# Step widths

# The number of epochs we cycle through before displaying progress
display_step = int(os.environ.get('ds_display_step', 1))

# The number of epochs we cycle through before checkpointing the model
checkpoint_step = int(os.environ.get('ds_checkpoint_step', 5))

# The number of epochs we cycle through before validating the model
validation_step = int(os.environ.get('ds_validation_step', 0))


# Checkpointing

# The directory in which checkpoints are stored
checkpoint_dir = os.environ.get('ds_checkpoint_dir', xdg.save_data_path('deepspeech'))

# Whether to resume from checkpoints when training
restore_checkpoint = bool(int(os.environ.get('ds_restore_checkpoint', 0)))

# The directory in which exported models are stored
export_dir = os.environ.get('ds_export_dir', None)

# The version number of the exported model
export_version = 1

# Whether to remove old exported models
remove_export = bool(int(os.environ.get('ds_remove_export', 0)))


# Reporting

# The number of phrases to print out during a WER report
report_count = int(os.environ.get('ds_report_count', 10))

# Whether to log device placement of the operators to the console
log_device_placement = bool(int(os.environ.get('ds_log_device_placement', 0)))

# Whether to log gradients and variables summaries to TensorBoard during training.
# This incurs a performance hit, so should generally only be enabled for debugging.
log_variables = bool(len(os.environ.get('ds_log_variables', '')))


# Initialization

# The default random seed that is used to initialize variables. Ensures reproducibility.
random_seed = int(os.environ.get('ds_random_seed', 4567)) # To be adjusted in case of bad luck

# The default standard deviation to use when initialising weights and biases
default_stddev = float(os.environ.get('ds_default_stddev', 0.046875))

# Individual standard deviations to use when initialising particular weights and biases
for var in ['b1', 'h1', 'b2', 'h2', 'b3', 'h3', 'b5', 'h5', 'b6', 'h6']:
    locals()['%s_stddev' % var] = float(os.environ.get('ds_%s_stddev' % var, default_stddev))

# Session settings

# Standard session configuration that'll be used for all new sessions.
session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement)


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
    # Use the /cpu:0 device for scoped operations
    with tf.device(tf.train.replica_device_setter(
                   worker_device=worker_device,
                   cluster=cluster)):
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
    b1 = variable_on_cpu('b1', [n_hidden_1], tf.random_normal_initializer(stddev=b1_stddev))
    h1 = variable_on_cpu('h1', [n_input + 2*n_input*n_context, n_hidden_1], tf.random_normal_initializer(stddev=h1_stddev))
    layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
    layer_1 = tf.nn.dropout(layer_1, (1.0 - dropout[0]))

    # 2nd layer
    b2 = variable_on_cpu('b2', [n_hidden_2], tf.random_normal_initializer(stddev=b2_stddev))
    h2 = variable_on_cpu('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(stddev=h2_stddev))
    layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), relu_clip)
    layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[1]))

    # 3rd layer
    b3 = variable_on_cpu('b3', [n_hidden_3], tf.random_normal_initializer(stddev=b3_stddev))
    h3 = variable_on_cpu('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=h3_stddev))
    layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), relu_clip)
    layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[2]))

    # Now we create the forward and backward LSTM units.
    # Both of which have inputs of length `n_cell_dim` and bias `1.0` for the forget gate of the LSTM.

    # Forward direction cell:
    lstm_fw_cell = core_rnn_cell.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
    lstm_fw_cell = core_rnn_cell.DropoutWrapper(lstm_fw_cell,
                                                input_keep_prob=1.0 - dropout[3],
                                                output_keep_prob=1.0 - dropout[3],
                                                seed=random_seed)
    # Backward direction cell:
    lstm_bw_cell = core_rnn_cell.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = core_rnn_cell.DropoutWrapper(lstm_bw_cell,
                                                input_keep_prob=1.0 - dropout[4],
                                                output_keep_prob=1.0 - dropout[4],
                                                seed=random_seed)

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
    b5 = variable_on_cpu('b5', [n_hidden_5], tf.random_normal_initializer(stddev=b5_stddev))
    h5 = variable_on_cpu('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=h5_stddev))
    layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), relu_clip)
    layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

    # Now we apply the weight matrix `h6` and bias `b6` to the output of `layer_5`
    # creating `n_classes` dimensional vectors, the logits.
    b6 = variable_on_cpu('b6', [n_hidden_6], tf.random_normal_initializer(stddev=b6_stddev))
    h6 = variable_on_cpu('h6', [n_hidden_5, n_hidden_6], tf.random_normal_initializer(stddev=h6_stddev))
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
    if use_warpctc:
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
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)
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
            with tf.device(tf.train.replica_device_setter(
                           worker_device=available_devices[i],
                           cluster=cluster)):
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


# Finally we define the top directory for all logs and our current log sub-directory of it.
# We also add some log helpers.
logs_dir = os.environ.get('ds_logs_dir', 'logs')
log_dir = '%s/%s' % (logs_dir, time.strftime("%Y%m%d-%H%M%S"))

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

def get_git_branch():
    return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip()


# Helpers
# =======

def calculate_and_print_wer_report(caption, results_tuple):
    r"""
    This routine will print a WER report with a given caption.
    It'll print the `mean` WER plus summaries of the ``ds_report_count`` top lowest
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
    items = items[:report_count]

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

def print_report(caption, batch_set_result):
    r"""
    This routine will print a report from a provided batch set result tuple.
    It takes a caption for titling the output plus the batch set result tuple.
    If the batch set result tuple contains accuracy and a report results tuple,
    a complete WER report will be calculated, printed and its mean WER returned.
    Otherwise it will just print the loss and return ``None``.
    """
    # Unpacking batch set result tuple
    loss, accuracy, results_tuple = batch_set_result

    # We always have a loss value
    title = caption + " loss=" + "{:.9f}".format(loss)

    mean_wer = None
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
    m, s = divmod(duration.seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


# Execution
# =========


# To run our different graphs in separate sessions,
# we first need to create some common infrastructure.
#
# At first we introduce the functions `read_data_sets` and `read_data_set` to read in data sets.
# The first returns a `DataSets` object of the selected importer, containing all available sets.
# The latter takes the name of the required data set
# (`'train'`, `'dev'` or `'test'`) as string and returns the respective set.
def read_data_sets(set_names=['train', 'dev', 'test']):
    r"""
    Returns a :class:`DataSets` object of the selected importer, containing all available sets.
    """
    # Obtain all the data sets
    return ds_importer_module.read_data_sets(ds_dataset_path,
                                             train_batch_size,
                                             dev_batch_size,
                                             test_batch_size,
                                             n_input,
                                             n_context,
                                             limit_dev=limit_dev,
                                             limit_test=limit_test,
                                             limit_train=limit_train,
                                             sets=set_names)


def calculate_loss_and_report(session, tower_results, feed_dict, total_batches, train_op=None, query_report=False):
    r"""
    Now let's introduce the main routine for training and inference.
    It takes a started execution context (given by ``execution_context``),
    a ``Session`` (``session``), an optional epoch index (``epoch``)
    and a flag (``query_report``) which indicates whether to calculate the WER
    report data or not.
    Its main duty is to iterate over all batches and calculate the mean loss.
    If a non-negative epoch is provided, it will also optimize the parameters.
    If ``query_report`` is ``False``, the default, it will return a tuple which
    contains the mean loss.
    If ``query_report`` is ``True``, the mean accuracy and individual results
    are also included in the returned tuple.
    """

    results_params, _, avg_accuracy, avg_loss = tower_results

    batches_per_device = ceil(float(total_batches) / len(available_devices))

    total_loss = 0.0
    params = OrderedDict()
    params['avg_loss'] = avg_loss

    # Facilitate a train run
    if not train_op is None:
        params['train_op'] = train_op

    # Requirements to display a WER report
    if query_report:
        # Reset accuracy
        total_accuracy = 0.0
        # Create report results tuple
        report_results = ([],[],[],[])
        # Extend the session.run parameters
        params['sample_results'] = [results_params, avg_accuracy]

    # Get the index of each of the session fetches so we can recover the results more easily
    param_idx = dict(zip(params.keys(), range(len(params))))
    params = params.values()

    # Loop over the batches
    for batch in range(int(batches_per_device)):
        if session.should_stop():
            break
        extra_params = { 'feed_dict': feed_dict }

        # Compute the batch
        result = session.run(params, **extra_params)

        # Add batch to loss
        total_loss += result[param_idx['avg_loss']]

        if query_report:
            sample_results = result[param_idx['sample_results']]
            # Collect individual sample results
            collect_results(report_results, sample_results[0])
            # Add batch to total_accuracy
            total_accuracy += sample_results[1]

    # Returning the batch set result tuple
    loss = total_loss / batches_per_device
    if query_report:
        return (loss, total_accuracy / batches_per_device, report_results)
    else:
        return (loss, None, None)



if __name__ == '__main__':

    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    # Queues that are used to gracefully stop parameter servers.
    # Each queue stands for one ps. A finished worker sends a token to each queue.
    # Each ps will dequeue as many tokens as there are workers before joining/quitting.
    with tf.device('/job:ps/task:0'):
        done_queues = [tf.FIFOQueue(1, tf.int32, shared_name=('queue%i' % i)) for i, ps in enumerate(ps_hosts)]
    token_placeholder = tf.placeholder(tf.int32)
    done_enqueues = [queue.enqueue(token_placeholder) for i, queue in enumerate(done_queues)]
    done_dequeues = [queue.dequeue() for queue in done_queues]

    if job_name == 'ps':
        with tf.Session(server.target) as session:
            for worker in worker_hosts:
                print ('Waiting for stop token...')
                token = session.run(done_dequeues[task_index])
                print ('Got a stop token from worker %i' %token)
        print ('Session stopped.')

    elif job_name == 'worker':

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                       worker_device=worker_device,
                       cluster=cluster)):

            # Determine, if we are the chief worker
            is_chief = (task_index == 0)

            # Create a variable to hold the global_step.
            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Read all data sets
            data_sets = read_data_sets()

            # Get the data sets
            switchable_data_set = SwitchableDataSet(data_sets)

            # Create the optimizer
            optimizer = create_optimizer()

            # Synchronous training
            optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                       replicas_to_aggregate=len(worker_hosts),
                                                       total_num_replicas=len(worker_hosts))

            # Get the data_set specific graph end-points
            tower_results = get_tower_results(switchable_data_set, optimizer)
            results, gradients, accuracy, loss = tower_results

            # Average tower gradients across GPUs
            avg_tower_gradients = average_gradients(gradients)

            # Apply gradients to modify the model
            apply_gradient_op = optimizer.apply_gradients(avg_tower_gradients, global_step=global_step)

            # Hook to handle initialization and queues for sync replicas.
            sync_replicas_hook = optimizer.make_session_run_hook(is_chief)

            class CoordHook(tf.train.SessionRunHook):
                def after_create_session(self, session, coord):
                    print ('Starting queue runners...')
                    self.threads = switchable_data_set.start_queue_threads(session, coord)
                    print ('Queue runners started.')
                def end(self, session):
                    # Closing the data_set queues
                    switchable_data_set.close_queue(session)
                    # Sending a 'done' token to each parameter server.
                    for enqueue in done_enqueues:
                        print ('Sending stop token to ps...')
                        session.run(enqueue, feed_dict={ token_placeholder: task_index })
                        print ('Sent stop token to ps.')


            # The StopAtStepHook handles stopping after running given steps.
            hooks=[tf.train.StopAtStepHook(last_step=epochs), CoordHook(), sync_replicas_hook]

            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=is_chief,
                                                   checkpoint_dir=checkpoint_dir,
                                                   hooks=hooks,
                                                   save_checkpoint_secs=10,
                                                   config=session_config) as session:
                print ('Starting training...')
                while not session.should_stop():

                    feed_dict = {}
                    switchable_data_set.set_data_set(feed_dict, data_sets.train)

                    # Run one epoch
                    results = calculate_loss_and_report(session,
                                                        tower_results,
                                                        feed_dict,
                                                        data_sets.train.total_batches,
                                                        train_op=apply_gradient_op,
                                                        query_report=True)

                    # Print WER report
                    print_report('Training', results)

                    feed_dict = {}
                    switchable_data_set.set_data_set(feed_dict, data_sets.dev)

                    # Run one epoch
                    results = calculate_loss_and_report(session,
                                                        tower_results,
                                                        feed_dict,
                                                        data_sets.dev.total_batches,
                                                        query_report=True)

                    # Print WER report
                    print_report('Validation', results)

                print ('Training stopped.')
            print ('Session stopped.')
    print ('Server stopped.')
