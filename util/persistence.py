import os
import re
import glob
import pandas
import tensorflow as tf
import time
from six.moves import range
from util.log import Logger

log = Logger('Persistence')

class CheckpointManager(object):
    def __init__(self, checkpoint_dir, load='recent', inter_secs=0, keep_n_inters=3, keep_n_epochs=5, init_op=None, saver=None):
        self.checkpoint_dir = checkpoint_dir
        self.load = load
        self.inter_secs = inter_secs
        self.keep_n_inters = keep_n_inters
        self.keep_n_epochs = keep_n_epochs
        self.init_op = init_op if init_op else tf.global_variables_initializer()
        self.saver = saver if saver else tf.train.Saver()
        self._str_inter = 'intermediate'
        self._str_epoch = 'epoch'
        self._t0 = None
        if checkpoint_dir:
            self._csv = os.path.join(checkpoint_dir, 'results.csv')
            if os.path.isfile(self._csv):
                self._results = pandas.read_csv(self._csv)
            else:
                self._results = pandas.DataFrame(columns=['loss', 'dev-loss'])
                self._results.index.rename('epoch', inplace=True)

    def _get_numbered_files(self, kind):
        entries = []
        entries_lookup = {}
        for filename in glob.glob(self.checkpoint_dir + '/*'):
            number = re.search('%s_([0-9]+)\\.ckpt.*' % kind, filename, re.IGNORECASE)
            if number:
                number = int(number.group(1))
                if number in entries_lookup:
                    _, files = entries_lookup[number]
                else:
                    files = []
                    entry = (number, files)
                    entries_lookup[number] = entry
                    entries.append(entry)
                files.append(filename)
        return sorted(entries, key=lambda e: e[0])

    def _prune_and_save(self, session, kind, keep, new_index=-1):
        _files = self._get_numbered_files(kind)
        if new_index < 0:
            # compute the next index
            new_index = _files[-1][0] + 1 if len(_files) > 0 else 1
        _filtered_files = []
        for number, filenames in _files:
            if new_index <= number:
                log.debug('Removing %s checkpoint files "%s.*", as its number is greater than %d...' % (kind, filename, new_index))
                for filename in filenames:
                    os.remove(filename)
            else:
                _filtered_files.append((number, filenames))
        for number, filenames in _filtered_files[:len(_filtered_files) - keep]:
            log.debug('Removing files of %s checkpoint %d, as only %d %s checkpoints are kept...' % (kind, number, keep, kind))
            for filename in filenames:
                os.remove(filename)
        filename = os.path.join(self.checkpoint_dir, '%s_%d.ckpt' % (kind, new_index))
        log.info('Checkpointing %s %d as "%s"...' % (kind, new_index, filename))
        self.saver.save(session, filename, write_state=False)

    def _init(self, session):
        log.info('Initializing graph')
        session.run(self.init_op)

    def start(self, session):
        if not self.checkpoint_dir:
            self._init(session)
            return
        file = None
        epoch_files = self._get_numbered_files(self._str_epoch)
        last_epoch = epoch_files[-1] if len(epoch_files) > 0 else None
        if self.load == 'recent':
            kind = self._str_inter
            files = self._get_numbered_files(self._str_inter)
            file = files[-1] if len(files) > 0 else last_epoch
        else:
            kind = self._str_epoch
            if self.load == 'best-dev':
                number = self._results.ix[self._results['dev-loss'].idxmin()]['epoch']
                if number:
                    file = (number, os.path.join(checkpoint_dir, '%s_%d.ckpt' % (kind, number)))
            elif self.load == 'last-epoch':
                file = last_epoch
        if file:
            number, filenames = file
            filename = '.'.join(filenames[0].split('.')[:-1])
            log.info('Restoring %s %d from checkpoint file "%s"...' % (kind, number, filename))
            self.saver.restore(session, filename)
        elif os.path.isfile(self.load):
            log.info('Restoring checkpoint file "%s"...' % self.load)
            self.saver.restore(session, self.load)
        else:
            self._init(session)
        self._t0 = time.time()

    def _step(self, session):
        self._prune_and_save(session, self._str_inter, self.keep_n_inters)

    def step(self, session):
        if not self.checkpoint_dir:
            return
        current_time = time.time()
        if self.inter_secs > 0 and current_time - self._t0 > self.inter_secs:
            self._t0 = current_time
            self._step(session)

    def epoch(self, session, epoch_number, loss, dev_loss=None):
        if not self.checkpoint_dir:
            return
        if self.inter_secs > 0:
            self._step(session)
        self._prune_and_save(session, self._str_epoch, self.keep_n_epochs, new_index=epoch_number)
        log.debug('Updating "%s"...' % self._csv)
        # removing higher epoch numbers from log, as they got pruned/overwritten
        self._results = self._results.drop(self._results[self._results.index > epoch_number].index)
        self._results.loc[epoch_number] = [loss, dev_loss]
        self._results.to_csv(self._csv)

