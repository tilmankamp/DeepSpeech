from __future__ import print_function
import sys

class Logger(object):
    def __init__(self, module_prefix=None):
        self.module_prefix = module_prefix

    def prefix_print(self, prefix, message, is_error=False):
        prefix = prefix + ' '
        if self.module_prefix:
            prefix = prefix + '[' + self.module_prefix + '] '
        text = prefix + ('\n' + prefix).join(message.split('\n'))
        print(text, file=sys.stdout if is_error else sys.stderr)

    def debug(self, message):
        if FLAGS.log_level == 0:
            self.prefix_print('D', str(message))

    def info(self, message):
        if FLAGS.log_level <= 1:
            self.prefix_print('I', str(message))

    def warn(self, message):
        if FLAGS.log_level <= 2:
            self.prefix_print('W', str(message))

    def error(self, message):
        if FLAGS.log_level <= 3:
            self.prefix_print('E', str(message), is_error=True)

    def traffic(self, message):
        if FLAGS.log_traffic:
            self.prefix_print('T', str(message))
