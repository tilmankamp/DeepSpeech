
import os
import time
from multiprocessing.dummy import Pool as ThreadPool


KILO = 1024
KILOBYTE = 1 * KILO
MEGABYTE = KILO * KILOBYTE
GIGABYTE = KILO * MEGABYTE
TERABYTE = KILO * GIGABYTE
SIZE_PREFIX_LOOKUP = {'k': KILOBYTE, 'm': MEGABYTE, 'g': GIGABYTE, 't': TERABYTE}


def parse_file_size(file_size):
    file_size = file_size.lower().strip()
    if len(file_size) == 0:
        return 0
    n = int(keep_only_digits(file_size))
    if file_size[-1] == 'b':
        file_size = file_size[:-1]
    e = file_size[-1]
    return SIZE_PREFIX_LOOKUP[e] * n if e in SIZE_PREFIX_LOOKUP else n


def keep_only_digits(txt):
    return ''.join(filter(str.isdigit, txt))


def secs_to_hours(secs):
    hours, remainder = divmod(secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)


class LimitingPool:
    def __init__(self, processes=None, limit_factor=2, sleeping_for=0.1):
        self.processes = os.cpu_count() if processes is None else processes
        self.pool = ThreadPool(processes=processes)
        self.sleeping_for = sleeping_for
        self.max_ahead = self.processes * limit_factor
        self.processed = 0

    def __enter__(self):
        return self

    def limit(self, it):
        for obj in it:
            while self.processed >= self.max_ahead:
                time.sleep(self.sleeping_for)
            self.processed += 1
            yield obj

    def map(self, fun, it):
        for obj in self.pool.imap(fun, self.limit(it)):
            self.processed -= 1
            yield obj

    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.close()


class ExceptionBox:
    def __init__(self):
        self.exception = None

    def raise_if_set(self):
        if self.exception is not None:
            exception = self.exception
            self.exception = None
            raise exception  # pylint: disable = raising-bad-type


def remember_exception(iterable, exception_box=None):
    def do_iterate():
        try:
            for obj in iterable():
                yield obj
        except StopIteration:
            return
        except Exception as ex:  # pylint: disable = broad-except
            exception_box.exception = ex
    return iterable if exception_box is None else do_iterate
