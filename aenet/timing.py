"""
A timing decorator for profiling and debugging purposes.

Use it like this:

    from strucconv.timing import Timing
    timing = Timing(outfile=YOUR_OUTPUT_FILE)

    @timing
    def your_function(...):
        ....

"""

import sys
import timeit

__author__ = "Nongnuch Artrith and Alexander Urban"
__email__ = "nartrith@atomistic.net"
__date__ = "2015-12-03"
__version__ = "0.1"


class Timing(object):

    def __init__(self, outfile=sys.stdout):
        if hasattr(outfile, 'write'):
            self.fp = outfile
            self._close_at_del = False
        else:
            self.fp = open(outfile, 'w')
            self._close_at_del = True

        self.frmt = "call to `{}' - elapsed time (s): {}\n"
        self.t0 = timeit.default_timer()

    def __del__(self):
        if self._close_at_del:
            self.fp.close()

    def __call__(self, func):
        def wrap(*args, **kwargs):
            t0 = timeit.default_timer()
            result = func(*args, **kwargs)
            dt = timeit.default_timer() - t0
            self.fp.write(self.frmt.format(func.__name__, dt))
            return result
        return wrap

    def reset(self, message=None):
        if message is not None:
            self.fp.write(message + "\n")
        self.t0 = timeit.default_timer()

    def log(self, message):
        dt = timeit.default_timer() - self.t0
        self.fp.write(message + " {}\n".format(dt))
