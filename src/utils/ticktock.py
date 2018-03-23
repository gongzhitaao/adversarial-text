from timeit import default_timer
from functools import wraps
import logging


__all__ = ['Timer', 'tick']


logger = logging.getLogger(__name__)
info = logger.info


class Timer(object):
    def __init__(self, msg='timer starts', timer=default_timer, factor=1,
                 fmt='elapsed {:.4f}s'):
        self.timer = timer
        self.factor = factor
        self.fmt = fmt
        self.end = None
        self.msg = msg

    def __call__(self):
        """
        Return the current time
        """
        return self.timer()

    def __enter__(self):
        """
        Set the start time
        """
        info(self.msg)
        self.start = self()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Set the end time
        """
        self.end = self()
        info(str(self))

    def __repr__(self):
        return self.fmt.format(self.elapsed)

    @property
    def elapsed(self):
        if self.end is None:
            # if elapsed is called in the context manager scope
            return (self() - self.start) * self.factor
        else:
            # if elapsed is called out of the context manager scope
            return (self.end - self.start) * self.factor


def tick(f):
    """Simple context timer.
    """
    @wraps(f)
    def wrapper(*args, **kw):
        start = default_timer()
        res = f(*args, **kw)
        end = default_timer()
        info('{0} elapsed: {1:.4f}s'.format(f.__name__, end-start))
        return res
    return wrapper
