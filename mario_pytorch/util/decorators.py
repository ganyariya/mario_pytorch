import time
from typing import Callable


def argument_print_decorator(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        print(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


def time_measurement_decorator(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        elapsed = time.time() - start

        print(f"elapsed time: {elapsed} s")
        return ret

    return wrapper
