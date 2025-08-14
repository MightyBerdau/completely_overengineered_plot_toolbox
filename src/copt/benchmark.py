import time

def timeit(func, *args, **kwargs):
    start = time.time()
    outputs = func(*args, **kwargs)
    end = time.time()
    elapsed = end - start
    return outputs, elapsed