import torch
from contextlib import contextmanager

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

@contextmanager
def count_time(name):
    starter.record()
    yield
    ender.record(); torch.cuda.synchronize()
    print(f"{name}:\t{starter.elapsed_time(ender):>6.2f}ms")