# The following code is modified from openai/evolution-strategies-starter
# (https://github.com/openai/evolution-strategies-starter)
# under the MIT License.

# Modifications Copyright (c) 2020 Uber Technologies, Inc.


import numpy as np
import logging

logger = logging.getLogger(__name__)

debug = False

# 모든 Pool의 worker network에 해당하는 노이즈를 위해 크기가 커야 함
#
class SharedNoiseTable(object):
    def __init__(self):
        import ctypes
        import multiprocessing
        seed = 42  # fixed
        # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        count = 250000000 if not debug else 1000000
        logger.info('Sampling {} random numbers with seed {}'.format(
            count, seed))
        # ctypes is a foreign function library for Python. It provides C compatible data types, and allows calling functions
        # in DLLs or shared libraries. It can be used to wrap these libraries in pure Python.
        # It is possible to create shared objects using shared memory which can be inherited by child processes.

        # Return a ctypes array allocated from shared memory. By default the return value is actually a synchronized wrapper for the array.
        # int(25e7)
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)

        # ctypes 배열 또는 POINTER에서 numpy 배열을 만듭니다.
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())


        import ctypes
        print(ctypes.sizeof(_shared_mem))

        assert self.noise.dtype == np.float32
        """
        A fixed bit generator using a fixed seed and a fixed series of calls to 'RandomState' methods using the same 
        parameters will always produce the same results up to roundoff error except when the values were incorrect.
        """
        self.noise[:] = np.random.RandomState(seed).randn(
            count)  # 64-bit to 32-bit conversion here
        logger.info('Sampled {} bytes'.format(self.noise.size * 4))

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)
