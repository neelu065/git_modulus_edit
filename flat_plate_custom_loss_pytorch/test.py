import concurrent.futures
import multiprocessing
import time

number = 10
import os
#
#
# def fun(i):
#     # i = args[0], n = args[1]
#     print('pointwise_wing_{} ...\n'.format(i + 1))
#     # time.sleep(1)
#     return i + 1
#
# from multiprocessing import Pool
# pool = Pool(int(os.popen('nproc').read()))
# results = pool.map(fun, [i for i in range(len(self.neighbour)])
#
# print(results)

# with multiprocessing.Pool(processes=3) as pool:
#     results = pool.starmap(fun, [i for i in range(number)])
# print(results)
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     results = [executor.submit(fun, i) for i in range(number)]
#
# print(results)
# from multiprocessing import Pool, freeze_support
# from functools import partial
# def func(a, b):
#     return a + b
#
# def main():
#     a_args = [1,2,3]  # i range
#     second_arg = 1  # self
#     with Pool() as pool:
#         L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
#         N = pool.map(partial(func, b=second_arg), a_args)
#     return print(N)
#
# if __name__=="__main__":
#     # freeze_support()
#     main()

# def host(id):
#     import socket
#     return "Rank: %d -- %s" % (id, socket.gethostname())


# if __name__ == '__main__':
#     from pathos.pools import ThreadPool as TPool
#     tpool = TPool()
#
#     print("Evaluate 10 items on 1 thread")
#     tpool.nthreads = 1
#     res3 = tpool.map(host, range(10))
#     print(tpool)
#     print('\n'.join(res3))
#     print('')
import numpy as np
def abc():
    band = np.random.random(1)

    def de(i):
        return [band + i, band + i + 1, band + i + 2], [band + i, band + i + 1, band + i + 2]

    from pathos.pools import ThreadPool as TPool
    tpool = TPool()

    print("Evaluate 10 items on 1 thread")
    tpool.nthreads = 20
    res3 = tpool.map(de, range(200))

    return res3

xy = abc()