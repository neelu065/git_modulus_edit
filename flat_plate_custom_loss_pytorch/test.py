import concurrent.futures
import multiprocessing
import time
import torch
# number = 10
# import os
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
# import numpy as np
# def abc():
#     band = np.random.random(1)
#
#     def de(i):
#         return [band + i, band + i + 1, band + i + 2], [band + i, band + i + 1, band + i + 2]
#
#     from pathos.pools import ThreadPool as TPool
#     tpool = TPool()
#
#     print("Evaluate 10 items on 1 thread")
#     tpool.nthreads = 20
#     res3 = tpool.map(de, range(200))
#
#     return res3
#
# xy = abc()

import torch


# Define the method you want to create a CUDA graph for
# def my_method(input):
#     output = input * 2 + 1
#     return output
#
#
# g = torch.cuda.CUDAGraph()
# # Enable CUDA graph recording
# with torch.cuda.graph(g):
#     # Create a tensor on the GPU
#     input = torch.randn(3, 3, device='cpu')
#
#     # Apply the method to the tensor
#     output = my_method(input)
#
# # Print the CUDA graph
# print(output.grad_fn.graph)

# import torch
#
# # Create a CUDA graph
# # graph = torch.cuda.create_graph()
#
# # Perform CUDA operations within the graph context
# # with torch.cuda.graph(graph):
# #     x = torch.randn(3, 3).cuda()
# #     y = x + 2
# #     z = y.sum()
#
# # device='cuda' if torch.cuda.is_available() else 'cpu'
# # torch.cuda.set_device(device)
# # Create a CUDA graph
# # with torch.cuda.device(device):
# #     graph = torch.cuda.current_device().create_graph()
# graph = torch.cuda.create_graph()
# # Define some tensors
# a = torch.randn(3, 3, device='cuda')
# b = torch.randn(3, 3, device='cuda')
# c = torch.randn(3, 3, device='cuda')
#
# # Add the operations to the graph
# with graph.record():
#     d = a + b
#     e = d * c
#
# # Execute the graph
# with torch.cuda.device(0):
#     torch.cuda.current_device().execute(graph)
#
# # Print the result
# print(e)


# N, D_in, H, D_out = 640, 4096, 2048, 1024
# model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
#                             torch.nn.Dropout(p=0.2),
#                             torch.nn.Linear(H, D_out),
#                             torch.nn.Dropout(p=0.1)).cuda()
# loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#
# # Placeholders used for capture
# static_input = torch.randn(N, D_in, device='cuda')
# static_target = torch.randn(N, D_out, device='cuda')
#
# # warmup
# # Uses static_input and static_target here for convenience,
# # but in a real setting, because the warmup includes optimizer.step()
# # you must use a few batches of real data.
# s = torch.cuda.Stream()
# s.wait_stream(torch.cuda.current_stream())
# with torch.cuda.stream(s):
#     for i in range(3):
#         optimizer.zero_grad(set_to_none=True)
#         y_pred = model(static_input)
#         loss = loss_fn(y_pred, static_target)
#         loss.backward()
#         optimizer.step()
# torch.cuda.current_stream().wait_stream(s)
#
# # capture
# g = torch.cuda.CUDAGraph()
# # Sets grads to None before capture, so backward() will create
# # .grad attributes with allocations from the graph's private pool
# optimizer.zero_grad(set_to_none=True)
# with torch.cuda.graph(g):
#     static_y_pred = model(static_input)
#     static_loss = loss_fn(static_y_pred, static_target)
#     static_loss.backward()
#     optimizer.step()
#
# real_inputs = [torch.rand_like(static_input) for _ in range(10)]
# real_targets = [torch.rand_like(static_target) for _ in range(10)]
#
# for data, target in zip(real_inputs, real_targets):
#     # Fills the graph's input memory with new data to compute on
#     static_input.copy_(data)
#     static_target.copy_(target)
#     # replay() includes forward, backward, and step.
#     # You don't even need to call optimizer.zero_grad() between iterations
#     # because the captured backward refills static .grad tensors in place.
#     g.replay()
#     # Params have been updated. static_y_pred, static_loss, and .grad
#     # attributes hold values from computing on this iteration's data.
#
import optuna

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    y = trial.suggest_uniform('y', -10, 10)
    # z1 = x ** 2 + y ** 2
    z2 = (x - 1) ** 2 + y ** 2
    return z2

study = optuna.create_study(directions=['minimize']) #, 'minimize'])
study.optimize(objective, n_trials=100)
