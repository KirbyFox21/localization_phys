import tensorcircuit as tc
import numpy as np
import matplotlib.pyplot as plt
import time

def fibonacci(n):
    if not(type(n) == int) or n < 0:
        return
    if n == 0 or n == 1:
        return 1
    else:
        f = fibonacci(n-1) + fibonacci(n-2)
        return f
# F14 = 610

def gen_H_Raman_real_nambu(L, tso, Mz, beta, t0=1, phi=1):
    Ham = np.zeros((2 * 2 * L, 2 * 2 * L), dtype=np.complex128)
    
    for i in range(L - 1):
        Ham[2 * i + 2, 2 * i] = t0  # 0, 2, 4, ... down
        Ham[2 * i + 3, 2 * i + 1] = - t0  # 1, 3, 5, ... up
        Ham[2 * i + 1, 2 * i + 2] = tso
        Ham[2 * i + 3, 2 * i] = - tso

        Ham[2 * i + 2 + 2 * L, 2 * i + 2 * L] = - t0
        Ham[2 * i + 3 + 2 * L, 2 * i + 1 + 2 * L] = t0
        Ham[2 * i + 1 + 2 * L, 2 * i + 2 + 2 * L] = - tso
        Ham[2 * i + 3 + 2 * L, 2 * i + 2 * L] = tso
    Ham[0, 2 * (L - 1)] = t0  # PBC条件
    Ham[1, 2 * (L - 1) + 1] = - t0
    Ham[2 * (L - 1) + 1, 0] = tso
    Ham[1, 2 * (L - 1)] = - tso

    Ham[0 + 2 * L, 2 * (L - 1) + 2 * L] = t0
    Ham[1 + 2 * L, 2 * (L - 1) + 1 + 2 * L] = - t0
    Ham[2 * (L - 1) + 1 + 2 * L, 0 + 2 * L] = tso
    Ham[1 + 2 * L, 2 * (L - 1) + 2 * L] = - tso
    Ham += Ham.conj().T  # 加上H.c.
    for i in range(L):  # 准周期势
        Ham[2 * i, 2 * i] = - Mz * np.cos(2 * np.pi * beta * (i + 1) + phi)  # 下标从0开始，但是格点从1开始，所以i + 1
        Ham[2 * i + 1, 2 * i + 1] = Mz * np.cos(2 * np.pi * beta * (i + 1) + phi)
        
        Ham[2 * i + 2 * L, 2 * i + 2 * L] = - Ham[2 * i, 2 * i]
        Ham[2 * i + 1 + 2 * L, 2 * i + 1 + 2 * L] = - Ham[2 * i + 1, 2 * i + 1] 
    return Ham