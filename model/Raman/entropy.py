import tensorcircuit as tc
import numpy as np
import matplotlib.pyplot as plt
import time

model_name = "Raman"
K = tc.set_backend("numpy")
tc.set_dtype("complex128")

def fibonacci(n):
    if not(type(n) == int) or n < 0:
        return
    if n == 0 or n == 1:
        return 1
    else:
        f = fibonacci(n-1) + fibonacci(n-2)
        return f
# F14 = 610

def gen_H_Raman_real_nambu(L, tso, Mz, beta, t0=1, phi=0):
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

    Ham[0 + 2 * L, 2 * (L - 1) + 2 * L] = - t0
    Ham[1 + 2 * L, 2 * (L - 1) + 1 + 2 * L] = t0
    Ham[2 * (L - 1) + 1 + 2 * L, 0 + 2 * L] = - tso
    Ham[1 + 2 * L, 2 * (L - 1) + 2 * L] = tso
    Ham += Ham.conj().T  # 加上H.c.
    for i in range(L):  # 准周期势
        Ham[2 * i, 2 * i] = - Mz * np.cos(2 * np.pi * beta * (i + 1) + phi)  # 下标从0开始，但是格点从1开始，所以i + 1
        Ham[2 * i + 1, 2 * i + 1] = Mz * np.cos(2 * np.pi * beta * (i + 1) + phi)
        
        Ham[2 * i + 2 * L, 2 * i + 2 * L] = - Ham[2 * i, 2 * i]
        Ham[2 * i + 1 + 2 * L, 2 * i + 1 + 2 * L] = - Ham[2 * i + 1, 2 * i + 1] 
    return Ham


def cal_S_of_t(L, tso, Mz_array, beta, steps, dt, t0=1, phi=0):
    S_array = np.zeros((len(Mz_array), steps), dtype=np.complex128)
    sub_system = list(range(2 * L // 2))
    if state_name == "bipartite_state":
        filled_indices = np.arange(0 , 2 * L // 2)
    elif state_name == "neel_state":
        filled_indices = np.arange(1, 2 * L, 2)

    for i, Mz in enumerate(Mz_array):
        print(f"Mz = {Mz:.2f} ({i + 1} / {len(Mz_array)})")
        H_evo = gen_H_Raman_real_nambu(L, tso, Mz, beta, t0, phi)
        system = tc.FGSSimulator(2 * L, filled=filled_indices)

        for j in range(steps):
            S_array[i, j] = system.entropy(sub_system)
            system.evol_ghamiltonian(H_evo * dt)

    S_array /= len(sub_system)  # 记得除以子系统长度！！
    file_name = f"S_{model_name}_{state_name}_L_{L}_Mz_{Mz_array[0]}_{Mz_array[-1]}_steps_{steps}_dt_{dt}"
    np.savez("data/" + file_name + ".npz", S_array=S_array)


def vis_S_of_t(L, Mz_array, steps, dt):
    file_name = f"S_{model_name}_{state_name}_L_{L}_Mz_{Mz_array[0]}_{Mz_array[-1]}_steps_{steps}_dt_{dt}"
    data = np.load("data/" + file_name + ".npz")
    S_array = data["S_array"]

    colors = plt.cm.viridis(np.linspace(0, 1, len(Mz_array)))
    plt.figure(figsize=(10, 6))
    for i, Mz in enumerate(Mz_array):
        plt.plot(np.arange(1, steps+1e-3), S_array[i, :], marker=".", linewidth=2, label=r"$M_z=%.2f$" % (Mz), color=colors[i])
    
    plt.title(rf"{state_name}, L={L}")
    plt.legend(loc='lower right')
    plt.xlabel(rf"$steps/{dt}$")
    plt.xlim(1, steps)
    plt.ylabel(r"$S/L$")
    plt.tight_layout()
    plt.savefig("fig/" + file_name + ".png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    np.random.seed(123)
    state_name = "neel_state"
    L = fibonacci(12)  #
    t0 = 1
    tso = 0.3
    # Mz_array = np.concatenate((np.arange(0, 0.5, 0.25), np.arange(0.5, 1.5, 0.1), np.arange(1.5, 2.0+1e-3, 0.25)))
    Mz_array = np.arange(0, 4 + 0.001, 0.25)
    beta = fibonacci(11) / fibonacci(12)  #
    phi = 0
    steps = 250
    dt = 100
    
    import os
    if not os.path.exists('data'):
        os.mkdir('data')  # 创建文件夹
    if not os.path.exists('fig'):
        os.mkdir('fig')
    del os

    start_time = time.time()
    cal_S_of_t(L, tso, Mz_array, beta, steps, dt, t0, phi)
    vis_S_of_t(L, Mz_array, steps, dt)
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time:.1f} s")