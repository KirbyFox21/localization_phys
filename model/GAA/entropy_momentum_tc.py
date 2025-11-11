import tensorcircuit as tc
import numpy as np
import matplotlib.pyplot as plt
import time

model_name = "GAA"
K = tc.set_backend("numpy")
tc.set_dtype("complex128")


def gen_H_GAA_momentum_nambu(L, t0, lbd, a, beta, phi=0):
    Ham = np.zeros((2 * L, 2 * L), dtype=np.complex128)
    V = np.zeros(L, dtype=np.complex128)
    for i in range(L):
        V[i] = 2 * lbd * np.cos(2 * np.pi * beta * (i + 1) + phi) / (1 - a * np.cos(2 * np.pi * beta * (i + 1) + phi))
        Ham[i, i] = - t0 * 2 * np.cos(i * 2 * np.pi / L)
    for i in range(L):
        for j in range(L):
            Ham[i, j] += np.sum(V / L * np.exp( - 1j * (i - j) * 2 * np.pi / L * np.arange(L, dtype=np.complex128)))  # * 是按元素乘法， @ 是矩阵乘法
    Ham[L:2*L-1, L:2*L-1] = - Ham[0:L-1, 0:L-1]
    return Ham


def cal_S_of_t(L, t0, lbd_array, a, beta, EF, steps, dt, phi=0):
    S_array = np.zeros((len(lbd_array), steps), dtype=np.complex128)
    sub_system = list(range(L // 2))
    if state_name == "bipartite_state":
        filled_indices = np.arange(EF)
    elif state_name == "neel_state":
        filled_indices = np.arange(0, L, 2)

    for i, lbd in enumerate(lbd_array):
        print(f"lbd = {lbd:.2f} ({i + 1} / {len(lbd_array)})")
        H_evo = gen_H_GAA_momentum_nambu(L, t0, lbd, a, beta, phi)
        system = tc.FGSSimulator(L, filled=filled_indices)

        for j in range(steps):
            S_array[i, j] = system.entropy(sub_system)
            system.evol_ghamiltonian(H_evo * dt)

    S_array /= len(sub_system)  # 记得除以子系统长度！！
    file_name = f"S_{model_name}_{state_name}_L_{L}_EF_{EF}_lbd_{lbd_array[0]}_{lbd_array[-1]}_steps_{steps}_dt_{dt}"
    np.savez("data/" + file_name + ".npz", S_array=S_array)


def vis_S_of_t(L, lbd_array, steps, dt):
    file_name = f"S_{model_name}_{state_name}_L_{L}_EF_{EF}_lbd_{lbd_array[0]}_{lbd_array[-1]}_steps_{steps}_dt_{dt}"
    data = np.load("data/" + file_name + ".npz")
    S_array = data["S_array"]

    colors = plt.cm.viridis(np.linspace(0, 1, len(lbd_array)))
    plt.figure(figsize=(10, 6))
    for i, lbd in enumerate(lbd_array):
        plt.plot(np.arange(1, steps+1e-3), S_array[i, :], marker=".", linewidth=2, label=r"$M_z=%.2f$" % (lbd), color=colors[i])
    
    plt.title(rf"{state_name}, L={L}")
    plt.legend(loc='lower right')
    plt.xlabel(rf"$steps/{dt}$")
    plt.xlim(1, steps)
    plt.ylabel(r"$S/L$")
    plt.tight_layout()
    plt.savefig("fig/" + file_name + ".png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    np.random.seed(123)
    state_name = "bipartite_state"
    L = 1000  #
    t0 = 1
    # lbd_array = np.concatenate((np.arange(0, 0.5, 0.25), np.arange(0.5, 1.5, 0.1), np.arange(1.5, 2.0+1e-3, 0.25)))
    lbd_array = np.arange(0.1, 2.5 + 0.001, 0.1)
    a = 0.3
    beta = (np.sqrt(5) - 1) / 2  #
    phi = 0
    # EF = L // 4
    steps = 250
    dt = 25  #
    
    import os
    if not os.path.exists('data'):
        os.mkdir('data')  # 创建文件夹
    if not os.path.exists('fig'):
        os.mkdir('fig')
    del os

    for i in range(1, 8):
        EF = L // 8 * i
        start_time = time.time()
        cal_S_of_t(L, t0, lbd_array, a, beta, EF, steps, dt, phi)
        vis_S_of_t(L, lbd_array, steps, dt)
        end_time = time.time()
        print(f"i = {i}, elapsed time: {end_time - start_time:.1f} s")