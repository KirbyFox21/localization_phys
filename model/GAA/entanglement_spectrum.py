import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import time


def cal_entanglement_spectrum(init_state, H_evo, dt):
    psi_t = expm( - 1j * H_evo * dt) @ init_state  # 别忘了 numpy 的矩阵乘法是 @
    Cor = psi_t @ psi_t.conj().T
    Cor_subsys = Cor[0:np.size(Cor, 0) // 2, 0:np.size(Cor, 1) // 2]  # 注意 python 要求索引的数据类型必须完全是 int ，而且注意索引从0开始！！
    xi, _ = np.linalg.eig(Cor_subsys)
    return xi

def cal_entanglement_entropy(xi):
    ee = np.sum( - xi * np.log(xi) - (1 - xi) * np.log(1 - xi)) / len(xi)  # 这里除以了子系统长度  # 一定要用 np.sum()，不要直接 sum
    # 另外 np.sum() 对 nan 值怎么处理，比如 log(0)
    return ee


def gen_H_GAA_momentum(L, t0, lbd, a, beta, phi=0):
    Ham = np.zeros((L, L), dtype=np.complex128)
    V = np.zeros(L, dtype=np.complex128)
    for i in range(L):
        V[i] = 2 * lbd * np.cos(2 * np.pi * beta * (i + 1) + phi) / (1 - a * np.cos(2 * np.pi * beta * (i + 1) + phi))
        Ham[i, i] = - t0 * 2 * np.cos(i * 2 * np.pi / L)
    for i in range(L):
        for j in range(L):
            Ham[i, j] += np.sum(V / L * np.exp( - 1j * (i - j) * 2 * np.pi / L * np.arange(L, dtype=np.complex128)))  # * 是按元素乘法， @ 是矩阵乘法
    return Ham


if __name__ == "__main__":
    L = 320  #
    t0 = 1
    lbd_array = np.arange(0.5, 1.5 + 0.001, 0.05)
    beta = (np.sqrt(5) - 1) / 2  #
    a = 0.3
    phi = 0
    # EF = L // 4
    steps = 250
    dt = 25  # 可以适当调大

    for k in range(1, 8):
        EF = L // 8 * k
        start_time = time.time()
        H_0 = gen_H_GAA_momentum(L, t0, 0, a, beta, phi)
        e, Vr = np.linalg.eig(H_0)
        init_state = Vr[:, 0:EF]

        S_array = np.zeros((len(lbd_array), steps), dtype=np.complex128)
        xi_array = np.zeros((len(lbd_array), steps, L // 2), dtype=np.complex128)
        for i, lbd in enumerate(lbd_array):
            H_evo = gen_H_GAA_momentum(L, t0, lbd, a, beta, phi)

            for j in range(steps):
                xi = cal_entanglement_spectrum(init_state, H_evo, dt * j)
                S_array[i, j] = cal_entanglement_entropy(xi)
                xi_array [i, j, :] = xi
            print(f"lbd = {lbd} ({i + 1} / {len(lbd_array)})")
        np.savez(f"EF_{EF}", S_array=S_array, xi_array=xi_array)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(lbd_array)))
        plt.figure(figsize=(10, 6))
        for i, lbd in enumerate(lbd_array):
            plt.plot(np.arange(1, steps+1e-3), S_array[i, :], marker=".", linewidth=2, label=r"$M_z=%.2f$" % (lbd), color=colors[i])
        plt.savefig(f"EF_{EF}")

        end_time = time.time()
        print(f"k = {k}, elapsed time: {end_time - start_time:.1f} s")


    for k in range(1, 8):
        EF = L // 8 * k
        file_name = f'EF_{EF}'
        data = np.load(file_name + '.npz')
        S_array = data['S_array']

        S_sat_array = np.zeros(len(lbd_array), dtype=np.complex128)

        for i, _ in enumerate(lbd_array):
            S_sat_array[i] = np.mean(S_array[i, 149:250])  # 数组切片也是包前不包后  #

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(lbd_array, S_sat_array, linewidth=2, marker='.')
        ax1.set_xlabel(r'$\lambda$')
        ax1.set_ylabel(r'$S_{sat}$', color='tab:blue')
        ax1.set_ylim([0, 0.45])
        # ax1.set_ylim([0, 69])
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        plt.savefig(file_name + '_S_sat.png', dpi=300, bbox_inches='tight')


    # for k in range(1, 8):
    #     EF = L // 8 * k
    #     file_name = f'EF_{EF}'
    #     data = np.load(file_name + '.npz')
    #     xi_array = data['xi_array']
        
    #     gap_avg_array = np.zeros(len(lbd_array), dtype=np.complex128)
    #     for i, _ in enumerate(lbd_array):
    #         gap = np.zeros(steps, dtype=np.complex128)
    #         for j in range(steps):
    #             idx_up = xi_array[i, j, :] >= 0.5
    #             min_up = np.min(xi_array[i, j, idx_up])
    #             idx_dn = xi_array[i, j, :] <= 0.5
    #             max_dn = np.max(xi_array[i, j, idx_dn])
    #             gap[j] = min_up - max_dn
    #         gap_avg_array[i] = np.mean(gap[149:250])

    #     fig, ax1 = plt.subplots(figsize=(10, 6))
    #     ax1.plot(lbd_array, gap_avg_array, linewidth=2, marker='.')
    #     ax1.set_xlabel(r'$\lambda$')
    #     ax1.set_ylabel(r'$\xi$', color='tab:green')
    #     ax1.set_ylim(bottom=0)  #
    #     # ax1.set_ylim([0, 69])
    #     ax1.tick_params(axis='y', labelcolor='tab:green')
        
    #     plt.savefig(file_name + '_ES_gap_avg.png', dpi=300, bbox_inches='tight')
    #     print(f"k = {k}")       

