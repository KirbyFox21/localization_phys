import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time


class FreeFermionSystem:
    def __init__(self, L, state_name:str='bipartite_state', spin:str='spinless', filled=None):
        """
        生成特定填充的实空间初态
        
        依据
        `Cor_{ij} = < c^\\dagger_i c_j > = \\sum_{\\alpha} \\phi^*_\\alpha (i) \\phi_\\alpha (j) `
        但是其实后面算Cor的时候，复共轭在后者而不是前者  # 这是个厄米矩阵，取复共轭不影响本征值，只会改变本征矢量
        """
        if not filled is None:
            init_state = np.zeros((L, len(filled)), dtype=np.complex128)
            for i, idx in enumerate(filled):
                init_state[idx, i] = 1
            self.psi = init_state
            return

        if spin == 'spinless':
            length = L
            dis = 1
            start = 0
        elif spin == 'up':
            length = 2 * L
            dis = 2
            start = 1
        elif spin == 'down':
            length = 2 * L
            dis = 2
            start = 0
        init_state = np.zeros((length, length // 2 // dis), dtype=np.complex128)
        if state_name == 'bipartite_state':
            for i in range(length // 2 // dis):
                init_state[start + i * dis, i] = 1
        elif state_name == 'neel_state':
            for i in range(length // 2 // dis):
                init_state[start + i * 2 * dis, i] = 1
        else:
            print("Error! ")
            return
        self.psi = init_state
        return

    def evol_sys(self, H_evo, t):
        self.psi = expm( - 1j * H_evo * t) @ self.psi
        
    def entropy(self, subsys):
        """
        subsys: np.array, list
        """
        Cor = self.psi @ self.psi.conj().T
        Cor_subsys = Cor[subsys, :][:, subsys]
        xi, _ = np.linalg.eig(Cor_subsys)
        ee = np.nansum( - xi * np.log(xi) - (1 - xi) * np.log(1 - xi))
        return ee
    
    def entropy_density(self, subsys):
        """
        subsys: np.array, list
        """
        Cor = self.psi @ self.psi.conj().T
        Cor_subsys = Cor[subsys, :][:, subsys]
        xi, _ = np.linalg.eig(Cor_subsys)
        ee = np.nansum( - xi * np.log(xi) - (1 - xi) * np.log(1 - xi)) / len(subsys)
        return ee


def gen_H_entangle(L):
    H = np.zeros((L + 1, L + 1), dtype=np.complex128)

    H[L//2+0, L] = 1
    H[L, L//2+0] = 1  #####################
    return H

def gen_H_Raman_real(L, tso, Mz, beta, t0=1, phi=0):
    Ham = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.complex128)

    for i in range(L - 1):
        Ham[2 * i + 2, 2 * i] = t0  # 0, 2, 4, ... down
        Ham[2 * i + 3, 2 * i + 1] = - t0  # 1, 3, 5, ... up
        Ham[2 * i + 1, 2 * i + 2] = tso
        Ham[2 * i + 3, 2 * i] = - tso

    Ham[0, 2 * (L - 1)] = t0  # PBC条件
    Ham[1, 2 * (L - 1) + 1] = - t0
    Ham[2 * (L - 1) + 1, 0] = tso
    Ham[1, 2 * (L - 1)] = - tso

    Ham += Ham.conj().T  # 加上H.c.

    for i in range(L):  # 准周期势
        Ham[2 * i, 2 * i] = - Mz * np.cos(2 * np.pi * beta * (i + 1) + phi)  # 下标从0开始，但是格点从1开始，所以i + 1
        Ham[2 * i + 1, 2 * i + 1] = Mz * np.cos(2 * np.pi * beta * (i + 1) + phi)

    return Ham


def cal_SIC_of_x(state_name, L, t0, tso, lbd_array, beta, phi, pre, steps, dt, f):
    H_ent = gen_H_entangle(2 * L)
    filled_indices = np.arange(0 , 2*L//f)
    # filled_indices = np.arange(L//f, L)
    if np.in1d(2*L//2+0, filled_indices) == False:  ########################  # 这也是用来建立纠缠的。如果第 L//2 个 site 无占据，则参考比特有占据，反之则无
        filled_indices = np.append(filled_indices, [2*L])

    if state_name != "random_state":
               
        SIC_array = np.zeros((len(lbd_array), steps, L//2))
        for i, lbd in enumerate(lbd_array):
            print(f"lbd = {lbd:.2f} ({i + 1} / {len(lbd_array)})")
            
            H_evo = gen_H_Raman_real(L, tso, lbd, beta, t0, phi)
            system = FreeFermionSystem(2*L + 1, filled=filled_indices)
            system.evol_sys(H_ent, np.pi/4)
            system.evol_sys(H_evo, pre)

            random_array = np.random.rand(steps)
            for j in range(steps):
                for x in range(L//2):
                    E_list = np.arange(2*L//2-x+0, 2*L//2+x+1+0, 1)  #####################
                    S_E = system.entropy(E_list)
                    S_R = system.entropy([2*L])
                    S_ER = system.entropy(np.append(E_list, 2*L))
                    SIC_array[i, j, x] = S_E + S_R - S_ER
                system.evol_sys(H_evo, dt * random_array[j])

    file_name = f"4 - SIC_of_x_{state_name}_f_{f}_L_{L}_lbd_{lbd_array[0]}_{lbd_array[-1]}_pre_{pre}_steps_{steps}_dt_{dt}"
    np.savez("data//" + file_name + ".npz", SIC_array=SIC_array)

def vis_SIC_of_x(state_name, L, lbd_array, pre, steps, dt, f):
    file_name = f"4 - SIC_of_x_{state_name}_f_{f}_L_{L}_lbd_{lbd_array[0]}_{lbd_array[-1]}_pre_{pre}_steps_{steps}_dt_{dt}"
    data = np.load("data//" + file_name + ".npz")
    SIC_array=data['SIC_array']

    plt.figure(figsize=(10, 6))

    for lbd_idx, lbd in enumerate(lbd_array):
        if state_name == "random_state":
            SIC_LTA_array = np.mean(SIC_array, axis=1)
            SIC_RSA_array = np.mean(SIC_LTA_array, axis=2)
            SIC_error_array = [np.sqrt(np.sum((SIC_LTA_array[lbd_idx, i, :] - SIC_RSA_array[lbd_idx, i]) ** 2) / (np.size(SIC_LTA_array, 2) - 1)) for i in range(np.size(SIC_LTA_array, 1))]
            plt.errorbar(range(L//2), SIC_RSA_array[lbd_idx, :] / np.log(2), SIC_error_array / np.log(2), linewidth=2, marker='.', label=rf"$\lambda={lbd:.2f}$")
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(lbd_array)))
            plt.plot(range(L//2), np.mean(SIC_array[lbd_idx, :, :] / np.log(2), 0), marker='.', linewidth=2, label=rf"$\lambda={lbd:.2f}$", color=colors[lbd_idx])

    plt.title(state_name)
    plt.legend()
    plt.xlabel(r'$|A|=x$')
    plt.ylabel(r'$SIC$')
    plt.tight_layout()
    plt.savefig("fig//" + file_name + ".png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    np.random.seed(123)
    state_name = "bipartite_state"
    f = 2
    L = 280
    t0 = 1
    tso = 0.3
    # lbd_array = np.concatenate((np.arange(0, 0.5, 0.25), np.arange(0.5, 1.5, 0.1), np.arange(1.5, 2.0+1e-3, 0.25)))
    # lbd_array = np.array([2.0])
    lbd_array = np.arange(0.1, 3.0 + 0.001, 0.1)
    beta = (np.sqrt(5) - 1) / 2
    phi = 0
    pre = 10000
    steps = 10
    dt = 10

    start_time = time.time()
    cal_SIC_of_x(state_name, L, t0, tso, lbd_array, beta, phi, pre, steps, dt, f)
    vis_SIC_of_x(state_name, L, lbd_array, pre, steps, dt, f)
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time:.1f} s")








# import sys
# # 获取Python版本号
# version = sys.version
# print("Python version:", version)
# model_name = "Raman"
# import tensorcircuit as tc
# import numpy as np
# import matplotlib.pyplot as plt
# import time

# K = tc.set_backend("numpy")
# tc.set_dtype("complex128")

# def fibonacci(n):
#     if not(type(n) == int) or n < 0:
#         return
#     if n == 0 or n == 1:
#         return 1
#     else:
#         f = fibonacci(n-1) + fibonacci(n-2)
#         return f
# # F14 = 610

# def gen_H_entangle_nambu(L, site=None):  # 这个哈密顿量用于建立参考比特与第 L//2 个 site 的纠缠
#     if site is None:
#         site = 2 * L // 2  # site 的默认值与 L 有关，但是不允许直接这样定义，所以在函数体内设置
#     H = np.zeros((2 * 2 * L + 2, 2 * 2 * L + 2), dtype=np.complex128)  # 第 L+1 个 site 是参考比特，python 里的编号是 L

#     H[site, 2 * L] = 1 / 2
#     H[2 * L, site] = 1 / 2
#     H[site + 2 * L + 1, 2 * L + 2 * L + 1] = - 1 / 2  # 显然这里没考虑自旋，觉得还是需要考虑一下的
#     H[2 * L + 2 * L + 1, site + 2 * L + 1] = - 1 / 2

#     return H


# def gen_H_Raman_real_nambu(L, tso, Mz, beta, t0=1, phi=0):
#     Ham = np.zeros((2 * 2 * L + 2, 2 * 2 * L + 2), dtype=np.complex128)

#     for i in range(L - 1):
#         Ham[2 * i + 2, 2 * i] = t0  # 0, 2, 4, ... down
#         Ham[2 * i + 3, 2 * i + 1] = - t0  # 1, 3, 5, ... up
#         Ham[2 * i + 1, 2 * i + 2] = tso
#         Ham[2 * i + 3, 2 * i] = - tso

#         Ham[2 * i + 2 + 2 * L + 1, 2 * i + 2 * L + 1] = - t0
#         Ham[2 * i + 3 + 2 * L + 1, 2 * i + 1 + 2 * L + 1] = t0
#         Ham[2 * i + 1 + 2 * L + 1, 2 * i + 2 + 2 * L + 1] = - tso
#         Ham[2 * i + 3 + 2 * L + 1, 2 * i + 2 * L + 1] = tso

#     Ham[0, 2 * (L - 1)] = t0  # PBC条件
#     Ham[1, 2 * (L - 1) + 1] = - t0
#     Ham[2 * (L - 1) + 1, 0] = tso
#     Ham[1, 2 * (L - 1)] = - tso

#     Ham[0 + 2 * L + 1, 2 * (L - 1) + 2 * L + 1] = - t0
#     Ham[1 + 2 * L + 1, 2 * (L - 1) + 1 + 2 * L + 1] = t0
#     Ham[2 * (L - 1) + 1 + 2 * L + 1, 0 + 2 * L + 1] = - tso
#     Ham[1 + 2 * L + 1, 2 * (L - 1) + 2 * L + 1] = tso

#     Ham += Ham.conj().T  # 加上H.c.

#     for i in range(L):  # 准周期势
#         Ham[2 * i, 2 * i] = - Mz * np.cos(2 * np.pi * beta * (i + 1) + phi)  # 下标从0开始，但是格点从1开始，所以i + 1
#         Ham[2 * i + 1, 2 * i + 1] = Mz * np.cos(2 * np.pi * beta * (i + 1) + phi)
        
#         Ham[2 * i + 2 * L + 1, 2 * i + 2 * L + 1] = - Ham[2 * i, 2 * i]
#         Ham[2 * i + 1 + 2 * L + 1, 2 * i + 1 + 2 * L + 1] = - Ham[2 * i + 1, 2 * i + 1]

#     return Ham


# def cal_SIC_of_x(L, tso, Mz_array, beta, pre, steps, dt, t0=1, phi=0, site=None):
#     if site is None:
#         site = 2 * L // 2  # site 的默认值与 L 有关，但是不允许直接这样定义，所以在函数体内设置
#     H_ent = gen_H_entangle_nambu(L, site)
#     filled_indices = np.arange(0 , 2 * L // 2)
#     # filled_indices = np.arange(L//f, L)
#     if np.in1d(site, filled_indices) == False:  # 这也是用来建立纠缠的。如果第 L//2 个 site 无占据，则参考比特有占据，反之则无
#         filled_indices = np.append(filled_indices, [2 * L])

#     SIC_array = np.zeros((len(Mz_array), steps, L // 2))
#     for i, Mz in enumerate(Mz_array):
#         print(f"Mz = {Mz:.2f} ({i + 1} / {len(Mz_array)})")
            
#         H_evo = gen_H_Raman_real_nambu(L, tso, Mz, beta, t0, phi)
#         system = tc.FGSSimulator(2 * L + 1, filled=filled_indices)
#         system.evol_ghamiltonian(H_ent * np.pi / 4)
#         system.evol_ghamiltonian(H_evo * pre)

#         random_array = np.random.rand(steps)
#         for j in range(steps):
#             for x in range(L // 2):
#                 E_list = np.arange(site - x, site + x + 0.001, 1)  # +0.001 使得列表取值能取到后一个数，且数据类型为浮点数，虽然它本身是整数
#                 S_E = system.entropy(E_list)
#                 S_R = system.entropy([L])
#                 S_ER = system.entropy(np.append(E_list, L))
#                 SIC_array[i, j, x] = S_E + S_R - S_ER
#             system.evol_ghamiltonian(H_evo * dt * random_array[j])

#     file_name = f"SIC_of_x_{model_name}_L_{L}_tso_{tso:.1f}_Mz_{Mz_array[0]}_{Mz_array[-1]}_pre_{pre}_steps_{steps}_dt_{dt}"
#     np.savez("data/" + file_name + ".npz", SIC_array=SIC_array)

# def vis_SIC_of_x(L, tso, Mz_array, pre, steps, dt):
#     file_name = f"SIC_of_x_{model_name}_L_{L}_tso_{tso:.1f}_Mz_{Mz_array[0]}_{Mz_array[-1]}_pre_{pre}_steps_{steps}_dt_{dt}"
#     data = np.load("data/" + file_name + ".npz")
#     SIC_array=data['SIC_array']

#     plt.figure(figsize=(10, 6))

#     for Mz_idx, Mz in enumerate(Mz_array):
#         if state_name == "bipartite_state":
#             plt.plot(range(L // 2), np.mean(SIC_array[Mz_idx, :, :] / np.log(2), 0), marker='.', linewidth=2, label=rf"$\lambda={Mz:.2f}$")
#             # mean(a, axis=())  # 表示对给定轴求平均值
#             # 多维数组，给定其中一个指标，其他全是 : ，则新数组的尺寸为原数组尺寸删掉给定的那个轴
#             # 例：a 的尺寸是 (2, 3, 4)，b = a [:, 1, :]，则 b 的尺寸是 (2, 4)

#     plt.title(model_name)
#     plt.legend()
#     plt.xlabel(r'$|A|=x$')
#     plt.ylabel(r'$SIC$')
#     plt.tight_layout()
#     plt.savefig("fig/" + file_name + ".png", dpi=300, bbox_inches="tight")
#     plt.show()

# if __name__ == "__main__":
#     np.random.seed(123)
#     state_name = "bipartite_state"
#     L = fibonacci(12)  # F14
#     tso = 0.3
#     # Mz_array = np.concatenate((np.arange(0, 0.5, 0.25), np.arange(0.5, 1.5, 0.1), np.arange(1.5, 2.0+1e-3, 0.25)))
#     Mz_array = np.arange(0, 4, 0.25)
#     beta = fibonacci(11) / fibonacci(12)  # F14
#     phi = np.pi/4  # 注意这个相位也是准周期势能的，有什么用
#     pre = 1000  # 10000
#     steps = 5  # 10
#     dt = 10
    
#     import os
#     if not os.path.exists('data'):
#         os.mkdir('data')  # 创建文件夹
#     if not os.path.exists('fig'):
#         os.mkdir('fig')
#     del os

#     start_time = time.time()
#     cal_SIC_of_x(L, tso, Mz_array, beta, pre, steps, dt, t0=1, phi=phi, site=None)
#     vis_SIC_of_x(L, tso, Mz_array, pre, steps, dt)
#     end_time = time.time()
#     print(f"elapsed time: {end_time - start_time:.1f} s")