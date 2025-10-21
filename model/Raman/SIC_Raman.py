import sys
# 获取Python版本号
version = sys.version
print("Python version:", version)
model_name = "Raman"
import tensorcircuit as tc
import numpy as np
import matplotlib.pyplot as plt
import time

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

def gen_H_entangle_nambu(L, site=None):  # 这个哈密顿量用于建立参考比特与第 L//2 个 site 的纠缠
    if site is None:
        site = 2 * L // 2  # site 的默认值与 L 有关，但是不允许直接这样定义，所以在函数体内设置
    H = np.zeros((2 * 2 * L + 2, 2 * 2 * L + 2), dtype=np.complex128)  # 第 L+1 个 site 是参考比特，python 里的编号是 L

    H[site, 2 * L] = 1 / 2
    H[2 * L, site] = 1 / 2
    H[site + 2 * L + 1, 2 * L + 2 * L + 1] = - 1 / 2  # 显然这里没考虑自旋，觉得还是需要考虑一下的
    H[2 * L + 2 * L + 1, site + 2 * L + 1] = - 1 / 2

    return H


def gen_H_Raman_real_nambu(L, tso, Mz, beta, t0=1, phi=1):
    Ham = np.zeros((2 * 2 * L + 2, 2 * 2 * L + 2), dtype=np.complex128)

    for i in range(L - 1):
        Ham[2 * i + 2, 2 * i] = t0  # 0, 2, 4, ... down
        Ham[2 * i + 3, 2 * i + 1] = - t0  # 1, 3, 5, ... up
        Ham[2 * i + 1, 2 * i + 2] = tso
        Ham[2 * i + 3, 2 * i] = - tso

        Ham[2 * i + 2 + 2 * L + 1, 2 * i + 2 * L + 1] = - t0
        Ham[2 * i + 3 + 2 * L + 1, 2 * i + 1 + 2 * L + 1] = t0
        Ham[2 * i + 1 + 2 * L + 1, 2 * i + 2 + 2 * L + 1] = - tso
        Ham[2 * i + 3 + 2 * L + 1, 2 * i + 2 * L + 1] = tso

    Ham[0, 2 * (L - 1)] = t0  # PBC条件
    Ham[1, 2 * (L - 1) + 1] = - t0
    Ham[2 * (L - 1) + 1, 0] = tso
    Ham[1, 2 * (L - 1)] = - tso

    Ham[0 + 2 * L + 1, 2 * (L - 1) + 2 * L + 1] = - t0
    Ham[1 + 2 * L + 1, 2 * (L - 1) + 1 + 2 * L + 1] = t0
    Ham[2 * (L - 1) + 1 + 2 * L + 1, 0 + 2 * L + 1] = - tso
    Ham[1 + 2 * L + 1, 2 * (L - 1) + 2 * L + 1] = tso

    Ham += Ham.conj().T  # 加上H.c.

    for i in range(L):  # 准周期势
        Ham[2 * i, 2 * i] = - Mz * np.cos(2 * np.pi * beta * (i + 1) + phi)  # 下标从0开始，但是格点从1开始，所以i + 1
        Ham[2 * i + 1, 2 * i + 1] = Mz * np.cos(2 * np.pi * beta * (i + 1) + phi)
        
        Ham[2 * i + 2 * L + 1, 2 * i + 2 * L + 1] = - Ham[2 * i, 2 * i]
        Ham[2 * i + 1 + 2 * L + 1, 2 * i + 1 + 2 * L + 1] = - Ham[2 * i + 1, 2 * i + 1]

    return Ham


def cal_SIC_of_x(L, tso, Mz_array, beta, pre, steps, dt, t0=1, phi=1, site=None):
    if site is None:
        site = 2 * L // 2  # site 的默认值与 L 有关，但是不允许直接这样定义，所以在函数体内设置
    H_ent = gen_H_entangle_nambu(L, site)
    filled_indices = np.arange(0 , 2 * L // 2)
    # filled_indices = np.arange(L//f, L)
    if np.in1d(site, filled_indices) == False:  # 这也是用来建立纠缠的。如果第 L//2 个 site 无占据，则参考比特有占据，反之则无
        filled_indices = np.append(filled_indices, [2 * L])

    SIC_array = np.zeros((len(Mz_array), steps, L // 2))
    for i, Mz in enumerate(Mz_array):
        print(f"Mz = {Mz:.2f} ({i + 1} / {len(Mz_array)})")
            
        H_evo = gen_H_Raman_real_nambu(L, tso, Mz, beta, t0, phi)
        system = tc.FGSSimulator(2 * L + 1, filled=filled_indices)
        system.evol_ghamiltonian(H_ent * np.pi / 4)
        system.evol_ghamiltonian(H_evo * pre)

        random_array = np.random.rand(steps)
        for j in range(steps):
            for x in range(L // 2):
                E_list = np.arange(site - x, site + x + 0.001, 1)  # +0.001 使得列表取值能取到后一个数，且数据类型为浮点数，虽然它本身是整数
                S_E = system.entropy(E_list)
                S_R = system.entropy([L])
                S_ER = system.entropy(np.append(E_list, L))
                SIC_array[i, j, x] = S_E + S_R - S_ER
            system.evol_ghamiltonian(H_evo * dt * random_array[j])

    file_name = f"SIC_of_x_{model_name}_L_{L}_tso_{tso:.1f}_Mz_{Mz_array[0]}_{Mz_array[-1]}_pre_{pre}_steps_{steps}_dt_{dt}"
    np.savez("data/" + file_name + ".npz", SIC_array=SIC_array)

def vis_SIC_of_x(L, tso, Mz_array, pre, steps, dt):
    file_name = f"SIC_of_x_{model_name}_L_{L}_tso_{tso:.1f}_Mz_{Mz_array[0]}_{Mz_array[-1]}_pre_{pre}_steps_{steps}_dt_{dt}"
    data = np.load("data/" + file_name + ".npz")
    SIC_array=data['SIC_array']

    plt.figure(figsize=(10, 6))

    for Mz_idx, Mz in enumerate(Mz_array):
        if state_name == "bipartite_state":
            plt.plot(range(L // 2), np.mean(SIC_array[Mz_idx, :, :] / np.log(2), 0), marker='.', linewidth=2, label=rf"$\lambda={Mz:.2f}$")
            # mean(a, axis=())  # 表示对给定轴求平均值
            # 多维数组，给定其中一个指标，其他全是 : ，则新数组的尺寸为原数组尺寸删掉给定的那个轴
            # 例：a 的尺寸是 (2, 3, 4)，b = a [:, 1, :]，则 b 的尺寸是 (2, 4)

    plt.title(model_name)
    plt.legend()
    plt.xlabel(r'$|A|=x$')
    plt.ylabel(r'$SIC$')
    plt.tight_layout()
    plt.savefig("fig/" + file_name + ".png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    np.random.seed(123)
    state_name = "bipartite_state"
    L = fibonacci(12)  # F14
    tso = 0.3
    # Mz_array = np.concatenate((np.arange(0, 0.5, 0.25), np.arange(0.5, 1.5, 0.1), np.arange(1.5, 2.0+1e-3, 0.25)))
    Mz_array = np.arange(0, 4, 0.25)
    beta = fibonacci(11) / fibonacci(12)  # F14
    phi = np.pi/4
    pre = 1000  # 10000
    steps = 5  # 10
    dt = 10
    
    import os
    if not os.path.exists('data'):
        os.mkdir('data')  # 创建文件夹
    if not os.path.exists('fig'):
        os.mkdir('fig')
    del os

    start_time = time.time()
    cal_SIC_of_x(L, tso, Mz_array, beta, pre, steps, dt, t0=1, phi=1, site=None)
    vis_SIC_of_x(L, tso, Mz_array, pre, steps, dt)
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time:.1f} s")