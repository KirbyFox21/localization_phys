
import sys
# 获取Python版本号
version = sys.version
print("Python version:", version)

import tensorcircuit as tc
import numpy as np
import matplotlib.pyplot as plt
import time

K = tc.set_backend("numpy")
tc.set_dtype("complex128")

def gen_H_entangle_nambu(L):  # 这个哈密顿量用于建立参考比特与第 L//2 个 site 的纠缠
    H = np.zeros((2*L+2, 2*L+2), dtype=np.complex128)  # 第 L+1 个 site 是参考比特，python 里的编号是 L

    H[L//2, L] = 1 / 2
    H[L, L//2] = 1 / 2
    H[L//2+L+1, L+L+1] = -1 / 2
    H[L+L+1, L//2+L+1] = -1 / 2

    return H

def gen_H_GAA_nambu(L, t, lbd, a, b, phi):  # 用 Nambu 哈密顿量是因为 tensorcircuit 的 fgs 使用了 Bogoliubov 变换。代入 c 和 Bogoliubov 变换的 alpha 的关系就能得到哈密顿量
    H = np.zeros((2*L+2, 2*L+2), dtype=np.complex128)

    for i in range(L-1):
        H[i, i+1] = t / 2
        H[i+1, i] = t / 2
        H[i+L+1, i+1+L+1] = -t / 2
        H[i+1+L+1, i+L+1] = -t / 2

    for i in range(L):
        H[i, i] = 2 * lbd * np.cos(2*np.pi*i*b+phi) / (1-a*np.cos(2*np.pi*i*b+phi)) / 2
        H[i+L+1, i+L+1] = -H[i, i]

    return H

def cal_SIC_of_x(state_name, L, t, lbd_array, a, b, phi, pre, steps, dt, f):
    H_ent = gen_H_entangle_nambu(L)
    filled_indices = np.arange(0 , L//f)
    # filled_indices = np.arange(L//f, L)
    if np.in1d(L//2, filled_indices) == False:  # 这也是用来建立纠缠的。如果第 L//2 个 site 无占据，则参考比特有占据，反之则无
        filled_indices = np.append(filled_indices, [L])

    if state_name == "bipartite_state":
        SIC_array = np.zeros((len(lbd_array), steps, L//2))
        for i, lbd in enumerate(lbd_array):
            print(f"ldb = {lbd:.2f} ({i + 1} / {len(lbd_array)})")
            
            H_evo = gen_H_GAA_nambu(L, t, lbd, a, b, phi)
            system = tc.FGSSimulator(L+1, filled=filled_indices)
            system.evol_ghamiltonian(2 * H_ent * np.pi/4)
            system.evol_ghamiltonian(2 * H_evo * pre)

            random_array = np.random.rand(steps)
            for j in range(steps):
                for x in range(L//2):
                    E_list = np.arange(L//2-x, L//2+x+0.001, 1)  # +0.001 使得列表取值能取到后一个数，且数据类型为浮点数，虽然它本身是整数
                    S_E = system.entropy(E_list)
                    S_R = system.entropy([L])
                    S_ER = system.entropy(np.append(E_list, L))
                    SIC_array[i, j, x] = S_E + S_R - S_ER
                system.evol_ghamiltonian(2 * H_evo * dt * random_array[j])

    file_name = f"4 - SIC_of_x_{state_name}_f_{f}_L_{L}_a_{a:.1f}_lbd_{lbd_array[0]}_{lbd_array[-1]}_pre_{pre}_steps_{steps}_dt_{dt}"
    np.savez("data//" + file_name + ".npz", SIC_array=SIC_array)

def vis_SIC_of_x(state_name, L, lbd_array, a, pre, steps, dt, f):
    file_name = f"4 - SIC_of_x_{state_name}_f_{f}_L_{L}_a_{a:.1f}_lbd_{lbd_array[0]}_{lbd_array[-1]}_pre_{pre}_steps_{steps}_dt_{dt}"
    data = np.load("data//" + file_name + ".npz")
    SIC_array=data['SIC_array']

    plt.figure(figsize=(10, 6))

    for lbd_idx, lbd in enumerate(lbd_array):
        if state_name == "bipartite_state":
            plt.plot(range(L//2), np.mean(SIC_array[lbd_idx, :, :] / np.log(2), 0), marker='.', linewidth=2, label=rf"$\lambda={lbd:.2f}$")
            # mean(a, axis=())  # 表示对给定轴求平均值
            # 多维数组，给定其中一个指标，其他全是 : ，则新数组的尺寸为原数组尺寸删掉给定的那个轴
            # 例：a 的尺寸是 (2, 3, 4)，b = a [:, 1, :]，则 b 的尺寸是 (2, 4)

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
    L = 300
    t = 1
    lbd_array = np.concatenate((np.arange(0, 0.5, 0.25), np.arange(0.5, 1.5, 0.1), np.arange(1.5, 2.0+1e-3, 0.25)))
    a = 0
    b = 2 / (np.sqrt(5) - 1)
    phi = np.pi/4
    pre = 10000
    steps = 10
    dt = 10
    
    import os
    if not os.path.exists('data'):
        os.mkdir('data')  # 创建文件夹
    if not os.path.exists('fig'):
        os.mkdir('fig')
    del os

    start_time = time.time()
    cal_SIC_of_x(state_name, L, t, lbd_array, a, b, phi, pre, steps, dt, f)
    vis_SIC_of_x(state_name, L, lbd_array, a, pre, steps, dt, f)
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time:.1f} s")
