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


def vis_S_sat_and_vs(L, Mz_array, steps, dt):
    file_name = f"S_{model_name}_{state_name}_L_{L}_Mz_{Mz_array[0]}_{Mz_array[-1]}_steps_{steps}_dt_{dt}"
    data = np.load("data/" + file_name + ".npz")
    S_array = data["S_array"]

    S_sat_array = np.zeros(len(Mz_array), dtype=np.complex128)
    vs_array = np.zeros(len(Mz_array), dtype=np.complex128)  # 记得指定数据类型

    for i, _ in enumerate(Mz_array):
        S_sat_array[i] = np.mean(S_array[i, 149:250])  # 数组切片也是包前不包后  #
        x = np.arange(10)
        coefficients = np.polyfit(x, S_array[i, 0:10], deg=1)  #
        vs_array[i] = coefficients[0]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(Mz_array, S_sat_array, linewidth=2, marker='.')
    ax1.set_xlabel(r'$M_z$')
    ax1.set_ylabel(r'$S_{sat}$', color='tab:blue')
    # ax1.set_ylim([0, 69])
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title(rf"{state_name}, L={L}")

    ax2 = ax1.twinx()
    ax2.plot(Mz_array, vs_array, marker='x', markersize=5, color='tab:green')
    ax2.set_ylabel(r'$v_s$', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    plt.tight_layout()
    plt.savefig('fig/' + f'S_sat_and_vs_{model_name}_{state_name}_L_{L}_Mz_{Mz_array[0]}_{Mz_array[-1]}_steps_{steps}_dt_{dt}.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    np.random.seed(123)
    state_name = "bipartite_state"
    L = fibonacci(12)  #
    t0 = 1
    tso = 0.3
    # Mz_array = np.concatenate((np.arange(0, 0.5, 0.25), np.arange(0.5, 1.5, 0.1), np.arange(1.5, 2.0+1e-3, 0.25)))
    Mz_array = np.arange(0, 4 + 0.001, 0.1)
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
    vis_S_sat_and_vs(L, Mz_array, steps, dt)
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time:.1f} s")