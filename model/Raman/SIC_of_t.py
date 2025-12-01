import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time

model_name = 'Raman'

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


def gen_H_entangle(L, bias):
    H = np.zeros((L + 1, L + 1), dtype=np.complex128)

    H[L//2+bias, L] = 1
    H[L, L//2+bias] = 1  #####################
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



def cal_SIC_of_t(L, t0, tso, lbd, beta, phi, x0, bias, steps, dt):
    H_ent = gen_H_entangle(2 * L, bias)
    filled_indices = np.arange(0 , 2*L//2)
    # filled_indices = np.arange(L//f, L)
    if np.in1d(2*L//2+bias, filled_indices) == False:  ########################  # 这也是用来建立纠缠的。如果第 L//2 个 site 无占据，则参考比特有占据，反之则无
        filled_indices = np.append(filled_indices, [2*L])
        
    SIC_array = np.zeros(steps)

    H_evo = gen_H_Raman_real(L, tso, lbd, beta, t0, phi)
    system = FreeFermionSystem(2*L + 1, filled=filled_indices)
    system.evol_sys(H_ent, np.pi/4)
    
    E_list = np.arange(2*L//2-x0+bias, 2*L//2+x0+1+bias, 1)  #####################
    for j in range(steps):
        S_E = system.entropy(E_list)
        S_R = system.entropy([2*L])
        S_ER = system.entropy(np.append(E_list, 2*L))
        SIC_array[j] = S_E + S_R - S_ER
        system.evol_sys(H_evo, dt)
    SIC_array /= np.log(2)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(steps) * dt, SIC_array)
    plt.xlabel('t')
    plt.ylabel('SIC')
    plt.ylim([-0.1, 2.1])
    plt.title(rf'{model_name}, |A|={x0}')
    plt.show()
    # plt.savefig(f'fig/bias_{bias}.png', dpi=300)
    # plt.close()

    
if __name__ == '__main__':
    L = 610
    t0 = 1
    tso = 0.3
    # lbd_array = np.concatenate((np.arange(0, 0.5, 0.25), np.arange(0.5, 1.5, 0.1), np.arange(1.5, 2.0+1e-3, 0.25)))
    # lbd_array = np.array([2.0])
    lbd = 3.5
    beta = (np.sqrt(5) - 1) / 2
    phi = 0
    steps = 200
    dt = 10
    # x0 = 5
    # bias = 24
    x0 = 5
    # for bias in range(0, 202, 2):
    #     cal_SIC_of_t(L, t0, tso, lbd, beta, phi, x0, bias, steps, dt)
    #     print(f'bias={bias}')
    bias = 23
    cal_SIC_of_t(L, t0, tso, lbd, beta, phi, x0, bias, steps, dt)