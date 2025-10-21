
import tensorcircuit as tc
import numpy as np
import matplotlib.pyplot as plt
import time

tc.set_backend("numpy")
tc.set_dtype("complex128")




def gen_H_GAA(L, t, lbd, a, b, phi):
    H = np.zeros((2 * L, 2 * L), dtype=np.complex128)
    for i in range(L - 1):
        H[i, i + 1] = t
        H[i + 1, i] = t
    for i in range(L):
        H[i, i] = 2 * lbd * np.cos(2*np.pi*i*b+phi) / (1-a*np.cos(2*np.pi*i*b+phi))
    return H


def cal_S_and_N_of_t(state_name, L, t, lbd_array, a, b, phi, steps, dt, f):
    S_array = np.zeros((len(lbd_array), steps), dtype=np.complex128)
    sub_system = list(range(L // 2))
    filled_indices = np.arange(0 , L//f)

    if state_name == "bipartite_state":

        for i, lbd in enumerate(lbd_array):
            print(f"ldb = {lbd:.2f} ({i + 1} / {len(lbd_array)})")
            H_GAA = gen_H_GAA(L, t, lbd, a, b, phi)
            system = tc.FGSSimulator(L, filled=filled_indices)

            for j in range(steps):
                S_array[i, j] = system.entropy(sub_system)
                system.evol_ghamiltonian(H_GAA * dt)

    file_name = f"2 - S_{state_name}_f_{f}_L_{L}_a_{a:.1f}_lbd_{lbd_array[0]}_{lbd_array[-1]}_steps_{steps}_dt_{dt}"
    np.savez("data//" + file_name + ".npz", S_array=S_array)



def vis_S_of_t(L, lbd_array, steps, dt, a, f):
    file_name = f"2 - S_{state_name}_f_{f}_L_{L}_a_{a:.1f}_lbd_{lbd_array[0]}_{lbd_array[-1]}_steps_{steps}_dt_{dt}"
    data = np.load("data//" + file_name + ".npz")
    S_array = data["S_array"]

    plt.figure(figsize=(10, 6))
    for i, lbd in enumerate(lbd_array):
        plt.plot(np.arange(1, steps+1e-3), S_array[i, :], marker=".", linewidth=2, label=r"$\lambda=%.2f$" % (lbd))
    
    plt.title(state_name)
    plt.legend()
    plt.xlabel(rf"$steps/{dt}$")
    plt.xlim(1, steps)
    plt.ylabel(r"$S$")
    plt.tight_layout()
    plt.savefig("fig//" + file_name + ".png", dpi=300, bbox_inches="tight")



if __name__ == "__main__":
    np.random.seed(100)
    state_name = "bipartite_state"
    f = 2
    L = 200
    t = 1
    lbd_array = np.concatenate((np.arange(0, 0.5, 0.25), np.arange(0.5, 1.5, 0.1), np.arange(1.5, 2.0+1e-3, 0.25)))
    a = 0.3
    b = 2 / (np.sqrt(5) - 1)
    phi = 0
    steps = 250
    dt = 100

    start_time = time.time()
    cal_S_and_N_of_t(state_name, L, t, lbd_array, a, b, phi, steps, dt, f)
    vis_S_of_t(L, lbd_array, steps, dt, a, f)
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time:.1f} s")



