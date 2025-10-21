import numpy as np
a = np.random.rand(5000, 5000)
e, eval = np.linalg.eig(a)
print("finished")
print(len(a))