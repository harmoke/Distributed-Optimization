import numpy as np
import pickle
import sys, time

m = int(sys.argv[1])
n = int(sys.argv[2])

B = np.random.randn(m, n)
b = np.random.randn(m)

pickle.dump(B, open("matrix_B.txt", "wb"))
pickle.dump(b, open("vector_b.txt", "wb"))

print("Data generation completed.")

t_start = time.time()
x_std = np.linalg.lstsq(B, b, rcond=None)[0]
t_end = time.time()
t_std = t_end - t_start
pickle.dump(x_std, open("solution_x.txt", "wb"))
f = open("standard_t.txt", "w")
f.write(f"{t_std}")
f.close()

print(f"Standard solution computed using {t_std}s.")