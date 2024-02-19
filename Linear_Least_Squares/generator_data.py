import numpy as np
import networkx as nx
import pickle, os, time, sys
from scipy.linalg import lstsq

def generate_data(n, m):
	print("Start data generation.")
	Q = np.random.randn(n, m)
	y = np.random.randn(n)
	print("Data generation completed.")
	
	print("Start standard solution computation.")
	t1 = time.time()
#	x_std = lstsq(Q, y, lapack_driver = 'gelsy')[0]
	x_std = np.linalg.lstsq(Q, y, rcond = None)[0]
	t2 = time.time()
	t_std = t2 - t1
	print(f"Standard solution computed using {t_std} seconds.")
	
	print("Start data files creation.")
	os.system("mkdir Data/Prob")
	pickle.dump(Q, open("Data/Prob/Q.txt", "wb"))
	pickle.dump(y, open("Data/Prob/y.txt", "wb"))
	os.system("mkdir Data/Sol")
	pickle.dump(x_std, open("Data/Sol/x_std.txt", "wb"))
	f = open("Data/Sol/t_std.txt", "w")
	f.write(f"{t_std}")
	f.close()
	print("Data files created.")

if __name__ == "__main__":
	matrix_height = int(sys.argv[1])
	matrix_width = int(sys.argv[2])
	generate_data(matrix_height, matrix_width)