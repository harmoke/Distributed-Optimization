import time, pickle, os
import numpy as np
from scipy.optimize import minimize
from sklearn import datasets, preprocessing

def ReLU(z):
	return max(0, z)

def cost_sm(w, X, Y): # sm : soft margin
	N = len(Y)
	sm = 0.0
	for i in range(N):
		z = 1.0 - Y[i] * np.inner(w, X[i])
		sm += ReLU(z)
	sm = sm / N * 0.1 + 0.5 * np.inner(w, w)
	return sm

def generate_data():
	print("Start data generation.")
	data = datasets.load_breast_cancer()
	X, Y = datasets.make_classification(n_samples = 10000, n_features = 100, n_classes = 2, random_state = 514)
	Y = 2 * Y - 1
	print("Data generation completed.")
	
	print("Start standard solution computation.")
	dim = len(X[0])
	w = np.zeros(dim, dtype = float)
	bnds = ((0, None) for _ in range(100))
	t1 = time.time()
	res = minimize(cost_sm, x0 = w, args = (X, Y), method = 'L-BFGS-B', tol = 1e-8, bounds = bnds)
	t2 = time.time()
	t_std = t2 - t1
	print(res)
	print(f"Standard solution computed using {t_std} seconds.")
	
	print("Start data files creation.")
	os.system("mkdir Data/Prob")
	pickle.dump(X, open("Data/Prob/X.txt", "wb"))
	pickle.dump(Y, open("Data/Prob/Y.txt", "wb"))
	os.system("mkdir Data/Sol")
	pickle.dump(res.x, open("Data/Sol/w_std.txt", "wb"))
	f = open("Data/Sol/t_std.txt", "w")
	f.write(f"{t_std}")
	f.close()
	print("Data files created.")

if __name__ == "__main__":
	generate_data()