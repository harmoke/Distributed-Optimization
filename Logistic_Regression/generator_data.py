import time, pickle, os
import numpy as np
from scipy.optimize import minimize
from sklearn import datasets, preprocessing

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

def cost_ll(w, X, Y): # ll : logistic likelihood
	N = len(Y)
	ll = 0.0
	for i in range(N):
		z = np.inner(w, X[i])
		ll -= Y[i] * np.log(sigmoid(z)) + (1 - Y[i]) * np.log(1 - sigmoid(z))
	return ll / N

def generate_data():
	print("Start data generation.")
	data = datasets.load_breast_cancer()
	X, Y = datasets.make_classification(n_samples = 5000, n_features = 25, n_classes = 2, random_state = 514)
	print("Data generation completed.")
	
	print("Start standard solution computation.")
	dim = len(X[0])
	w = np.zeros(dim, dtype = float)
	t1 = time.time()
	res = minimize(cost_ll, x0 = w, args = (X, Y))
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