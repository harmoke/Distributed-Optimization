import numpy as np
import networkx as nx
import pickle, os, time, sys

def allocate_data(N):
	print("Start data loading.")
	X = np.array(pickle.load(open("Data/Prob/X.txt", "rb")))
	Y = np.array(pickle.load(open("Data/Prob/Y.txt", "rb")))
	print("Data loading completed.")
	
	print("Start data partition.")
	X_block = np.vsplit(X, N)
	Y_block = np.hsplit(Y, N)
	print("Data partition completed.")
	
	print("Start data allocation.")
	os.system(f"mkdir Data/N={N}")
	for i in range(N):
		pickle.dump(X_block[i], open(f"Data/N={N}/X_allocated_{i}.txt", "wb"))
		pickle.dump(Y_block[i], open(f"Data/N={N}/Y_allocated_{i}.txt", "wb"))
	print("Data allocation completed.")

if __name__ == "__main__":
	number_of_nodes = int(sys.argv[1])
	allocate_data(number_of_nodes)