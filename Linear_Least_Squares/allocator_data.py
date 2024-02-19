import numpy as np
import networkx as nx
import pickle, os, time, sys

def allocate_data(N):
	print("Start data loading.")
	Q = np.array(pickle.load(open("Data/Prob/Q.txt", "rb")))
	y = np.array(pickle.load(open("Data/Prob/y.txt", "rb")))
	print("Data loading completed.")
	
	print("Start data partition.")
	Q_block = np.vsplit(Q, N)
	y_block = np.hsplit(y, N)
	print("Data partition completed.")
	
	print("Start data allocation.")
	os.system(f"mkdir Data/N={N}")
	for i in range(N):
		pickle.dump(Q_block[i], open(f"Data/N={N}/Q_allocated_{i}.txt", "wb"))
		pickle.dump(y_block[i], open(f"Data/N={N}/y_allocated_{i}.txt", "wb"))
	print("Data allocation completed.")

if __name__ == "__main__":
	number_of_nodes = int(sys.argv[1])
	allocate_data(number_of_nodes)