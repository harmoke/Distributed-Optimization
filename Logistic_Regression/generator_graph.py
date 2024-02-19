import networkx as nx
import pickle, os, sys

def build_graph(N, type):
	if type == "cycle":
		G = nx.cycle_graph(N)
	elif type == "complete":
		G = nx.complete_graph(N)
	elif type == "random":
		G = nx.erdos_renyi_graph(N, 0.5, seed = 514, directed = False)
	
	print(f"Start graph files creation with N = {N} and type = {type}")
	os.mkdir(f"Graph/N={N}/{type}")
	for i in range(N):
		pickle.dump(G.adj[i], open(f"Graph/N={N}/{type}/neighbor_of_{i}.txt", "wb"))
	print("Graph files created.")

if __name__ == "__main__":
	number_of_nodes = int(sys.argv[1])
	graph_type = sys.argv[2]
	build_graph(number_of_nodes, graph_type)