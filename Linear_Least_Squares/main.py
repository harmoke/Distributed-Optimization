import os, sys

def build_graph(path):
	os.system("mkdir Graph")
	for i in range(2, 11):
		os.system(f"mkdir Graph/N={i}")
		os.system("%s generator_graph.py %d %s" %(path, i, "cycle"))
		os.system("%s generator_graph.py %d %s" %(path, i, "complete"))
		os.system("%s generator_graph.py %d %s" %(path, i, "random"))

def generate_data(path, n, m):
	os.system(f"{path} generator_data.py {n} {m}")

if __name__ == "__main__":
	# python path
	path = "python"
	
	# graph
	graph_isExist = os.path.exists("Graph")
	if graph_isExist == False:
		build_graph(path)
	
	# data
	data_isExist = os.path.exists("Data")
	if data_isExist == False:
		os.system("mkdir Data")
		n = 63000
		m = 4000
		generate_data(path, n, m)
	
	# log
	log_isExist = os.path.exists("Log")
	if log_isExist == False:
		os.system("mkdir Log")
	
	# result
	res_isExist = os.path.exists("Res")
	if res_isExist == False:
		os.system("mkdir Res")
	
	# run
	toFile = False
	N = sys.argv[1]
	type = sys.argv[2]
#	N = 10
#	type = "cycle"
	os.system(f"{path} allocator_data.py {N}")
	print("PPCM begins to work.")
	if toFile:
		os.system(f"mpirun -n {N} {path} PPCM.py {type} | tee Log/PPCM_N={N}_{type}.txt")
	else:
		os.system(f"mpirun -n {N} {path} PPCM.py {type}")
	print("WAGM begins to work.")
	if toFile:
		os.system(f"mpirun -n {N} {path} WAGM.py {type} | tee Log/WAGM_N={N}_{type}.txt")
	else:
		os.system(f"mpirun -n {N} {path} WAGM.py {type}")
	
	# clean
	os.system(f"rm -rf Data/N={N}")