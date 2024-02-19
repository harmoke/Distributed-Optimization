import os, sys

def build_graph(path):
	os.system("mkdir Graph")
	N = 5
	os.system(f"mkdir Graph/N={N}")
	os.system("%s generator_graph.py %d %s" %(path, N, "cycle"))
	os.system("%s generator_graph.py %d %s" %(path, N, "complete"))

def generate_data(path):
	os.system(f"{path} generator_data.py")

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
		generate_data(path)
	
	# log
	log_isExist = os.path.exists("Log")
	if log_isExist == False:
		os.system("mkdir Log")
	
	# result
	res_isExist = os.path.exists("Res")
	if res_isExist == False:
		os.system("mkdir Res")
	
	# run
	toFile = True
#	N = sys.argv[1]
#	type = sys.argv[2]
	N = 5
	type = "complete"
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