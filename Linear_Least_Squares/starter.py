import os

#os.system("rm -rf Data")

path = "python"
type = ["cycle", "complete", "random"]

for N in range(2, 11):
	for i in range(2):
		os.system(f"{path} main.py {N} {type[i]}")