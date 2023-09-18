import os
path = "/Users/apple/anaconda3/bin/python3"
#path = "python"
m = 120
n = 45
os.system("%s Data_Generator.py %d %d" %(path, m, n))
os.system("mkdir Logs")
P = [2, 4, 6, 8, 10]
for p in P:
	print("begin with p=%d." %(p))
	os.system("mkdir Logs/p=%d" %(p))
	os.system("%s PPCM.py %d > Logs/p=%d/log_PPCM.txt" %(path, p, p))
	print("PPCM is finished.")
	os.system("%s WAGM.py %d > Logs/p=%d/log_WAGM.txt" %(path, p, p))
	print("WAGM is finished.")
	print("done with p=%d." %(p))